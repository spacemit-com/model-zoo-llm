/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "openai_backend.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <thread>
#include <mutex>
#include <vector>

#ifdef USE_HTTP_CLIENT
#include <curl/curl.h>
#include <map>
#include <nlohmann/json.hpp>
#endif

namespace spacemit_llm {

#ifdef USE_HTTP_CLIENT

// =============================================================================
// Reasoning content filtering
// =============================================================================

static size_t suffix_prefix_len(const std::string& text, const std::string& pattern) {
    const size_t max_len = std::min(text.size(), pattern.size() - 1);
    for (size_t len = max_len; len > 0; --len) {
        if (text.compare(text.size() - len, len, pattern, 0, len) == 0) {
            return len;
        }
    }
    return 0;
}

struct ThinkBlockFilter {
    bool in_think = false;
    std::string pending;

    std::string filter(const std::string& chunk) {
        static const std::string kOpen = "<think>";
        static const std::string kClose = "</think>";

        std::string data = pending + chunk;
        pending.clear();

        std::string out;
        size_t pos = 0;
        while (pos < data.size()) {
            if (in_think) {
                size_t close = data.find(kClose, pos);
                if (close == std::string::npos) {
                    size_t keep = suffix_prefix_len(data.substr(pos), kClose);
                    if (keep > 0) {
                        pending = data.substr(data.size() - keep);
                    }
                    return out;
                }
                pos = close + kClose.size();
                in_think = false;
                continue;
            }

            size_t open = data.find(kOpen, pos);
            if (open == std::string::npos) {
                std::string tail = data.substr(pos);
                size_t keep = suffix_prefix_len(tail, kOpen);
                out += tail.substr(0, tail.size() - keep);
                if (keep > 0) {
                    pending = tail.substr(tail.size() - keep);
                }
                return out;
            }

            out += data.substr(pos, open - pos);
            pos = open + kOpen.size();
            in_think = true;
        }

        return out;
    }

    std::string flush() {
        std::string out;
        if (!in_think) {
            out = pending;
        }
        pending.clear();
        return out;
    }
};

static std::string strip_think_blocks(const std::string& text) {
    ThinkBlockFilter filter;
    std::string out = filter.filter(text);
    out += filter.flush();
    return out;
}

// =============================================================================
// curl_global_init with call_once
// =============================================================================

static std::once_flag g_curl_init_flag;

static void ensure_curl_init() {
    std::call_once(g_curl_init_flag, []() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
    });
}

// =============================================================================
// Basic write callback (for sync requests)
// =============================================================================

static size_t WriteCallback(
    void* contents, size_t size, size_t nmemb, std::string* data) {
    data->append(static_cast<char*>(contents), size * nmemb);
    return size * nmemb;
}

// =============================================================================
// Stream context (single-turn streaming with cancellation)
// =============================================================================

struct StreamData {
    std::function<bool(
        const std::string& chunk, bool is_finished,
        const std::string& error)> callback;
    std::string buffer;
    bool cancelled = false;
    bool finished = false;
    bool content_delivered = false;  // true if any content was passed to callback
    bool emit_reasoning = true;
    ThinkBlockFilter think_filter;
};

static bool EmitStreamContent(StreamData* sd, const std::string& content) {
    std::string visible = sd->emit_reasoning ? content : sd->think_filter.filter(content);
    if (visible.empty()) return true;
    sd->content_delivered = true;
    return sd->callback(visible, false, "");
}

static bool FlushStreamContent(StreamData* sd) {
    if (sd->emit_reasoning) return true;
    return EmitStreamContent(sd, sd->think_filter.flush());
}

static size_t StreamWriteCallback(
    void* contents, size_t size, size_t nmemb, StreamData* sd) {
    if (sd->cancelled || sd->finished) return 0;

    size_t total_size = size * nmemb;
    sd->buffer.append(static_cast<const char*>(contents), total_size);

    // Parse SSE format: "data: {...}\n" or "data: {...}\r\n"
    size_t pos = 0;
    while (pos < sd->buffer.length()) {
        size_t data_pos = sd->buffer.find("data: ", pos);
        if (data_pos == std::string::npos) {
            sd->buffer = sd->buffer.substr(pos);
            break;
        }

        size_t line_end = sd->buffer.find('\n', data_pos);
        if (line_end == std::string::npos) {
            sd->buffer = sd->buffer.substr(data_pos);
            break;
        }

        std::string data_line = sd->buffer.substr(
            data_pos + 6, line_end - data_pos - 6);
        while (!data_line.empty() && data_line.back() == '\r') {
            data_line.pop_back();
        }

        if (data_line == "[DONE]") {
            if (!FlushStreamContent(sd)) {
                sd->cancelled = true;
                return 0;
            }
            sd->finished = true;
            sd->callback("", true, "");
            sd->buffer.clear();
            return total_size;
        }

        try {
            auto json_chunk = nlohmann::json::parse(data_line);

            if (json_chunk.contains("choices") &&
                json_chunk["choices"].is_array() &&
                !json_chunk["choices"].empty()) {
                auto& choice = json_chunk["choices"][0];

                if (choice.contains("delta") && choice["delta"].is_object()) {
                    auto& delta = choice["delta"];
                    if (delta.contains("content") &&
                        !delta["content"].is_null() &&
                        delta["content"].is_string()) {
                        std::string s = delta["content"].get<std::string>();
                        if (!EmitStreamContent(sd, s)) {
                            sd->cancelled = true;
                            return 0;
                        }
                    }
                    if (sd->emit_reasoning) {
                        for (const char* key : {"reasoning_content", "thinking"}) {
                            if (delta.contains(key) && !delta[key].is_null() &&
                                delta[key].is_string()) {
                                std::string s = delta[key].get<std::string>();
                                if (!EmitStreamContent(sd, s)) {
                                    sd->cancelled = true;
                                    return 0;
                                }
                            }
                        }
                    }
                }

                // Some backends (e.g. DeepSeek R1) send message.content in stream
                if (choice.contains("message") && choice["message"].is_object() &&
                    choice["message"].contains("content") &&
                    !choice["message"]["content"].is_null() &&
                    choice["message"]["content"].is_string()) {
                    std::string s = choice["message"]["content"].get<std::string>();
                    if (!EmitStreamContent(sd, s)) {
                        sd->cancelled = true;
                        return 0;
                    }
                }

                if (choice.contains("finish_reason") &&
                    !choice["finish_reason"].is_null()) {
                    if (!FlushStreamContent(sd)) {
                        sd->cancelled = true;
                        return 0;
                    }
                    sd->finished = true;
                    sd->callback("", true, "");
                    sd->buffer.clear();
                    return total_size;
                }
            }

            if (json_chunk.contains("error")) {
                std::string error_msg =
                    json_chunk["error"].contains("message")
                    ? json_chunk["error"]["message"]
                        .get<std::string>()
                    : "Unknown error";
                sd->finished = true;
                sd->callback("", true, error_msg);
                sd->buffer.clear();
                return total_size;
            }
        } catch (...) {
            // skip malformed chunk
        }

        pos = line_end + 1;
    }

    return total_size;
}

static int StreamProgressCallback(
    void* clientp, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    auto* sd = static_cast<StreamData*>(clientp);
    return sd->cancelled ? 1 : 0;
}

// =============================================================================
// Chat stream context (multi-turn with tool_calls accumulation)
// =============================================================================

struct ChatStreamData {
    std::function<bool(
        const std::string& chunk, bool is_done,
        const std::string& error)> callback;
    std::string buffer;
    std::string error;
    bool cancelled = false;
    bool finished = false;
    bool emit_reasoning = true;
    ThinkBlockFilter think_filter;

    // Accumulated results
    std::string accumulated_content;
    std::string accumulated_reasoning_content;

    struct ToolCallAcc {
        std::string id;
        std::string type;
        std::string name;
        std::string arguments;
    };
    std::map<int, ToolCallAcc> tool_calls;
};

static bool EmitChatContent(ChatStreamData* sd, const std::string& content) {
    std::string visible = sd->emit_reasoning ? content : sd->think_filter.filter(content);
    if (visible.empty()) return true;
    sd->accumulated_content += visible;
    return sd->callback(visible, false, "");
}

static bool FlushChatContent(ChatStreamData* sd) {
    if (sd->emit_reasoning) return true;
    return EmitChatContent(sd, sd->think_filter.flush());
}

static size_t ChatStreamWriteCallback(
    void* contents, size_t size, size_t nmemb,
    ChatStreamData* sd) {
    if (sd->cancelled || sd->finished) return 0;

    size_t total_size = size * nmemb;
    sd->buffer.append(static_cast<const char*>(contents), total_size);

    size_t pos = 0;
    while (pos < sd->buffer.length()) {
        size_t data_pos = sd->buffer.find("data: ", pos);
        if (data_pos == std::string::npos) {
            sd->buffer = sd->buffer.substr(pos);
            break;
        }

        size_t line_end = sd->buffer.find('\n', data_pos);
        if (line_end == std::string::npos) {
            sd->buffer = sd->buffer.substr(data_pos);
            break;
        }

        std::string data_line = sd->buffer.substr(
            data_pos + 6, line_end - data_pos - 6);
        while (!data_line.empty() && data_line.back() == '\r') {
            data_line.pop_back();
        }

        if (data_line == "[DONE]") {
            if (!FlushChatContent(sd)) {
                sd->cancelled = true;
                return 0;
            }
            sd->finished = true;
            sd->callback("", true, "");
            sd->buffer.clear();
            return total_size;
        }

        try {
            auto json_chunk = nlohmann::json::parse(data_line);

            if (json_chunk.contains("choices") &&
                json_chunk["choices"].is_array() &&
                !json_chunk["choices"].empty()) {
                auto& choice = json_chunk["choices"][0];

                if (choice.contains("delta")) {
                    auto& delta = choice["delta"];

                    // Content
                    if (delta.contains("content") &&
                        !delta["content"].is_null() &&
                        delta["content"].is_string()) {
                        std::string content =
                            delta["content"].get<std::string>();
                        if (!content.empty()) {
                            if (!EmitChatContent(sd, content)) {
                                sd->cancelled = true;
                                return 0;
                            }
                        }
                    }

                    if (delta.contains("reasoning_content") &&
                        !delta["reasoning_content"].is_null() &&
                        delta["reasoning_content"].is_string()) {
                        std::string reasoning =
                            delta["reasoning_content"].get<std::string>();
                        if (!reasoning.empty()) {
                            sd->accumulated_reasoning_content += reasoning;
                            if (sd->emit_reasoning) {
                                bool cont = sd->callback(reasoning, false, "");
                                if (!cont) {
                                    sd->cancelled = true;
                                    return 0;
                                }
                            }
                        }
                    }
                    if (delta.contains("thinking") &&
                        !delta["thinking"].is_null() &&
                        delta["thinking"].is_string()) {
                        std::string thinking =
                            delta["thinking"].get<std::string>();
                        if (!thinking.empty()) {
                            sd->accumulated_reasoning_content += thinking;
                            if (sd->emit_reasoning) {
                                bool cont = sd->callback(thinking, false, "");
                                if (!cont) {
                                    sd->cancelled = true;
                                    return 0;
                                }
                            }
                        }
                    }

                    // Tool calls (delta accumulation)
                    if (delta.contains("tool_calls") &&
                        delta["tool_calls"].is_array()) {
                        for (const auto& tc : delta["tool_calls"]) {
                            int idx = tc.value("index", 0);
                            auto& acc = sd->tool_calls[idx];
                            if (tc.contains("id"))
                                acc.id = tc["id"].get<std::string>();
                            if (tc.contains("type"))
                                acc.type = tc["type"].get<std::string>();
                            if (tc.contains("function")) {
                                auto& func = tc["function"];
                                if (func.contains("name") &&
                                    !func["name"].is_null())
                                    acc.name +=
                                        func["name"].get<std::string>();
                                if (func.contains("arguments") &&
                                    !func["arguments"].is_null())
                                    acc.arguments +=
                                        func["arguments"]
                                            .get<std::string>();
                            }
                        }
                    }
                }

                if (choice.contains("finish_reason") &&
                    !choice["finish_reason"].is_null()) {
                    if (!FlushChatContent(sd)) {
                        sd->cancelled = true;
                        return 0;
                    }
                    sd->finished = true;
                    sd->callback("", true, "");
                    sd->buffer.clear();
                    return total_size;
                }
            }

            if (json_chunk.contains("error")) {
                std::string error_msg =
                    json_chunk["error"].contains("message")
                    ? json_chunk["error"]["message"]
                        .get<std::string>()
                    : "Unknown error";
                sd->error = error_msg;
                sd->finished = true;
                sd->callback("", true, error_msg);
                sd->buffer.clear();
                return total_size;
            }
        } catch (...) {
            // skip malformed chunk
        }

        pos = line_end + 1;
    }

    return total_size;
}

static int ChatStreamProgressCallback(
    void* clientp, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    auto* sd = static_cast<ChatStreamData*>(clientp);
    return sd->cancelled ? 1 : 0;
}

// =============================================================================
// Helper: build curl headers
// =============================================================================

static struct curl_slist* make_headers(
    const std::string& api_key, bool sse = false) {
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    if (sse) {
        headers = curl_slist_append(headers, "Accept: text/event-stream");
    }
    if (!api_key.empty() && api_key != "EMPTY") {
        std::string auth = "Authorization: Bearer " + api_key;
        headers = curl_slist_append(headers, auth.c_str());
    }
    return headers;
}

// =============================================================================
// Helper: build messages JSON array from LLMService::ChatMessage vector
// =============================================================================

static nlohmann::json message_to_json(const LLMService::ChatMessage& msg) {
    nlohmann::json m;
    switch (msg.role) {
        case LLMService::ChatMessage::Role::SYSTEM:    m["role"] = "system"; break;
        case LLMService::ChatMessage::Role::USER:      m["role"] = "user"; break;
        case LLMService::ChatMessage::Role::ASSISTANT: m["role"] = "assistant"; break;
        case LLMService::ChatMessage::Role::TOOL:      m["role"] = "tool"; break;
    }
    bool has_tool_calls = msg.role == LLMService::ChatMessage::Role::ASSISTANT &&
        !msg.tool_calls_json.empty() && msg.tool_calls_json != "[]";
    bool has_reasoning = msg.role == LLMService::ChatMessage::Role::ASSISTANT &&
        !msg.reasoning_content.empty();
    if (msg.content.empty() && (has_tool_calls || has_reasoning)) {
        m["content"] = nullptr;
    } else {
        m["content"] = msg.content;
    }
    if (msg.role == LLMService::ChatMessage::Role::ASSISTANT &&
        !msg.reasoning_content.empty()) {
        m["reasoning_content"] = msg.reasoning_content;
    }

    // Assistant with tool_calls
    if (msg.role == LLMService::ChatMessage::Role::ASSISTANT &&
        !msg.tool_calls_json.empty()) {
        try {
            m["tool_calls"] =
                nlohmann::json::parse(msg.tool_calls_json);
        } catch (...) {}
    }

    // Tool response with tool_call_id
    if (msg.role == LLMService::ChatMessage::Role::TOOL &&
        !msg.tool_call_id.empty()) {
        m["tool_call_id"] = msg.tool_call_id;
    }

    return m;
}

static nlohmann::json messages_to_json(
    const std::vector<LLMService::ChatMessage>& messages) {
    nlohmann::json arr = nlohmann::json::array();

    size_t i = 0;
    std::string merged_system;
    while (i < messages.size() &&
            messages[i].role == LLMService::ChatMessage::Role::SYSTEM) {
        if (messages[i].content.empty()) {
            ++i;
            continue;
        }
        if (!merged_system.empty()) {
            merged_system += "\n\n";
        }
        merged_system += messages[i].content;
        ++i;
    }
    if (!merged_system.empty()) {
        arr.push_back({{"role", "system"}, {"content", merged_system}});
    }
    for (; i < messages.size(); ++i) {
        arr.push_back(message_to_json(messages[i]));
    }
    return arr;
}

// =============================================================================
// HTTP POST (sync)
// =============================================================================

static std::string http_post(
    const std::string& url, const std::string& json_data,
    const std::string& api_key) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize CURL");
    }

    std::string response_data;
    struct curl_slist* headers = make_headers(api_key);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1L);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 60L);

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);

    if (res != CURLE_OK) {
        curl_easy_cleanup(curl);
        throw std::runtime_error(
            "CURL error: " + std::string(curl_easy_strerror(res)));
    }

    long response_code = 0;  // NOLINT(runtime/int)
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    curl_easy_cleanup(curl);

    if (response_code != 200) {
        throw std::runtime_error(
            "HTTP error: " + std::to_string(response_code)
            + ", response: " + response_data);
    }

    return response_data;
}

// =============================================================================
// HTTP POST streaming (single-turn, with cancellation support)
// =============================================================================

static void http_post_stream(
    const std::string& url,
    const std::string& json_data,
    const std::string& api_key,
    StreamData& stream_data
) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        stream_data.callback("", true, "Failed to initialize CURL");
        return;
    }

    struct curl_slist* headers = make_headers(api_key, true);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, StreamWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream_data);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1L);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 60L);

    // Enable progress callback for mid-transfer cancellation
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION,
        StreamProgressCallback);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &stream_data);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);

    CURLcode res = curl_easy_perform(curl);
    long response_code = 0;  // NOLINT(runtime/int)
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    // If we requested stream but got no SSE content (e.g. server returned
    // a single JSON body), try to parse as non-streaming response.
    if (res == CURLE_OK && response_code == 200 && !stream_data.content_delivered &&
        !stream_data.buffer.empty() && !stream_data.cancelled) {
        const std::string& body = stream_data.buffer;
        if (body.size() > 1 && body[0] == '{' &&
            body.find("data: ") == std::string::npos) {
            try {
                auto j = nlohmann::json::parse(body);
                if (j.contains("choices") && j["choices"].is_array() &&
                    !j["choices"].empty()) {
                    auto& c = j["choices"][0];
                    std::string content;
                    if (c.contains("message") && c["message"].contains("content") &&
                        !c["message"]["content"].is_null()) {
                        content = c["message"]["content"].get<std::string>();
                        if (!stream_data.emit_reasoning) {
                            content = strip_think_blocks(content);
                        }
                    } else if (c.contains("delta") && c["delta"].is_object()) {
                        for (const char* key : {"content", "reasoning_content", "thinking"}) {
                            if (!stream_data.emit_reasoning &&
                                std::string(key) != "content") {
                                continue;
                            }
                            if (c["delta"].contains(key) && !c["delta"][key].is_null() &&
                                c["delta"][key].is_string()) {
                                content += c["delta"][key].get<std::string>();
                            }
                        }
                        if (!stream_data.emit_reasoning) {
                            content = strip_think_blocks(content);
                        }
                    }
                    if (!content.empty()) {
                        stream_data.content_delivered = true;
                        stream_data.callback(content, false, "");
                    }
                }
            } catch (...) {
                // ignore parse errors
            }
        }
    }

    // Ensure final callback if not already sent
    if (!stream_data.finished) {
        if (stream_data.cancelled) {
            stream_data.callback("", true, "cancelled");
        } else if (res != CURLE_OK) {
            stream_data.callback("", true,
                "CURL error: " + std::string(curl_easy_strerror(res)));
        } else if (response_code != 200) {
            stream_data.callback("", true,
                "HTTP error: " + std::to_string(response_code));
        } else {
            stream_data.callback("", true, "");
        }
    }
}

// =============================================================================
// HTTP POST chat streaming (multi-turn, with tool_calls + cancellation)
// =============================================================================

static LLMService::ChatResult http_post_chat_stream(
    const std::string& url,
    const std::string& json_data,
    const std::string& api_key,
    ChatStreamData& stream_data
) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        stream_data.callback("", true, "Failed to initialize CURL");
        return LLMService::ChatResult{"", "", false, "Failed to initialize CURL", ""};
    }

    struct curl_slist* headers = make_headers(api_key, true);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,
        ChatStreamWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream_data);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1L);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 60L);

    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION,
        ChatStreamProgressCallback);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &stream_data);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);

    CURLcode res = curl_easy_perform(curl);
    long response_code = 0;  // NOLINT(runtime/int)
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    // Ensure final callback if not already sent
    if (!stream_data.finished) {
        if (stream_data.cancelled) {
            stream_data.error = "cancelled";
            stream_data.callback("", true, stream_data.error);
        } else if (res != CURLE_OK) {
            stream_data.error =
                "CURL error: " + std::string(curl_easy_strerror(res));
            stream_data.callback("", true, stream_data.error);
        } else if (response_code != 200) {
            stream_data.error = "HTTP error: " + std::to_string(response_code);
            if (!stream_data.buffer.empty()) {
                std::string body = stream_data.buffer;
                constexpr size_t kMaxErrorBody = 512;
                if (body.size() > kMaxErrorBody) {
                    body = body.substr(0, kMaxErrorBody) + "...";
                }
                stream_data.error += ", response: " + body;
            }
            stream_data.callback("", true, stream_data.error);
        } else {
            stream_data.callback("", true, "");
        }
    }

    // Build LLMService::ChatResult
    LLMService::ChatResult result;
    result.content = stream_data.accumulated_content;
    result.cancelled = stream_data.cancelled;
    result.error = stream_data.error;
    result.reasoning_content = stream_data.accumulated_reasoning_content;

    if (!stream_data.tool_calls.empty()) {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& [idx, tc] : stream_data.tool_calls) {
            std::string tool_call_id = tc.id;
            if (tool_call_id.empty()) {
                tool_call_id = "call_" + std::to_string(idx);
            }
            arr.push_back({
                {"id", tool_call_id},
                {"type", tc.type.empty() ? "function" : tc.type},
                {"function", {
                    {"name", tc.name},
                    {"arguments", tc.arguments}
                }}
            });
        }
        result.tool_calls_json = arr.dump();
    }

    return result;
}

#endif  // USE_HTTP_CLIENT

// =============================================================================
// OpenAIBackend::Impl
// =============================================================================

struct OpenAIBackend::Impl {
    std::string model;
    std::string api_base;
    std::string api_key;
    int reasoning_budget = -1;
    bool initialized = false;

    std::thread worker_thread;
    std::mutex thread_mutex;

    void join_worker() {
        std::lock_guard<std::mutex> lock(thread_mutex);
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }

    ~Impl() {
        join_worker();
    }
};

// =============================================================================
// Constructor / Destructor / Initialize
// =============================================================================

OpenAIBackend::OpenAIBackend(const BackendConfig& config)
    : pimpl_(std::make_unique<Impl>()) {
    pimpl_->model = config.openai.model;
    pimpl_->api_base = config.openai.api_base;
    pimpl_->api_key = config.openai.api_key;
    pimpl_->reasoning_budget = config.openai.reasoning_budget;
}

OpenAIBackend::~OpenAIBackend() = default;

bool OpenAIBackend::initialize() {
#ifdef USE_HTTP_CLIENT
    ensure_curl_init();
    pimpl_->initialized = true;
    return true;
#else
    pimpl_->initialized = false;
    return false;
#endif
}

// =============================================================================
// complete (sync, single-turn)
// =============================================================================

std::string OpenAIBackend::complete(
    const std::string& user_text, const std::string& prompt,
    int max_tokens) {
#ifdef USE_HTTP_CLIENT
    if (!pimpl_->initialized) {
        initialize();
    }

    nlohmann::json request;
    request["model"] = pimpl_->model;
    request["max_tokens"] = max_tokens;
    request["temperature"] = 0.7;

    nlohmann::json messages = nlohmann::json::array();
    if (!prompt.empty()) {
        messages.push_back({{"role", "system"}, {"content", prompt}});
    }
    messages.push_back({{"role", "user"}, {"content", user_text}});
    request["messages"] = messages;

    std::string url = pimpl_->api_base + "/chat/completions";
    std::string response = http_post(
        url, request.dump(), pimpl_->api_key);

    auto json_response = nlohmann::json::parse(response);
    std::string content = json_response["choices"][0]["message"]["content"]
        .get<std::string>();
    if (pimpl_->reasoning_budget == 0) {
        content = strip_think_blocks(content);
    }
    return content;
#else
    (void)user_text;
    (void)prompt;
    (void)max_tokens;
    throw std::runtime_error("HTTP client not compiled in");
#endif
}

// =============================================================================
// complete_async (async, single-turn) — thread held in Impl, joined before reuse
// =============================================================================

void OpenAIBackend::complete_async(
    const std::string& user_text,
    const std::string& prompt,
    int max_tokens,
    std::function<void(
        const std::string& result, const std::string& error)> callback
) {
    pimpl_->join_worker();
    pimpl_->worker_thread = std::thread(
        [this, user_text, prompt, max_tokens, callback]() {
            try {
                std::string result = complete(user_text, prompt, max_tokens);
                callback(result, "");
            } catch (const std::exception& e) {
                callback("", e.what());
            }
        });
}

// =============================================================================
// complete_stream (streaming, single-turn, with cancellation)
// =============================================================================

void OpenAIBackend::complete_stream(
    const std::string& user_text,
    const std::string& prompt,
    int max_tokens,
    std::function<bool(
        const std::string& chunk, bool is_finished,
        const std::string& error)> callback
) {
#ifdef USE_HTTP_CLIENT
    if (!pimpl_->initialized) {
        initialize();
    }

    nlohmann::json request;
    request["model"] = pimpl_->model;
    request["max_tokens"] = max_tokens;
    request["temperature"] = 0.7;
    request["stream"] = true;

    nlohmann::json messages = nlohmann::json::array();
    if (!prompt.empty()) {
        messages.push_back({{"role", "system"}, {"content", prompt}});
    }
    messages.push_back({{"role", "user"}, {"content", user_text}});
    request["messages"] = messages;

    pimpl_->join_worker();
    pimpl_->worker_thread = std::thread([this, request, callback]() {
        StreamData stream_data;
        stream_data.callback = callback;
        stream_data.emit_reasoning = pimpl_->reasoning_budget != 0;
        try {
            std::string url = pimpl_->api_base + "/chat/completions";
            http_post_stream(
                url, request.dump(), pimpl_->api_key, stream_data);
        } catch (const std::exception& e) {
            if (!stream_data.finished) {
                callback("", true, e.what());
            }
        }
    });
#else
    (void)user_text;
    (void)prompt;
    (void)max_tokens;
    callback("", true, "HTTP client not compiled in");
#endif
}

// =============================================================================
// chat (sync, multi-turn)
// =============================================================================

std::string OpenAIBackend::chat(
    const std::vector<LLMService::ChatMessage>& messages, int max_tokens) {
#ifdef USE_HTTP_CLIENT
    if (!pimpl_->initialized) {
        initialize();
    }

    nlohmann::json request;
    request["model"] = pimpl_->model;
    request["max_tokens"] = max_tokens;
    request["temperature"] = 0.7;
    request["messages"] = messages_to_json(messages);

    std::string url = pimpl_->api_base + "/chat/completions";
    std::string response = http_post(
        url, request.dump(), pimpl_->api_key);

    auto json_response = nlohmann::json::parse(response);
    std::string content = json_response["choices"][0]["message"]["content"]
        .get<std::string>();
    if (pimpl_->reasoning_budget == 0) {
        content = strip_think_blocks(content);
    }
    return content;
#else
    (void)messages;
    (void)max_tokens;
    throw std::runtime_error("HTTP client not compiled in");
#endif
}

// =============================================================================
// chat_stream (streaming, multi-turn, blocking, with tools + cancellation)
// =============================================================================

LLMService::ChatResult OpenAIBackend::chat_stream(
    const std::vector<LLMService::ChatMessage>& messages,
    std::function<bool(
        const std::string& chunk, bool is_done,
        const std::string& error)> callback,
    int max_tokens,
    const std::string& tools_json
) {
#ifdef USE_HTTP_CLIENT
    if (!pimpl_->initialized) {
        initialize();
    }

    nlohmann::json request;
    request["model"] = pimpl_->model;
    request["max_tokens"] = max_tokens;
    request["temperature"] = 0.7;
    request["stream"] = true;
    request["messages"] = messages_to_json(messages);

    // Add tools if provided
    if (!tools_json.empty() && tools_json != "[]") {
        try {
            request["tools"] = nlohmann::json::parse(tools_json);
        } catch (...) {
            // Invalid tools JSON, skip
        }
    }

    ChatStreamData stream_data;
    stream_data.callback = callback;
    stream_data.emit_reasoning = pimpl_->reasoning_budget != 0;

    std::string url = pimpl_->api_base + "/chat/completions";
    return http_post_chat_stream(
        url, request.dump(), pimpl_->api_key, stream_data);
#else
    (void)messages;
    (void)max_tokens;
    (void)tools_json;
    callback("", true, "HTTP client not compiled in");
    return LLMService::ChatResult{"", "", false, "HTTP client not compiled in", ""};
#endif
}

// =============================================================================
// update_config / is_available
// =============================================================================

void OpenAIBackend::update_config(
    const std::string& key, const std::string& value) {
    if (key == "api_base") {
        pimpl_->api_base = value;
    } else if (key == "api_key") {
        pimpl_->api_key = value;
    } else if (key == "model") {
        pimpl_->model = value;
    } else if (key == "reasoning_budget") {
        pimpl_->reasoning_budget = std::stoi(value);
    }
}

bool OpenAIBackend::is_available() const {
#ifdef USE_HTTP_CLIENT
    return true;
#else
    return false;
#endif
}

}  // namespace spacemit_llm
