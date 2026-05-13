/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llm_service.h"
#include "llm_backend.h"
#include "llm_backend_factory.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spacemit_llm {

// =============================================================================
// Impl
// =============================================================================

struct LLMServiceImpl {
    std::unique_ptr<LLMBackend> backend;

    std::string prompt;
    int max_tokens;
    std::string api_base;

    LLMService::BackendType backend_type;

    // Metrics (protected by processing_mutex for compound updates)
    double last_inference_time{0.0};
    int total_inferences{0};
    double total_inference_time{0.0};
    double last_ttft_ms{0.0};  // Time to first token (streaming)
    int64_t last_output_tokens{-1};
    std::atomic<bool> is_processing{false};
    std::mutex processing_mutex;

    LLMServiceImpl(
        std::unique_ptr<LLMBackend> backend_ptr,
        LLMService::BackendType type)
        : backend(std::move(backend_ptr)), backend_type(type) {}
};

// =============================================================================
// Constructors
// =============================================================================

LLMService::LLMService(
    const std::string& model,
    const std::string& api_base,
    const std::string& api_key,
    const std::string& prompt,
    int max_tokens
) {
    BackendConfig config;
    config.type = BackendConfig::Type::OPENAI_API;
    config.openai.model = model;
    config.openai.api_base = api_base;
    config.openai.api_key = api_key;
    config.openai.reasoning_budget = -1;

    auto backend = LLMBackendFactory::create(config);
    if (!backend) {
        throw std::runtime_error("Failed to create OpenAI API backend");
    }

    pimpl_ = std::make_unique<LLMServiceImpl>(
        std::move(backend), BackendType::OPENAI_API);
    pimpl_->prompt = prompt;
    pimpl_->max_tokens = max_tokens;
    pimpl_->api_base = api_base;
}

LLMService::LLMService(
    const std::string& config_dir,
    const std::string& prompt,
    int max_tokens
) {
    BackendConfig config;
    config.type = BackendConfig::Type::CUSTOM;
    config.custom.backend_name = "custom";
    config.custom.params["config_dir"] = config_dir;

    auto backend = LLMBackendFactory::create(config);
    if (!backend) {
        throw std::runtime_error("Failed to create custom backend");
    }

    pimpl_ = std::make_unique<LLMServiceImpl>(
        std::move(backend), BackendType::CUSTOM);
    pimpl_->prompt = prompt;
    pimpl_->max_tokens = max_tokens;
}

LLMService::~LLMService() = default;

// =============================================================================
// Single-turn completion
// =============================================================================

std::string LLMService::complete(
    const std::string& user_text, const std::string& prompt) {
    std::lock_guard<std::mutex> lock(pimpl_->processing_mutex);
    pimpl_->is_processing = true;

    auto start_time = std::chrono::high_resolution_clock::now();
    std::string result;

    try {
        std::string actual_prompt =
            prompt.empty() ? pimpl_->prompt : prompt;
        result = pimpl_->backend->complete(
            user_text, actual_prompt, pimpl_->max_tokens);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();

        pimpl_->last_inference_time = static_cast<double>(duration);
        pimpl_->total_inferences++;
        pimpl_->total_inference_time += static_cast<double>(duration);
        pimpl_->last_ttft_ms = 0.0;  // N/A for non-streaming
    } catch (const std::exception& e) {
        result = "Error: " + std::string(e.what());
    }

    pimpl_->is_processing = false;
    return result;
}

void LLMService::complete_async(
    const std::string& user_text,
    std::function<void(
        const std::string& result, const std::string& error)> callback,
    const std::string& prompt
) {
    std::string actual_prompt =
        prompt.empty() ? pimpl_->prompt : prompt;

    {
        std::lock_guard<std::mutex> lock(pimpl_->processing_mutex);
        if (pimpl_->is_processing.load()) {
            callback("", "LLM is busy processing another request");
            return;
        }
        pimpl_->is_processing = true;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    auto wrapped_callback = [this, start_time, callback](
        const std::string& result, const std::string& error) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();

        {
            std::lock_guard<std::mutex> lock(pimpl_->processing_mutex);
            pimpl_->last_inference_time = static_cast<double>(duration);
            pimpl_->total_inferences++;
            pimpl_->total_inference_time +=
                static_cast<double>(duration);
            pimpl_->last_ttft_ms = 0.0;  // N/A for non-streaming
            pimpl_->is_processing = false;
        }

        callback(result, error);
    };

    pimpl_->backend->complete_async(
        user_text, actual_prompt, pimpl_->max_tokens, wrapped_callback);
}

void LLMService::complete_stream(
    const std::string& user_text,
    std::function<bool(
        const std::string& chunk, bool is_finished,
        const std::string& error)> callback,
    const std::string& prompt
) {
    std::string actual_prompt =
        prompt.empty() ? pimpl_->prompt : prompt;

    {
        std::lock_guard<std::mutex> lock(pimpl_->processing_mutex);
        if (pimpl_->is_processing.load()) {
            callback("", true,
                "LLM is busy processing another request");
            return;
        }
        pimpl_->is_processing = true;
        pimpl_->last_output_tokens = -1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    auto first_chunk_done = std::make_shared<bool>(false);
    // Count content chunks as proxy for output tokens when server sends no usage
    auto output_chunk_count = std::make_shared<int>(0);

    auto wrapped_callback = [this, start_time, first_chunk_done, output_chunk_count, callback](
        const std::string& chunk, bool is_finished,
        const std::string& error) -> bool {
        if (!chunk.empty()) {
            if (!*first_chunk_done) {
                *first_chunk_done = true;
                auto ttft_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start_time).count();
                std::lock_guard<std::mutex> lock(pimpl_->processing_mutex);
                pimpl_->last_ttft_ms = static_cast<double>(ttft_ms);
            }
            (*output_chunk_count)++;
        }
        if (is_finished) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - start_time).count();

            {
                std::lock_guard<std::mutex> lock(
                    pimpl_->processing_mutex);
                pimpl_->last_inference_time =
                    static_cast<double>(duration);
                pimpl_->total_inferences++;
                pimpl_->total_inference_time +=
                    static_cast<double>(duration);
                if (pimpl_->last_output_tokens < 0) {
                    pimpl_->last_output_tokens = *output_chunk_count;
                }
                pimpl_->is_processing = false;
            }
        }

        return callback(chunk, is_finished, error);
    };

    pimpl_->backend->complete_stream(
        user_text, actual_prompt, pimpl_->max_tokens, wrapped_callback);
}

// =============================================================================
// Multi-turn chat
// =============================================================================

std::string LLMService::chat(
    const std::vector<ChatMessage>& messages) {
    std::lock_guard<std::mutex> lock(pimpl_->processing_mutex);
    pimpl_->is_processing = true;

    auto start_time = std::chrono::high_resolution_clock::now();
    std::string result;

    try {
        result = pimpl_->backend->chat(messages, pimpl_->max_tokens);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();

        pimpl_->last_inference_time = static_cast<double>(duration);
        pimpl_->total_inferences++;
        pimpl_->total_inference_time += static_cast<double>(duration);
        pimpl_->last_ttft_ms = 0.0;  // N/A for non-streaming
    } catch (const std::exception& e) {
        result = "Error: " + std::string(e.what());
    }

    pimpl_->is_processing = false;
    return result;
}

LLMService::ChatResult LLMService::chat_stream(
    const std::vector<ChatMessage>& messages,
    std::function<bool(
        const std::string& chunk, bool is_done,
        const std::string& error)> callback,
    const std::string& tools_json
) {
    {
        std::lock_guard<std::mutex> lock(pimpl_->processing_mutex);
        if (pimpl_->is_processing.load()) {
            callback("", true,
                "LLM is busy processing another request");
            return LLMService::ChatResult{
                "", "", false,
                "LLM is busy processing another request", ""};
        }
        pimpl_->is_processing = true;
        pimpl_->last_output_tokens = -1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    auto first_chunk_done = std::make_shared<bool>(false);
    auto output_chunk_count = std::make_shared<int>(0);

    auto wrapped_callback = [this, start_time, first_chunk_done, output_chunk_count, &callback](
        const std::string& chunk, bool is_done,
        const std::string& error) -> bool {
        if (!chunk.empty()) {
            if (!*first_chunk_done) {
                *first_chunk_done = true;
                auto ttft_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start_time).count();
                std::lock_guard<std::mutex> lock(pimpl_->processing_mutex);
                pimpl_->last_ttft_ms = static_cast<double>(ttft_ms);
            }
            (*output_chunk_count)++;
        }
        if (is_done) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - start_time).count();

            {
                std::lock_guard<std::mutex> lock(
                    pimpl_->processing_mutex);
                pimpl_->last_inference_time =
                    static_cast<double>(duration);
                pimpl_->total_inferences++;
                pimpl_->total_inference_time +=
                    static_cast<double>(duration);
                if (pimpl_->last_output_tokens < 0) {
                    pimpl_->last_output_tokens = *output_chunk_count;
                }
                pimpl_->is_processing = false;
            }
        }

        return callback(chunk, is_done, error);
    };

    LLMService::ChatResult result = pimpl_->backend->chat_stream(
        messages, wrapped_callback, pimpl_->max_tokens, tools_json);

    // Ensure is_processing is reset even if callback wasn't called with is_done
    {
        std::lock_guard<std::mutex> lock(pimpl_->processing_mutex);
        pimpl_->is_processing = false;
    }

    return result;
}

// =============================================================================
// Configuration
// =============================================================================

void LLMService::update_prompt(
    const std::string& new_prompt, int max_tokens) {
    pimpl_->prompt = new_prompt;
    if (max_tokens > 0) {
        pimpl_->max_tokens = max_tokens;
    }
}

void LLMService::update_model(const std::string& new_model) {
    pimpl_->backend->update_config("model", new_model);
}

void LLMService::update_api_settings(
    const std::string& api_base, const std::string& api_key) {
    if (pimpl_->backend_type != BackendType::OPENAI_API) {
        throw std::runtime_error(
            "update_api_settings only available in OpenAI API mode");
    }

    pimpl_->backend->update_config("api_base", api_base);
    pimpl_->backend->update_config("api_key", api_key);
    pimpl_->api_base = api_base;
}

void LLMService::update_reasoning_budget(int reasoning_budget) {
    pimpl_->backend->update_config(
        "reasoning_budget", std::to_string(reasoning_budget));
}

// =============================================================================
// Metrics
// =============================================================================

LLMService::Metrics LLMService::get_metrics() const {
    std::lock_guard<std::mutex> lock(pimpl_->processing_mutex);
    Metrics m;
    m.total_requests = pimpl_->total_inferences;
    m.is_processing = pimpl_->is_processing.load();
    m.last_latency_ms = pimpl_->last_inference_time;
    m.avg_latency_ms = (pimpl_->total_inferences > 0)
        ? pimpl_->total_inference_time / pimpl_->total_inferences : 0.0;
    m.last_ttft_ms = pimpl_->last_ttft_ms;
    m.last_output_tokens = pimpl_->last_output_tokens;
    double gen_ms = m.last_latency_ms - m.last_ttft_ms;
    if (m.last_output_tokens > 0 && gen_ms > 0) {
        m.last_tokens_per_second = m.last_output_tokens / (gen_ms / 1000.0);
    }
    return m;
}

LLMService::BackendType LLMService::get_backend_type() const {
    return pimpl_->backend_type;
}

}  // namespace spacemit_llm
