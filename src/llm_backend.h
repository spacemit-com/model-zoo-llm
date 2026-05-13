/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef LLM_BACKEND_H
#define LLM_BACKEND_H

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "llm_service.h"

namespace spacemit_llm {

/**
 * LLM Backend Interface
 */
class LLMBackend {
public:
    virtual ~LLMBackend() = default;

    virtual bool initialize() = 0;

    /**
     * Single-turn completion (sync)
     */
    virtual std::string complete(
        const std::string& user_text,
        const std::string& prompt,
        int max_tokens) = 0;

    /**
     * Single-turn completion (async)
     */
    virtual void complete_async(
        const std::string& user_text,
        const std::string& prompt,
        int max_tokens,
        std::function<void(const std::string& result, const std::string& error)> callback) = 0;

    /**
     * Single-turn completion (streaming).
     * Callback returns bool: false = cancel the stream.
     */
    virtual void complete_stream(
        const std::string& user_text,
        const std::string& prompt,
        int max_tokens,
        std::function<bool(const std::string& chunk, bool is_finished, const std::string& error)> callback
    ) {
        // Default: non-streaming fallback
        complete_async(user_text, prompt, max_tokens,
            [callback](const std::string& result, const std::string& error) {
                if (!error.empty()) {
                    callback("", true, error);
                } else {
                    callback(result, true, "");
                }
            });
    }

    /**
     * Multi-turn chat (sync)
     */
    virtual std::string chat(
        const std::vector<LLMService::ChatMessage>& messages,
        int max_tokens
    ) {
        // Default: extract last user message and call complete()
        std::string user_text, prompt;
        for (const auto& msg : messages) {
            if (msg.role == LLMService::ChatMessage::Role::SYSTEM) prompt = msg.content;
            if (msg.role == LLMService::ChatMessage::Role::USER) user_text = msg.content;
        }
        return complete(user_text, prompt, max_tokens);
    }

    /**
     * Multi-turn chat (streaming, blocking)
     * Supports messages array + optional tools (OpenAI function calling format).
     * Callback returns bool: false = cancel.
     * Returns ChatResult with content and optional tool_calls_json.
     */
    virtual LLMService::ChatResult chat_stream(
        const std::vector<LLMService::ChatMessage>& messages,
        std::function<bool(const std::string& chunk, bool is_done, const std::string& error)> callback,
        int max_tokens,
        const std::string& tools_json = ""
    ) {
        // Default: fall back to sync chat
        (void)tools_json;
        LLMService::ChatResult result;
        try {
            result.content = chat(messages, max_tokens);
            callback(result.content, true, "");
        } catch (const std::exception& e) {
            result.error = e.what();
            callback("", true, result.error);
        }
        return result;
    }

    virtual void update_config(const std::string& /* key */, const std::string& /* value */) {}
    virtual std::string get_backend_name() const = 0;
    virtual bool is_available() const = 0;
};

/**
 * Backend Configuration
 */
struct BackendConfig {
    enum class Type {
        OPENAI_API,
        CUSTOM
    };

    Type type;

    struct {
        std::string model;
        std::string api_base;
        std::string api_key;
        int reasoning_budget = -1;
    } openai;

    struct {
        std::string backend_name;
        std::map<std::string, std::string> params;
    } custom;
};

}  // namespace spacemit_llm

#endif  // LLM_BACKEND_H
