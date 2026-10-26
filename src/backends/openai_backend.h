/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPENAI_BACKEND_H
#define OPENAI_BACKEND_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "llm_backend.h"

namespace spacemit_llm {

/**
 * OpenAI-compatible API backend (text-only)
 * Supports HTTP calls to OpenAI-compatible endpoints (Ollama, vLLM, SGLang, etc.)
 */
class OpenAIBackend : public LLMBackend {
public:
    explicit OpenAIBackend(const BackendConfig& config);
    ~OpenAIBackend() override;

    bool initialize() override;

    std::string complete(
        const std::string& user_text,
        const std::string& prompt,
        int max_tokens) override;

    void complete_async(
        const std::string& user_text,
        const std::string& prompt,
        int max_tokens,
        std::function<void(const std::string& result, const std::string& error)> callback) override;

    void complete_stream(
        const std::string& user_text,
        const std::string& prompt,
        int max_tokens,
        std::function<bool(const std::string& chunk, bool is_finished, const std::string& error)> callback) override;

    std::string chat(
        const std::vector<LLMService::ChatMessage>& messages,
        int max_tokens) override;

    LLMService::ChatResult chat_stream(
        const std::vector<LLMService::ChatMessage>& messages,
        std::function<bool(const std::string& chunk, bool is_done, const std::string& error)> callback,
        int max_tokens,
        const std::string& tools_json = "") override;

    void update_config(const std::string& key, const std::string& value) override;
    void set_abort_check(std::function<bool()> check) override;
    std::string get_backend_name() const override { return "OpenAI API"; }
    bool is_available() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace spacemit_llm

#endif  // OPENAI_BACKEND_H
