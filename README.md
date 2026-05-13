# LLM 组件

## 1. 项目简介

本组件是方便开发者快速基于进迭时空的平台体验大语言模型。同时组件基于llama.cpp引擎封装了一层通用的C++接口并提供的示例应用，方便用户基于API接口调用实现Ai Agent应用，其组件接口支持的功能特性如下：

| 类别       | 支持                                                                 |
| ---------- | -------------------------------------------------------------------- |
| 推理方式   | 单轮 `complete()`、多轮 `chat()`；同步、异步 `complete_async()`、流式 `complete_stream()` / `chat_stream()` |
| 后端       | OpenAI 兼容 HTTP API（vLLM / SGLang / llama-server 等）；可扩展自定义 Backend |
| 高级能力   | 多轮对话、Tool Calling（OpenAI function calling 格式）                |
| 可观测性   | 延迟、调用次数等 `get_metrics()`                                     |

## 2. 验证模型

按以下顺序即可从零完成环境、模型列表验证与示例运行。

### 2.1. 安装依赖

- **编译环境**：CMake ≥ 3.15，C++17 编译器（GCC/Clang/MSVC）。
- **HTTP 后端**：libcurl、nlohmann_json；**K3 等嵌入式平台**需安装推理工具 `llama.cpp-tools-spacemit`。

```bash
sudo apt-get update

# 安装 spacemit 加速版本的llama.cpp 推理工具
sudo apt install llama.cpp-tools-spacemit
```

### 2.2. 下载模型

使用 llama.cpp 时需准备 **GGUF** 格式模型。推荐统一将模型下载到默认路径 **`~/.cache/models/llm`**，便于统一管理。为了方便大家快速验证，可以访问我司的服务器下载模型。

**模型源**：

- **Spacemit 镜像（推荐）**：<https://archive.spacemit.com/spacemit-ai/model_zoo/llm/>  
  预置多种 GGUF 模型（如 Qwen2.5、Qwen3、Deepseek等），可直接下载到默认目录：

  ```bash
  mkdir -p ~/.cache/models/llm
  cd ~/.cache/models/llm
  # 示例下载qwen2.5 0.5b的模型
  wget https://archive.spacemit.com/spacemit-ai/model_zoo/llm/qwen2.5-0.5b-instruct-q4_0.gguf
  # 示例下载qwen2.5 1.5b的模型
  wget https://archive.spacemit.com/spacemit-ai/model_zoo/llm/qwen2.5-1.5b-instruct-q4_0.gguf
  # 示例下载glm-edge 1.5b的模型
  wget https://archive.spacemit.com/spacemit-ai/model_zoo/llm/glm-edge-1.5b-chat-q4_0.gguf
  # 示例下载qwen2.5 3b的模型
  wget https://archive.spacemit.com/spacemit-ai/model_zoo/llm/qwen2.5-3b-instruct-q4_0.gguf
  # 示例下载qwen3-30B-A3B的模型
  wget https://archive.spacemit.com/spacemit-ai/model_zoo/llm/Qwen3-30B-A3B-Q4_0.gguf
  ```

- **Hugging Face**：在 [Hugging Face](https://huggingface.co/models?library=gguf) 搜索 GGUF 模型，下载 `.gguf` 到 `~/.cache/models/llm` 或自定义路径。

### 2.3. 测试模型

验证对话
```bash
llama-cli -m ~/.cache/models/llm/qwen2.5-0.5b-instruct-q4_0.gguf \
  -t 4 -p "Hello, please introduce yourself."
```

验证性能
```bash
llama-bench -m ~/.cache/models/llm/qwen2.5-0.5b-instruct-q4_0.gguf -t 8 
```

## 3. 应用开发

本组件支持两种使用方式，可按场景选择其一：

- **在 SDK 中构建运行**：在已拉取的 SpacemiT Robot SDK 工程内，使用 `mm` 等 SDK 构建命令编译本组件，产物与其它 SDK 组件一起部署到 `output/staging`，适合整机集成、与 SDK 内其它模块（如 asr、tts等）联调。
- **独立构建运行**：单独克隆本组件仓库，用 CMake 在本地编译、运行，不依赖完整 SDK，适合快速体验、单独开发 LLM 能力或在不使用 repo 的环境下使用。


### 3.1. 安装依赖


- **编译环境**：CMake ≥ 3.15，C++17 编译器（GCC/Clang/MSVC）。
- **HTTP 后端**：libcurl、nlohmann_json；**K3 等嵌入式平台**需安装推理工具 `llama.cpp-tools-spacemit`。

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libcurl4-openssl-dev nlohmann-json3-dev

# 安装 spacemit 加速版本的llama.cpp 推理工具
sudo apt install llama.cpp-tools-spacemit
```

### 3.2. 在SDK中构建运行

#### 3.2.1. 编译

本组件已纳入 SpacemiT Robot SDK 时，在 SDK 根目录下执行。SDK 拉取与初始化见 [SpacemiT Robot SDK Manifest](https://github.com/spacemit-robotics/manifest)（使用 repo 管理时需先完成 `repo init`、`repo sync` 等）。

```bash
source build/envsetup.sh
cd components/model_zoo/llm
mm
```

构建产物会安装到 SDK 的 `output/staging` 目录。

#### 3.2.2. 运行

运行前在 SDK 根目录执行 `source build/envsetup.sh`，以便 PATH 与库路径指向 `output/staging`。

在**默认路径**下启动 OpenAI 兼容服务（如 8080 端口）：

```bash
llama-server -m ~/.cache/models/llm/qwen2.5-0.5b-instruct-q4_0.gguf -t 8 --port 8080 &
```

运行示例程序（API 地址、模型名、系统提示、最大 token 数可按需修改）：

```bash
llm_chat "你好" "http://localhost:8080/v1" "qwen2.5-0.5b" "You are a helpful assistant." 256
```

> SDK 构建后可直接执行 `llm_chat`，无需再 `cd build`。

### 3.3. 独立构建运行

#### 3.3.1. 编译

在仓库中进入 LLM 组件目录，配置并编译：

```bash
git clone https://github.com/spacemit-robotics/model_zoo_llm.git
cd model_zoo_llm
mkdir build && cd build
cmake ..
make -j
```

#### 3.3.2. 运行

在**默认路径**下启动 OpenAI 兼容服务（如 8080 端口）：

```bash
llama-server -m ~/.cache/models/llm/qwen2.5-0.5b-instruct-q4_0.gguf -t 8 --port 8080 &
```

运行示例程序（API 地址、模型名、系统提示、最大 token 数可按需修改）：

```bash
cd build
./example/cpp/llm_chat "你好" "http://localhost:8080/v1" "qwen2.5-0.5b" "You are a helpful assistant." 256
```

若使用**云端 / 远端 OpenAI 兼容服务**（如 DeepSeek 等），只需将第 2 个参数替换为云端的 `api_base`，并通过环境变量传入 API Key，例如：

```bash
export OPENAI_API_KEY=你的云端key
./example/cpp/llm_chat "你好" "https://api.deepseek.com" "deepseek-chat" "You are a helpful assistant." 256
```


### 3.4. API使用

- API 说明、多轮对话与 Tool Calling、自定义 Backend 等详见 **官方文档**（链接待补充）。
- 头文件入口：`include/llm_service.h`；实现为 PIMPL，仅依赖该头文件即可集成。

## 4. 常见问题

暂无

## 5. 版本与发布

版本以本目录 `package.xml` 中的 `<version>` 为准。

| 版本   | 说明 |
| ------ | ---- |
| 0.1.0  | 初始版本。提供 C++ LLM 推理与对话接口，支持 OpenAI 兼容 HTTP API（llama-server / vLLM 等）、单轮/多轮与流式调用、Tool Calling。 |

## 6. 贡献方式

欢迎参与贡献：提交 Issue 反馈问题，或通过 Pull Request 提交代码。

- **编码规范**：本组件 C++ 代码遵循 [Google C++ 风格指南](https://google.github.io/styleguide/cppguide.html)，请按该规范编写与修改代码。
- **提交前检查**：请在提交前运行本仓库的 lint 脚本，确保通过风格检查：
  ```bash
  # 在仓库根目录执行（检查全仓库）
  bash scripts/lint/cpplint.sh

  # 仅检查本组件
  bash scripts/lint/cpplint.sh components/model_zoo/llm
  ```
  脚本路径：`scripts/lint/lint_cpp.sh`。若未安装 `cpplint`，可先执行：`pip install cpplint` 或 `pipx install cpplint`。

## 7. License

本组件源码文件头声明为 Apache-2.0，最终以本目录 `LICENSE` 文件为准。

## 8. 附录：模型性能

 ```bash
llama-server -m ~/.cache/models/llm/qwen2.5-0.5b-instruct-q4_0.gguf -t 8 --port 8080 &

cat > prompt_128.txt << 'EOF'
You are a helpful assistant. Please answer the following question in detail.
Explain the basic concepts of large language models, including tokens, context
length, and how they generate text. Make the explanation easy to understand.
EOF

./llm_chat "$(cat prompt_128.txt)" \
  "http://localhost:8080/v1" \
  "qwen2.5-0.5b" \
  "You are a helpful assistant." \
  128

# 上面的步骤可以形成脚本进行测试
./scripts/bench_llm.sh ~/.cache/models/llm/qwen2.5-0.5b-instruct-q4_0.gguf
```
> 说明：以下性能数据是基于K3平台按照上面步骤的实测数据。性能数据为阶段性信息，当前还在持续优化中，最终性能数据将在正式发布前进行修改，请持续关注文档修订记录。

| 模型                  | 参数量 | 量化参数 | first token latency (ms) | token per second (tokens/s) | E2E latency (s) |
|-----------------------|--------|----------|--------------------------|-----------------------------|-----------------|
| Qwen2.5-0.5B-Instruct | 0.5B   | Q4_0     | 184                      | 47.8                        |     2.8         |
| Qwen2.5-0.5B-Instruct | 0.5B   | Q4_0     | 184                      | 47.8                        |     2.8         |
| Qwen3-0.6B            | 0.6B   | Q4_K_M   | 250                      | 36.4                        |     3.7         |
| LFM2.5-1.2B-Instruct  | 1.2B   | Q4_0     | 406                      | 25.4                        |     5.3         |
| Qwen2.5-1.5B-Instruct | 1.5B   | Q4_0     | 398                      | 20.0                        |     6.8         |
| Deepseek R1-1.5B      | 1.5B   | Q4_0     | 463                      | 19.8                        |     6.8         |
| glm-edge-1.5b-chat    | 1.5B   | Q4_0     | 367                      | 21.8                        |     6.2         |
| Qwen3-1.7B            | 1.7B   | Q8_0     | 635                      | 11.6                        |     11.5        |
| Qwen2.5-3B-Instruct   | 3B     | Q4_0     | 806                      | 11.6                        |     12.5        |
| SmallThinker-4B-A0.6B | 4B     | Q4_0     | 345                      | 29.5                        |     4.6         |
| Qwen3-4B              | 4B     | Q4_K_M   | 1477                     | 7.4                         |     18.5        |
| Qwen3-30B-A3B         | 30B    | Q4_0     | 1495                     | 10.0                        |     14.0        |
