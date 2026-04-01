# Qwen2.5 本地部署基础环境搭建 README

## 项目简介

本项目基于开源大语言模型 Qwen2.5，完成从**模型本地部署**到**可视化对话机器人**的全流程实现。核心采用「Ollama 轻量化部署 + FastAPI 后端 + Streamlit/Gradio 前端」架构，避开选学模块（vLLM 推理加速、自定义 GGUF 文件、模型微调等），聚焦核心功能落地，适合新手快速上手。

项目可实现：本地离线对话、模型参数可视化配置、流式输出回复、历史对话记忆等功能，为后续扩展 RAG、Agent 等高级功能奠定基础。

## 环境要求

### 硬件要求

- 显卡：NVIDIA GPU（支持 CUDA，推荐 RTX 4060 8G 及以上）
- 内存：16G 及以上（保证模型加载和推理流畅）

### 软件要求

- 操作系统：Windows 10/11 64 位
- CUDA 版本：12.7（已验证，适配 RTX 4060）
- 包管理工具：Anaconda/Miniconda（推荐 Anaconda）

## 核心依赖库

|     库名     |   版本   |              用途              |
| :----------: | :------: | :----------------------------: |
|    torch     |  2.6.0   |        深度学习核心框架        |
| transformers |  4.45.0  |      模型加载、分词、推理      |
|  modelscope  |  1.18.1  |        魔搭社区模型下载        |
|    openai    |  1.71.0  | 对接 Ollama 的 OpenAI 兼容 API |
|   fastapi    | 0.115.12 |       构建高性能后端 API       |
|   uvicorn    |  0.34.0  |       运行 FastAPI 服务        |
|  streamlit   |  1.39.0  |   快速搭建可视化前端（推荐）   |
|    gradio    |  5.0.2   |   备选前端界面（二选一即可）   |
|   requests   | 内置依赖 |      前后端通信、接口调用      |

## 项目目录结构

```
├── models/                # 模型缓存目录
│   └── Qwen/
│       └── Qwen2.5-0.5B-Instruct/  # Qwen2.5-0.5B 模型文件
├── basic_infer.py         # 模型基础推理验证脚本
├── ollama_openai_test.py  # Ollama API 调用测试脚本
├── fastapi_chat.py        # FastAPI 后端服务脚本
├── streamlit_chat.py      # Streamlit 前端界面脚本
├── gradio_chat.py         # Gradio 前端界面脚本
├── README.md              # 项目说明文档（本文档）
└── .gitignore            # Git 忽略文件（已包含 models/ 等大文件）
```

## 实现步骤

### 步骤 1：创建并激活 conda 虚拟环境

打开 Anaconda Prompt（或 CMD/PowerShell），执行以下命令创建指定 Python 版本的虚拟环境：

```
conda create -n RAG python=3.11.9
```

出现`Proceed ([y]/n)?`提示时，输入`y`回车确认安装。

激活创建的虚拟环境：

```
conda activate RAG
```

验证：终端前缀显示`(RAG)`即表示环境激活成功。

### 步骤 2：配置 conda 国内镜像源（解决网络下载问题）

执行以下命令将 conda 默认源替换为**清华源**，避免包下载失败：

```
conda config --remove-key channels
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2/
conda config --set show_channel_urls yes
```

### 步骤 3：安装 GPU 版 PyTorch（适配 CUDA 12.7）

在`(RAG)`环境下，执行以下命令安装**适配 CUDA 12.7**的 PyTorch 及配套组件，使用 PyTorch 官方源 + 国内镜像双重保障：

```
# 官方源（优先推荐，适配CUDA 12.7）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# 若官方源失败，使用清华源兜底
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**验证 PyTorch 安装**：执行以下 Python 代码，验证 CUDA 是否可用：

```
import torch
print(torch.cuda.is_available())  # 输出True即表示GPU版PyTorch安装成功
```

### 步骤 4：安装核心依赖库

安装模型下载、推理所需的核心依赖，指定版本保证兼容性，使用清华源加速：

```
pip install transformers==4.45.0 modelscope==1.18.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- `transformers`：HuggingFace 官方库，提供模型加载、分词、推理核心能力
- `modelscope`：魔搭社区库，用于下载 Qwen2.5 预训练模型

### 步骤 5：下载 Qwen2.5 预训练模型

本教程选择**Qwen2.5-7B-Instruct**（适配 8G GPU，效果优于 0.5B 版），通过魔搭社区下载到本地。

新建 Python 文件`download_model.py`，写入以下代码：

```
from modelscope.hub.snapshot_download import snapshot_download

# 下载Qwen2.5-7B-Instruct模型，本地缓存到models文件夹
llm_model_dir = snapshot_download(
    model_id='Qwen/Qwen2.5-7B-Instruct',
    cache_dir='models'  # 模型本地保存路径，可自定义
)
print(f"模型下载完成，本地保存路径：{llm_model_dir}")
```

在终端执行该脚本，开始下载模型：

```
python download_model.py
```

**注意**：

- 模型大小约 13G，建议在网络稳定的环境下下载，下载过程请勿断网
- 下载完成后，本地会生成`models/Qwen/Qwen2.5-7B-Instruct`目录，包含模型所有文件

### 步骤 6：Qwen2.5 模型基础推理验证

新建 Python 文件`basic_infer.py`，写入以下代码，验证模型是否能正常加载和推理：

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 检测设备：优先使用GPU，无GPU则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备：{device}")

# 加载分词器和预训练模型（替换为自己的模型本地路径）
tokenizer = AutoTokenizer.from_pretrained("models/Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "models/Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16  # 半精度加载，节省显存
).to(device)

# 定义对话输入
prompt = "你好，请介绍一下你自己"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# 处理输入格式，适配Qwen2.5对话模板
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# 模型生成回答
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512,  # 最大生成token数
    temperature=0.5      # 生成随机性，值越低越稳定
)
# 截取生成内容（去除输入部分）
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
# 解码为文本并输出
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("模型回答：\n", response)
```

执行推理脚本：

```
python basic_infer.py
```

**预期结果**：终端输出设备信息后，打印模型的对话回复，即表示模型本地部署、推理验证成功。

![16ad883cd0464c0daa8ea2652023ae89](C:\Users\毛姐\AppData\Local\Temp\16ad883cd0464c0daa8ea2652023ae89.png)

### 步骤 7：Ollama 部署 Qwen2.5（核心步骤）

Ollama 是轻量级模型部署工具，无需复杂配置即可快速运行模型：

1. 安装 Ollama：访问 [Ollama 官网](https://ollama.com/)，下载对应系统版本并一键安装。
2. 验证安装：终端执行 `ollama -v`，显示版本号即安装成功。
3. 拉取并运行 Qwen2.5 模型：

```
# 拉取 Qwen2.5-0.5B 模型（与本地下载模型版本一致）
ollama pull qwen2.5:0.5b
# 运行模型（保持终端后台运行，不要关闭）
ollama run qwen2.5:0.5b
```

测试模型：终端输入 `你好`，模型返回回复即部署成功。

### 步骤 8：测试 Ollama 的 OpenAI 兼容 API

确保 Ollama 模型后台运行（步骤 4 终端未关闭）。

运行 `ollama_openai_test.py` 脚本：

```
python ollama_openai_test.py
```

预期结果：终端流式输出模型回复，无报错即 API 调用成功。

![084f12a8dafe4516a1edd8c8d59f81b6](C:\Users\毛姐\AppData\Local\Temp\084f12a8dafe4516a1edd8c8d59f81b6.png)

### 步骤 9：启动 FastAPI 后端服务

运行后端脚本：

```
python fastapi_chat.py
```

启动成功标志：终端显示 `Uvicorn running on http://0.0.0.0:6066`。

![9200480d7d914e01925298e4d4b938bc](C:\Users\毛姐\AppData\Local\Temp\9200480d7d914e01925298e4d4b938bc.png)

接口测试（可选）：浏览器访问 `http://127.0.0.1:6066/docs`，可看到自动生成的 API 文档，测试 `/chat` 接口是否正常响应。

![bf0cc7d34a5641ef8f9e15a99d1e49c6](C:\Users\毛姐\AppData\Local\Temp\bf0cc7d34a5641ef8f9e15a99d1e49c6.png)

### 步骤 10：启动可视化前端（二选一）

#### 方案 A：Streamlit 前端（推荐）

运行前端脚本：

```
streamlit run streamlit_chat.py
```

![01def8a247a5457283f381d37c7a53b3](C:\Users\毛姐\AppData\Local\Temp\01def8a247a5457283f381d37c7a53b3.png)

首次启动提示：无需输入邮箱，直接按回车键跳过。

访问界面：自动打开浏览器，地址为 `http://localhost:8501`。

![6262a685aa07430aaf316ead93e55617](C:\Users\毛姐\AppData\Local\Temp\6262a685aa07430aaf316ead93e55617.png)

#### 方案 B：Gradio 前端（备选）

运行前端脚本：

```
python gradio_chat.py
```

![a8a3932c13f441dea223a12aed646d1b](C:\Users\毛姐\AppData\Local\Temp\a8a3932c13f441dea223a12aed646d1b.png)

访问界面：浏览器访问 `http://127.0.0.1:7860`。

![64b0c09349874559a3725d3d24007d6a](C:\Users\毛姐\AppData\Local\Temp\64b0c09349874559a3725d3d24007d6a.png)
