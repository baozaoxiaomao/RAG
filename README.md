# Qwen2.5 本地部署基础环境搭建 README

## 项目简介

本项目基于开源大语言模型 Qwen2.5，完成从模型本地部署到可视化对话机器人的全流程实现。核心采用「Ollama 轻量化部署 + FastAPI 后端 + Streamlit/Gradio 前端」架构，避开选学模块（vLLM 推理加速、自定义 GGUF 文件、模型微调等），聚焦核心功能落地，适合新手快速上手。
项目可实现：本地离线对话、模型参数可视化配置、流式输出回复、历史对话记忆等功能，为后续扩展 RAG、Agent 等高级功能奠定基础。

## 环境要求

### 硬件要求

- 显卡：NVIDIA GPU（支持 CUDA，推荐 RTX 4060 8G 及以上）
- 内存：16G 及以上（保证模型加载和推理流畅）

### 软件要求

- 操作系统：Windows 10/11 64 位
- CUDA 版本：12.7（已验证，适配 RTX 4060）
- 包管理工具：Anaconda/Miniconda（推荐 Anaconda）

### 步骤 1：创建并激活 conda 虚拟环境

1. 打开 Anaconda Prompt（或 CMD/PowerShell），执行以下命令创建指定 Python 版本的虚拟环境：

```
conda create -n RAG python=3.11.9
```

1. 出现`Proceed ([y]/n)?`提示时，输入`y`回车确认安装。
2. 激活创建的虚拟环境：

```
conda activate RAG
```

1. 验证：终端前缀显示`(RAG)`即表示环境激活成功。

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

1. 新建 Python 文件`download_model.py`，写入以下代码：

```
from modelscope.hub.snapshot_download import snapshot_download

# 下载Qwen2.5-7B-Instruct模型，本地缓存到models文件夹
llm_model_dir = snapshot_download(
    model_id='Qwen/Qwen2.5-7B-Instruct',
    cache_dir='models'  # 模型本地保存路径，可自定义
)
print(f"模型下载完成，本地保存路径：{llm_model_dir}")
```

1. 在终端执行该脚本，开始下载模型：

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

## 目录结构说明

```
├── models/                # 模型缓存目录
│   └── Qwen/
│       └── Qwen2.5-7B-Instruct/  # Qwen2.5-7B模型文件
├── download_model.py      # 模型下载脚本
├── basic_infer.py         # 模型基础推理验证脚本
└── README.md              # 环境搭建说明文档
```

## 后续工作

本教程完成了 Qwen2.5 模型的**基础环境搭建 + 本地部署 + 推理验证**，后续可基于此环境继续实现：

1. Ollama 部署 Qwen2.5，实现模型轻量化调用
2. vLLM 推理加速，提升模型生成速度
3. 搭建 FastAPI 后端 + Streamlit/Gradio 前端，实现可视化对话机器人
4. 基于 LangChain 实现 RAG 检索增强生成和 Agent 智能工具调用
