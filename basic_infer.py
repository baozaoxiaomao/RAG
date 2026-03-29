import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 检测设备：有GPU用GPU，无则用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

# 加载分词器和模型（路径为模型下载后的本地路径）
tokenizer = AutoTokenizer.from_pretrained("RAG/models/Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("RAG/models/Qwen/Qwen2.5-0.5B-Instruct").to(device)

# 定义对话内容
prompt = "你好"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},  # 修复文档的语法错误：加{}
    {"role": "user", "content": prompt}
]

# 处理输入格式
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# 模型生成回答
generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
# 截取生成的内容（去掉输入部分）
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
# 解码为文本
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 输出结果
print("模型回答：", response)