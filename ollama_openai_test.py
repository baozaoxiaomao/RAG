from openai import OpenAI

# 加载Ollama本地服务，api_key填ollama即可
api_key = 'ollama'
base_url = 'http://localhost:11434/v1'
client = OpenAI(api_key=api_key, base_url=base_url)

# 流式输出测试（推荐，对话机器人用流式）
response = client.chat.completions.create(
    model='qwen2.5:0.5b',  # 与你Ollama运行的模型名称一致
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好"}
    ],
    max_tokens=150,
    temperature=0.7,
    stream=True
)
# 逐块打印回复
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# 非流式输出测试（可选）
# response = client.chat.completions.create(
#     model='qwen2.5:0.5b',
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "你好"}
#     ],
#     max_tokens=150,
#     temperature=0.7,
#     stream=False
# )
# print(response.choices[0].message.content)