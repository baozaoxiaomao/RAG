from fastapi import FastAPI, Body
from openai import AsyncOpenAI
from typing import List
from fastapi.responses import StreamingResponse

# 初始化FastAPI应用
app = FastAPI()

# 初始化异步OpenAI客户端，对接Ollama
api_key = 'ollama'
base_url = 'http://localhost:11434/v1'
aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)

# 初始化对话列表
messages = []

# 定义/chat接口，POST请求，流式输出
@app.post("/chat")
async def chat(
    query: str = Body(..., description="用户输入"),
    sys_prompt: str = Body("你是一个有用的助手。", description="系统提示词"),
    history: List = Body([], description="历史对话"),
    history_len: int = Body(1, description="保留历史对话的轮数"),
    temperature: float = Body(0.5, description="LLM采样温度"),
    top_p: float = Body(0.5, description="LLM采样概率"),
    max_tokens: int = Body(None, description="LLM最大token数量")
):
    global messages
    # 控制历史对话长度，只保留最新的N轮
    if history_len > 0:
        history = history[-2 * history_len:]
    # 清空并重新构建对话消息
    messages.clear()
    messages.append({"role": "system", "content": sys_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": query})

    # 调用Ollama的OpenAI API
    response = await aclient.chat.completions.create(
        model="qwen2.5:0.5b",  # 与Ollama的模型名称一致
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True
    )

    # 流式生成响应
    async def generate_response():
        async for chunk in response:
            chunk_msg = chunk.choices[0].delta.content
            if chunk_msg:
                yield chunk_msg

    # 返回流式响应
    return StreamingResponse(generate_response(), media_type="text/plain")

# 启动服务
if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0表示局域网可访问，端口6066（可自定义）
    uvicorn.run(app, host="0.0.0.0", port=6066, log_level="info")