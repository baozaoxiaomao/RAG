import gradio as gr
import requests

# 对接FastAPI后端
backend_url = "http://127.0.0.1:6066/chat"

# 与后端交互的函数
def chat_with_backend(prompt, history, sys_prompt, history_len, temperature, top_p, max_tokens, stream):
    # 处理历史对话，去除Gradio的metadata字段
    history_none_metadata = [{"role": h.get("role"), "content": h.get("content")} for h in history]
    # 构建请求参数
    data = {
        "query": prompt,
        "sys_prompt": sys_prompt,
        "history": history_none_metadata,
        "history_len": history_len,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }
    # 调用后端接口
    response = requests.post(backend_url, json=data, stream=True)
    if response.status_code == 200:
        chunks = ""
        if stream:
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                chunks += chunk
                yield chunks
        else:
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                chunks += chunk
            yield chunks

# 搭建Gradio界面
with gr.Blocks(fill_width=True, fill_height=True) as demo:
    with gr.Tab("🤖 Qwen2.5 聊天机器人"):
        gr.Markdown("## Qwen2.5 本地聊天机器人")
        with gr.Row():
            # 左侧参数配置栏
            with gr.Column(scale=1, variant="panel"):
                sys_prompt = gr.Textbox(label="系统提示词", value="你是一个有用的助手。")
                history_len = gr.Slider(minimum=1, maximum=10, value=1, label="保留历史对话轮数")
                temperature = gr.Slider(minimum=0.01, maximum=2.0, value=0.5, step=0.01, label="temperature")
                top_p = gr.Slider(minimum=0.01, maximum=1.0, value=0.5, step=0.01, label="top_p")
                max_tokens = gr.Slider(minimum=512, maximum=4096, value=1024, step=8, label="max_tokens")
                stream = gr.Checkbox(label="流式输出", value=True)
            # 右侧聊天界面
            with gr.Column(scale=10):
                chatbot = gr.Chatbot(type="messages", height=500)
                # 聊天交互逻辑
                gr.ChatInterface(
                    fn=chat_with_backend,
                    type="messages",
                    chatbot=chatbot,
                    additional_inputs=[sys_prompt, history_len, temperature, top_p, max_tokens, stream]
                )

# 启动Gradio服务
if __name__ == "__main__":
    demo.launch()