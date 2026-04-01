import streamlit as st
import requests

# 定义FastAPI后端地址，与后端端口一致
backend_url = "http://127.0.0.1:6066/chat"

# 页面配置：标题、图标、布局
st.set_page_config(page_title="Qwen2.5聊天机器人", page_icon="🤖", layout="centered")
st.title("🤖 Qwen2.5 聊天机器人")

# 清空聊天历史的函数
def clear_chat_history():
    st.session_state.history = []

# 侧边栏：模型参数配置
with st.sidebar:
    st.title("⚙️ 配置")
    sys_prompt = st.text_input("系统提示词:", value="你是一个专业的AI助手，回答简洁准确。")
    history_len = st.slider("保留历史对话轮数:", min_value=1, max_value=10, value=1, step=1)
    temperature = st.slider("温度(temperature):", min_value=0.01, max_value=2.0, value=0.5, step=0.01)
    top_p = st.slider("采样概率(top_p):", min_value=0.01, max_value=1.0, value=0.5, step=0.01)
    max_tokens = st.slider("最大输出token:", min_value=256, max_value=4096, value=1024, step=8)
    stream = st.checkbox("流式输出", value=True)
    st.button("清空聊天历史", on_click=clear_chat_history)

# 初始化聊天历史，存放在session_state中
if "history" not in st.session_state:
    st.session_state.history = []

# 显示历史聊天记录
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入框
if prompt := st.chat_input("请输入你的问题..."):
    # 显示用户输入
    with st.chat_message("user"):
        st.markdown(prompt)
    # 构建请求参数，传给FastAPI后端
    data = {
        "query": prompt,
        "sys_prompt": sys_prompt,
        "history_len": history_len,
        "history": st.session_state.history,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }
    # 调用FastAPI后端接口，流式请求
    response = requests.post(backend_url, json=data, stream=True)
    # 处理响应
    if response.status_code == 200:
        chunks = ""
        assistant_placeholder = st.chat_message("assistant")
        assistant_text = assistant_placeholder.markdown("")
        # 流式输出
        if stream:
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                chunks += chunk
                assistant_text.markdown(chunks)
        else:
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                chunks += chunk
            assistant_text.markdown(chunks)
        # 将对话加入历史记录
        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.history.append({"role": "assistant", "content": chunks})
    else:
        st.error(f"接口请求失败，状态码：{response.status_code}")