from modelscope.hub.snapshot_download import snapshot_download
# 下载Qwen2.5-0.5B-Instruct（轻量版，适合本地），本地保存到models文件夹
llm_model_dir = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', cache_dir='models')
print("模型下载完成，保存路径：", llm_model_dir)