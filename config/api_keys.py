import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 校验必要的密钥是否存在
required_keys = [
    "DEEPSEEK_API_KEY", 
    "HF_TOKEN_7B", 
    "HF_TOKEN_14B", 
    "OPENROUTER_API_KEY"
]
for key in required_keys:
    if not os.getenv(key):
        raise ValueError(f"请在.env文件中配置缺失的密钥：{key}")

# DeepSeek API密钥
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# HuggingFace Token配置
HF_TOKENS = {
    "7B": os.getenv("HF_TOKEN_7B"),
    "14B": os.getenv("HF_TOKEN_14B")
}

# OpenRouter完整配置（包含API密钥）
OPENROUTER_CONFIG = {
    "api_key": os.getenv("OPENROUTER_API_KEY"),  # API密钥在此处
    "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    "extra_headers": {
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://your-site-url.com"),
        "X-Title": os.getenv("OPENROUTER_X_TITLE", "Essay Evaluation Project")
    }
}

# 模型端点配置
MODEL_ENDPOINTS = {
    "7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:nscale",
    "14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B:novita",
    "70B": "deepseek/deepseek-r1-distill-llama-70b:free"
}