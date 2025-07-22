import os
from typing import Dict, Any

def get_config():
    config = {
        "api": {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "openai_api_base": os.getenv("OPENAI_API_BASE"),
            "model_name": os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        },
        "paths": {
            "chat_history_db": "data/chat_history_test.db"
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }
    
    if not config["api"]["openai_api_key"]:
        raise ValueError("环境变量 OPENAI_API_KEY 未设置")
    if not config["api"]["openai_api_base"]:
        raise ValueError("环境变量 OPENAI_API_BASE 未设置")
    
    return config 

