"""配置检查脚本"""
import os
import sys
import json
import platform
import requests
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from config.config import settings

def get_default_config() -> dict:
    """从Config类获取默认配置"""
    config = settings
    return {
        # 应用配置
        "DEBUG": str(config.DEBUG),
        "LOG_LEVEL": config.LOG_LEVEL,
        "TESTING": str(config.TESTING),
        
        # OpenAI配置
        "OPENAI_API_KEY": config.OPENAI_API_KEY,
        "OPENAI_BASE_URL": config.OPENAI_BASE_URL,
        "OPENAI_MODEL": config.OPENAI_MODEL,
        "EMBEDDING_MODEL": config.EMBEDDING_MODEL,
        
        # 模型配置
        "MODEL_TEMPERATURE": str(config.MODEL_CONFIG["temperature"]),
        "MODEL_MAX_TOKENS": str(config.MODEL_CONFIG["max_tokens"]),
        "MODEL_TOP_P": str(config.MODEL_CONFIG["top_p"]),
        "MODEL_FREQUENCY_PENALTY": str(config.MODEL_CONFIG["frequency_penalty"]),
        "MODEL_PRESENCE_PENALTY": str(config.MODEL_CONFIG["presence_penalty"]),
        
        # 模板生成配置
        "TEMPLATE_MAX_CONTEXTS": str(config.TEMPLATE_GENERATION_CONFIG["max_contexts"]),
        "TEMPLATE_MAX_TEMPLATES": str(config.TEMPLATE_GENERATION_CONFIG["max_templates"]),
        "TEMPLATE_TEMPERATURE": str(config.TEMPLATE_GENERATION_CONFIG["temperature"]),
        "TEMPLATE_MAX_TOKENS": str(config.TEMPLATE_GENERATION_CONFIG["max_tokens"]),
        
        # Prompt优化配置
        "PROMPT_OPTIMIZATION_TEMPERATURE": str(config.PROMPT_OPTIMIZATION_CONFIG["temperature"]),
        "PROMPT_OPTIMIZATION_MAX_TOKENS": str(config.PROMPT_OPTIMIZATION_CONFIG["max_tokens"]),
        "PROMPT_OPTIMIZATION_CHUNK_SIZE": str(config.PROMPT_OPTIMIZATION_CONFIG["chunk_size"]),
        "PROMPT_OPTIMIZATION_CHUNK_OVERLAP": str(config.PROMPT_OPTIMIZATION_CONFIG["chunk_overlap"]),
        
        # API配置
        "CORS_ORIGINS": "*",
        "MAX_FILE_SIZE": str(config.FILE_PROCESSING_CONFIG["max_file_size"])
    }

def check_env_variables() -> bool:
    """检查环境变量配置
    
    Returns:
        bool: 是否检查通过
    """
    print("\n[检查] 开始检查环境变量配置...")
    
    # 检查OpenAI API配置
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        print("[错误] 未设置OPENAI_API_KEY")
        return False
    print("[成功] OPENAI_API_KEY已设置")
    
    # 检查其他配置
    print("[信息] 当前配置:")
    print(f"- DEBUG: {settings.DEBUG}")
    print(f"- LOG_LEVEL: {settings.LOG_LEVEL}")
    print(f"- TESTING: {settings.TESTING}")
    print(f"- OPENAI_MODEL: {settings.OPENAI_MODEL}")
    
    return True

def check_directories() -> bool:
    """检查目录配置
    
    Returns:
        bool: 是否检查通过
    """
    print("\n[检查] 开始检查目录配置...")
    
    # 检查并创建必要的目录
    directories = [
        settings.DATA_DIR,
        settings.LOG_DIR,
        settings.UPLOAD_DIR,
        Path(settings.VECTOR_STORE_CONFIG["persist_directory"])
    ]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"[成功] 目录已就绪: {directory}")
        except Exception as e:
            print(f"[错误] 创建目录失败 {directory}: {str(e)}")
            return False
    
    return True

def check_openai_api() -> bool:
    """检查OpenAI API连接
    
    Returns:
        bool: 是否检查通过
    """
    print("\n[检查] 开始检查OpenAI API连接...")
    
    try:
        # 构建请求头
        headers = {
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # 发送测试请求
        response = requests.get(
            f"{settings.OPENAI_BASE_URL}/models",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            print("[成功] OpenAI API连接正常")
            return True
        else:
            error_msg = response.text.encode('utf-8').decode('ascii', errors='ignore')
            print(f"[错误] API请求失败: {response.status_code} - {error_msg}")
            return False
            
    except Exception as e:
        error_msg = str(e).encode('utf-8').decode('ascii', errors='ignore')
        print(f"[错误] API连接失败: {error_msg}")
        return False

def save_check_result(is_valid: bool):
    """保存检查结果
    
    Args:
        is_valid: 是否检查通过
    """
    result = {
        "timestamp": str(datetime.now()),
        "is_valid": is_valid,
        "system": platform.system(),
        "python_version": platform.python_version()
    }
    
    result_file = settings.DATA_DIR / "config_check.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def main():
    """主函数"""
    print("[开始] 开始配置检查...")
    
    # 检查操作系统
    system = platform.system()
    print(f"\n[系统] 当前操作系统: {system}")
    
    # 检查环境变量
    env_valid = check_env_variables()
    
    # 检查目录
    dirs_valid = check_directories()
    
    # 检查OpenAI API
    api_valid = check_openai_api()
    
    # 总体检查结果
    all_valid = env_valid and dirs_valid and api_valid
    
    # 保存检查结果
    save_check_result(all_valid)
    
    if all_valid:
        print("\n[成功] 所有检查通过！配置有效。")
        return True
    else:
        print("\n[错误] 配置检查失败！请修复上述问题。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 