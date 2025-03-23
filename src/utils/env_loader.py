"""环境变量加载工具"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_env_file(env_file: str = None) -> bool:
    """加载环境变量文件
    
    Args:
        env_file: 环境变量文件路径，如果为None则自动查找
        
    Returns:
        bool: 是否成功加载
    """
    try:
        # 如果未指定环境变量文件，则按优先级查找
        if env_file is None:
            # 项目根目录
            root_dir = Path(__file__).resolve().parent.parent.parent
            
            # 按优先级查找环境变量文件
            potential_env_files = [
                root_dir / ".env.local",  # 本地开发环境
                root_dir / ".env.development",  # 开发环境
                root_dir / ".env.production",  # 生产环境
                root_dir / ".env",  # 默认环境
            ]
            
            # 使用第一个存在的环境变量文件
            for env_path in potential_env_files:
                if env_path.exists():
                    env_file = str(env_path)
                    break
        
        # 如果找到环境变量文件，则加载
        if env_file and Path(env_file).exists():
            # 使用python-dotenv加载环境变量
            load_dotenv(env_file, override=True)
            logger.info(f"已加载环境变量文件: {env_file}")
            
            # 打印关键配置信息（不包含敏感信息）
            log_env_vars = [
                "DEBUG", "LOG_LEVEL", "TESTING", 
                "OPENAI_MODEL", "EMBEDDING_MODEL", "VISION_MODEL",
                "VISION_MODEL_TEMPERATURE", "VISION_MODEL_MAX_TOKENS"
            ]
            
            for var in log_env_vars:
                value = os.getenv(var)
                if value:
                    logger.info(f"环境变量 {var} = {value}")
            
            return True
        else:
            logger.warning(f"未找到环境变量文件，将使用默认配置")
            return False
            
    except Exception as e:
        logger.error(f"加载环境变量文件失败: {str(e)}", exc_info=True)
        return False 