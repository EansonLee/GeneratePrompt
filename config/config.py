"""配置文件"""
import os
from pathlib import Path
from typing import Dict, Any

# 基础路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
UPLOAD_DIR = BASE_DIR / "uploads"

def safe_int(value: str, default: int) -> int:
    """安全地将字符串转换为整数"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def safe_float(value: str, default: float) -> float:
    """安全地将字符串转换为浮点数"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

class Config:
    """配置类"""
    
    # 基础路径配置
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = DATA_DIR
    LOG_DIR: Path = LOG_DIR
    UPLOAD_DIR: Path = UPLOAD_DIR
    VECTOR_DB_PATH: str = str(DATA_DIR / "vector_store")
    
    # 应用配置
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    TESTING: bool = os.getenv("TESTING", "False").lower() == "true"
    
    # OpenAI配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "sk-FastAPIvE1M0Ktm0qjx1IZm4LIA1bVdR0mZ0aOtH3BCz2wjn")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.fastapi.ai/v1")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")
    DEFAULT_MODEL_NAME: str = OPENAI_MODEL
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # 向量数据库配置
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    VECTOR_STORE_CONFIG: Dict[str, Any] = {
        "collection_name": "prompt_optimization",
        "distance_strategy": "cosine",
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "separators": ["\n\n", "\n", " ", ""],
        "persist_directory": VECTOR_DB_PATH
    }
    
    # 模型配置
    MODEL_CONFIG: Dict[str, Any] = {
        "temperature": safe_float(os.getenv("MODEL_TEMPERATURE", "0.7"), 0.7),
        "max_tokens": safe_int(os.getenv("MODEL_MAX_TOKENS", "4000"), 4000),
        "top_p": safe_float(os.getenv("MODEL_TOP_P", "0.95"), 0.95),
        "frequency_penalty": safe_float(os.getenv("MODEL_FREQUENCY_PENALTY", "0.0"), 0.0),
        "presence_penalty": safe_float(os.getenv("MODEL_PRESENCE_PENALTY", "0.1"), 0.1)
    }
    
    # 模板生成配置
    TEMPLATE_GENERATION_CONFIG: Dict[str, Any] = {
        "max_contexts": safe_int(os.getenv("TEMPLATE_MAX_CONTEXTS", "5"), 5),
        "max_templates": safe_int(os.getenv("TEMPLATE_MAX_TEMPLATES", "3"), 3),
        "temperature": safe_float(os.getenv("TEMPLATE_TEMPERATURE", "0.7"), 0.7),
        "max_tokens": safe_int(os.getenv("TEMPLATE_MAX_TOKENS", "2000"), 2000)
    }
    
    # Prompt优化配置
    PROMPT_OPTIMIZATION_CONFIG: Dict[str, Any] = {
        "temperature": safe_float(os.getenv("PROMPT_OPTIMIZATION_TEMPERATURE", "0.7"), 0.7),
        "max_tokens": safe_int(os.getenv("PROMPT_OPTIMIZATION_MAX_TOKENS", "4000"), 4000),
        "chunk_size": safe_int(os.getenv("PROMPT_OPTIMIZATION_CHUNK_SIZE", "1000"), 1000),
        "chunk_overlap": safe_int(os.getenv("PROMPT_OPTIMIZATION_CHUNK_OVERLAP", "200"), 200)
    }
    
    # 文件处理配置
    FILE_PROCESSING_CONFIG: Dict[str, Any] = {
        "supported_extensions": [
            ".txt", ".md", ".markdown",
            ".py", ".js", ".jsx", ".ts", ".tsx",
            ".json", ".yaml", ".yml",
            ".zip", ".rar", ".7z"
        ],
        "max_file_size": safe_int(os.getenv("MAX_FILE_SIZE", "10485760"), 10485760),  # 10MB
        "allowed_mime_types": [
            "text/plain",
            "text/markdown",
            "application/json",
            "application/javascript",
            "application/x-javascript",
            "text/javascript",
            "application/typescript",
            "application/x-typescript",
            "application/yaml",
            "application/x-yaml",
            "application/zip",
            "application/x-rar-compressed",
            "application/x-7z-compressed"
        ]
    }
    
    # API配置
    API_CONFIG: Dict[str, Any] = {
        "title": "Prompt生成优化API",
        "description": "提供Prompt模板生成和优化的API服务",
        "version": "1.0.0",
        "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
        "cors_methods": ["*"],
        "cors_headers": ["*"]
    }
    
    # 系统提示词
    TEMPLATE_SYSTEM_PROMPT: str = """你是一个专业的模板生成助手。请根据提供的上下文信息和历史模板，生成一个新的、完整的项目模板。

模板必须包含以下所有部分（缺一不可）：

## 前端技术
- 必须指定前端框架（如React、Vue等）
- UI组件库选择
- 构建工具
- 开发语言（如TypeScript）

## 后端技术
- 后端框架选择
- 编程语言版本
- ORM或数据访问层
- API框架

## 数据库技术
- 主数据库选择
- 缓存解决方案
- 数据备份策略

## API设计
- API风格（REST/GraphQL）
- 认证方案
- 接口文档工具
- 错误处理策略

## 导航设计
- 导航栏类型
- 菜单结构
- 路由设计
- 面包屑导航

## 响应式设计
- 断点设计
- 布局系统
- 适配策略
- 媒体查询方案

## 用户交互流程
- 用户认证流程
- 主要功能流程
- 错误处理流程
- 反馈机制

## 状态管理方案
- 状态管理库选择
- 数据流转方案
- 缓存策略
- 持久化方案

## 数据流设计
- 数据流向图
- 组件通信方式
- 数据更新策略
- 数据同步机制

## 组件设计
- 组件划分策略
- 组件复用方案
- 样式解决方案
- 主题设计"""

    PROMPT_OPTIMIZATION_SYSTEM_PROMPT: str = """你是一个专业的prompt优化助手。
请根据提供的上下文信息和模板，优化用户的prompt。
优化后的prompt应该：
1. 包含完整的页面信息
2. 明确技术要求
3. 描述交互细节
4. 考虑性能和用户体验
"""

# 创建全局配置实例
settings = Config() 