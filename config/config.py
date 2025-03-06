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
    VECTOR_STORE_CONFIG: Dict[str, Any] = {
        "collection_name": "prompt_optimization",
        "distance_strategy": "cosine",  # 使用余弦相似度
        "chunk_size": 1000,  # 减小分块大小以提高性能
        "chunk_overlap": 200,  # 减小重叠大小
        "separators": ["\n\n", "\n", ". ", ", ", " "],  # 优化分隔符
        "persist_directory": VECTOR_DB_PATH,
        "enable_dynamic_chunking": True,  # 启用动态分块
        "enable_semantic_deduplication": True,  # 启用语义去重
        "deduplication_threshold": 0.95,  # 去重阈值
        "batch_size": 32,  # 批处理大小
        "max_retries": 3,  # 最大重试次数
        "timeout": 30,  # 超时时间（秒）
        "cache_ttl": 3600,  # 缓存生存时间（秒）
        "max_concurrent_requests": 5,  # 最大并发请求数
        "slow_query_threshold": 5.0,  # 降低慢查询阈值到5秒
        # 嵌入缓存配置
        "embedding_cache": {
            "enabled": True,  # 启用嵌入缓存
            "max_size": 10000,  # 最大缓存条目数
            "ttl": 86400,  # 缓存生存时间（24小时）
            "persist_interval": 300,  # 缓存持久化间隔（5分钟）
            "compression": True  # 启用缓存压缩
        }
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
        "temperature": 0.3,  # 降低温度提高稳定性
        "max_tokens": safe_int(os.getenv("PROMPT_OPTIMIZATION_MAX_TOKENS", "2000"), 2000),  # 减小token数量
        "chunk_size": 1000,  # 减小分块大小
        "chunk_overlap": 200,  # 减小重叠大小
        # 缓存配置
        "cache_size": safe_int(os.getenv("PROMPT_CACHE_SIZE", "2000"), 2000),  # 增加缓存大小
        "cache_ttl": 86400,  # 增加缓存时间到24小时
        # 并行处理配置
        "max_workers": safe_int(os.getenv("PROMPT_MAX_WORKERS", "4"), 4),
        "batch_size": safe_int(os.getenv("PROMPT_BATCH_SIZE", "5"), 5),  # 减小批处理大小
        # 向量检索优化
        "similarity_threshold": 0.75,
        "max_similar_prompts": 5,  # 减少检索数量
        "embedding_batch_size": safe_int(os.getenv("PROMPT_EMBEDDING_BATCH_SIZE", "16"), 16),  # 减小批处理大小
        # 性能监控
        "enable_monitoring": os.getenv("PROMPT_ENABLE_MONITORING", "True").lower() == "true",
        "slow_query_threshold": 5.0,  # 降低慢查询阈值到5秒
        "max_retries": 2,  # 减少重试次数
        "retry_delay": 0.5,  # 减少重试延迟
        # RAG配置
        "search_type": "similarity",  # 改用简单相似度搜索以提高性能
        "lambda_mult": 0.7,
        "rerank_top_k": 3,
        # 嵌入缓存配置
        "embedding_cache": {
            "enabled": True,  # 启用嵌入缓存
            "max_size": 5000,  # 最大缓存条目数
            "ttl": 86400,  # 缓存生存时间（24小时）
            "persist_interval": 300  # 缓存持久化间隔（5分钟）
        }
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
        "title": "Prompt Generator API",
        "description": "用于生成和优化Prompt的API服务",
        "version": "1.0.0",
        "cors_origins": ["*"],  # 允许所有来源，生产环境中应该限制
        "cors_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "cors_headers": ["*"],
    }
    
    # 系统提示词
    TEMPLATE_SYSTEM_PROMPT: str = """你是一个专业的模板生成助手。请根据提供的上下文信息和历史模板，生成一个新的、完整的项目模板。

模板必须包含以下所有部分（缺一不可）：

## 工程目录结构
- 源代码目录（src/）
  - 前端代码目录
  - 后端代码目录
  - 共享代码目录
- 资源目录（assets/）
  - 图片资源
  - 样式资源
  - 静态文件
- 配置目录（config/）
  - 环境配置
  - 应用配置
  - 部署配置
- 测试目录（tests/）
  - 单元测试
  - 集成测试
  - E2E测试
- 文档目录（docs/）
  - API文档
  - 开发文档
  - 部署文档
- 构建目录（build/）
  - 构建脚本
  - 构建配置
  - 输出目录

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
5. 明确工程目录结构和文件组织方式，包括：
   - 源代码目录组织
   - 资源文件目录
   - 配置文件目录
   - 测试目录
   - 文档目录
   - 构建输出目录
"""

# 创建全局配置实例
settings = Config() 