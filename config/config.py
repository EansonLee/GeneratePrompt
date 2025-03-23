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
    
    @property
    def DEBUG(self) -> bool:
        """是否开启调试模式"""
        return os.getenv("DEBUG", "False").lower() == "true"
    
    @property
    def LOG_LEVEL(self) -> str:
        """日志级别"""
        return os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def TESTING(self) -> bool:
        """是否处于测试模式"""
        return os.getenv("TESTING", "False").lower() == "true"
    
    @property
    def OPENAI_API_KEY(self) -> str:
        """OpenAI API密钥"""
        return os.getenv("OPENAI_API_KEY", "sk-FastAPIvE1M0Ktm0qjx1IZm4LIA1bVdR0mZ0aOtH3BCz2wjn")
    
    @property
    def OPENAI_BASE_URL(self) -> str:
        """OpenAI API基础URL"""
        return os.getenv("OPENAI_BASE_URL", "https://api.fastapi.ai/v1")
    
    @property
    def OPENAI_TIMEOUT(self) -> float:
        """获取OpenAI API调用超时时间（秒）"""
        timeout_str = os.getenv("OPENAI_TIMEOUT", "60.0")
        try:
            return float(timeout_str)
        except ValueError:
            logger.warning(f"OPENAI_TIMEOUT值无效: {timeout_str}，使用默认值60.0")
            return 60.0
    
    @property
    def OPENAI_MODEL(self) -> str:
        """OpenAI模型名称"""
        return os.getenv("OPENAI_MODEL", "gpt-4")
    
    @property
    def DEFAULT_MODEL_NAME(self) -> str:
        """默认模型名称"""
        return self.OPENAI_MODEL
    
    @property
    def EMBEDDING_MODEL(self) -> str:
        """嵌入模型名称"""
        return os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    @property
    def MODEL_CONFIG(self) -> Dict[str, Any]:
        """模型配置"""
        return {
            "temperature": safe_float(os.getenv("MODEL_TEMPERATURE", "0.7"), 0.7),
            "max_tokens": safe_int(os.getenv("MODEL_MAX_TOKENS", "4000"), 4000),
            "top_p": safe_float(os.getenv("MODEL_TOP_P", "0.95"), 0.95),
            "frequency_penalty": safe_float(os.getenv("MODEL_FREQUENCY_PENALTY", "0.0"), 0.0),
            "presence_penalty": safe_float(os.getenv("MODEL_PRESENCE_PENALTY", "0.1"), 0.1)
        }
    
    @property
    def TEMPLATE_GENERATION_CONFIG(self) -> Dict[str, Any]:
        """模板生成配置"""
        return {
            "max_contexts": safe_int(os.getenv("TEMPLATE_MAX_CONTEXTS", "5"), 5),
            "max_templates": safe_int(os.getenv("TEMPLATE_MAX_TEMPLATES", "3"), 3),
            "temperature": safe_float(os.getenv("TEMPLATE_TEMPERATURE", "0.7"), 0.7),
            "max_tokens": safe_int(os.getenv("TEMPLATE_MAX_TOKENS", "2000"), 2000)
        }
    
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
    
    @property
    def PROMPT_OPTIMIZATION_CONFIG(self) -> Dict[str, Any]:
        """Prompt优化配置"""
        return {
            "temperature": safe_float(os.getenv("PROMPT_OPTIMIZATION_TEMPERATURE", "0.3"), 0.3),
            "max_tokens": safe_int(os.getenv("PROMPT_OPTIMIZATION_MAX_TOKENS", "2000"), 2000),
            "chunk_size": safe_int(os.getenv("PROMPT_OPTIMIZATION_CHUNK_SIZE", "1000"), 1000),
            "chunk_overlap": safe_int(os.getenv("PROMPT_OPTIMIZATION_CHUNK_OVERLAP", "200"), 200),
            # 缓存配置
            "cache_size": safe_int(os.getenv("PROMPT_CACHE_SIZE", "2000"), 2000),
            "cache_ttl": 86400,  # 增加缓存时间到24小时
            # 并行处理配置
            "max_workers": safe_int(os.getenv("PROMPT_MAX_WORKERS", "4"), 4),
            "batch_size": safe_int(os.getenv("PROMPT_BATCH_SIZE", "5"), 5),
            # 向量检索优化
            "similarity_threshold": 0.75,
            "max_similar_prompts": 5,
            "embedding_batch_size": safe_int(os.getenv("PROMPT_EMBEDDING_BATCH_SIZE", "16"), 16),
            # 性能监控
            "enable_monitoring": os.getenv("PROMPT_ENABLE_MONITORING", "True").lower() == "true",
            "slow_query_threshold": 5.0,
            "max_retries": 2,
            "retry_delay": 0.5,
            # RAG配置
            "search_type": "similarity",
            "lambda_mult": 0.7,
            "rerank_top_k": 3,
            # 嵌入缓存配置
            "embedding_cache": {
                "enabled": True,
                "max_size": 5000,
                "ttl": 86400,
                "persist_interval": 300
            }
        }
    
    @property
    def DESIGN_PROMPT_CONFIG(self) -> Dict[str, Any]:
        """设计图Prompt生成配置"""
        return {
            "model_name": os.getenv("DESIGN_PROMPT_MODEL", "gpt-4"),
            "temperature": safe_float(os.getenv("DESIGN_PROMPT_TEMPERATURE", "0.5"), 0.5),
            "max_tokens": safe_int(os.getenv("DESIGN_PROMPT_MAX_TOKENS", "3000"), 3000),
            # 支持的技术栈
            "supported_tech_stacks": ["Android", "iOS", "Flutter"],
            # 支持的图片格式
            "supported_image_formats": [".jpg", ".jpeg", ".png", ".webp"],
            # 最大图片大小 (5MB)
            "max_image_size": safe_int(os.getenv("MAX_IMAGE_SIZE", "5242880"), 5242880),
            # RAG配置
            "rag_methods": ["similarity", "mmr", "hybrid"],
            "default_rag_method": "similarity",
            "default_retriever_top_k": 3,
            # Agent配置
            "agent_types": ["ReActAgent", "ConversationalRetrievalAgent"],
            "default_agent_type": "ReActAgent",
            "default_context_window_size": 4000,
            # 缓存配置
            "cache_enabled": True,
            "cache_ttl": 86400,  # 24小时
        }
    
    @property
    def FILE_PROCESSING_CONFIG(self) -> Dict[str, Any]:
        """文件处理配置"""
        return {
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
    
    @property
    def API_CONFIG(self) -> Dict[str, Any]:
        """API配置"""
        return {
            "title": "提示词生成 API",
            "description": "用于生成和优化提示词的 API",
            "version": "0.1.0",
            "cors_origins": ["*"],  # 允许所有来源，生产环境应该限制
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
你的任务是优化用户提供的prompt，使其更加清晰、具体、有效。
请分析用户的prompt，并提供一个优化后的版本。"""

    @property
    def VISION_MODEL(self) -> str:
        """视觉模型名称"""
        return os.getenv("VISION_MODEL", "gpt-4o")

    @property
    def VISION_MODEL_CONFIG(self) -> Dict[str, Any]:
        """视觉模型配置"""
        return {
            "temperature": safe_float(os.getenv("VISION_MODEL_TEMPERATURE", "0.3"), 0.3),
            "max_tokens": safe_int(os.getenv("VISION_MODEL_MAX_TOKENS", "3000"), 3000)
        }

    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return {
            "level": self.LOG_LEVEL,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }

# 创建全局配置实例
settings = Config() 