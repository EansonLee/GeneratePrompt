"""配置文件"""

import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any

# 加载环境变量
load_dotenv()

# 基础路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(DATA_DIR / "vector_store"))

# OpenAI配置
OPENAI_API_KEY = "sk-FastAPITi1j0BFd0pIv5oonN0pozt1mDi0SRz5E3mKCje0DE"
OPENAI_BASE_URL = "https://api.free.fastapi.ai/v1"
OPENAI_MODEL = "free-gpt-4o"
DEFAULT_MODEL_NAME = OPENAI_MODEL

# 应用配置
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = "INFO"

# 文本分块配置
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 向量数据库配置
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_STORE_CONFIG = {
    "collection_name": "prompt_optimization",
    "distance_strategy": "cosine",
    "persist_directory": VECTOR_DB_PATH,
    "CHUNK_SIZE": 1000,
    "CHUNK_OVERLAP": 200,
    "SEPARATORS": ["\n\n", "\n", "。", "，", " ", ""]
}

# 提示模板配置
SYSTEM_TEMPLATE = """你是一个专业的React提示工程专家。你的任务是基于项目上下文优化用户的提示，使其更加具体和有针对性。

你有以下工具可以使用：
1. search_context: 搜索项目相关的上下文信息，包括代码示例、最佳实践等
2. search_react_code: 搜索相关的React代码示例
3. search_best_practices: 搜索相关的React最佳实践
4. search_prompt_history: 搜索历史优化过的提示

在优化提示时，你需要：
1. 使用工具搜索相关信息
2. 分析项目的技术栈和架构
3. 参考现有的组件和功能
4. 遵循代码风格和最佳实践
5. 考虑UI/UX设计规范
6. 注意性能和可维护性要求

你的输出应该是一个优化后的提示，它应该：
1. 保持原始提示的核心意图
2. 包含具体的技术要求
3. 明确架构和设计规范
4. 指定UI/UX要求
5. 包含性能和质量标准
6. 考虑测试和文档需求

请记住在每个步骤都要思考和解释你的决策过程。"""

USER_TEMPLATE = """请基于以下信息优化提示：

原始提示：
{prompt}

项目上下文：
{context}

相关代码：
{code}

最佳实践：
{best_practices}

请生成一个优化后的、更具体的提示。"""

# Agent配置
AGENT_CONFIG = {
    "temperature": 0.7,  # 提高创造性
    "max_tokens": 4000,  # 增加输出长度限制
    "top_p": 0.95,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.1,  # 略微提高以增加输出多样性
}

# 提示优化特定配置
PROMPT_OPTIMIZATION_TEMPERATURE = 0.7
PROMPT_OPTIMIZATION_MAX_TOKENS = 1000

# 向量搜索配置
SEARCH_CONFIG = {
    "k": 5,  # 返回最相似的文档数量
    "fetch_k": 20,  # MMR搜索时初始获取的文档数量
    "lambda_mult": 0.5,  # MMR多样性参数
    "search_type": "mmr"  # 使用MMR搜索以平衡相关性和多样性
}

# React Agent配置
REACT_AGENT_CONFIG = {
    "max_iterations": 5,  # 最大迭代次数
    "early_stopping_method": "generate",  # 使用生成式方法进行早停
    "verbose": True,  # 启用详细输出
    "handle_parsing_errors": True,  # 优雅处理解析错误
    "return_intermediate_steps": True  # 返回中间步骤以便分析
}

# 模型配置
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2000
REACT_AGENT_TEMPERATURE = 0.7
REACT_AGENT_MAX_TOKENS = 4000

# 模板生成配置
TEMPLATE_GENERATION_CONFIG = {
    "max_contexts": 5,  # 最大上下文数量
    "max_templates": 3,  # 最大模板数量
    "temperature": 0.7,  # 温度
    "max_tokens": 2000,  # 最大token数
}

# 模板生成系统提示词
TEMPLATE_SYSTEM_PROMPT = """你是一个专业的模板生成助手。请根据提供的上下文信息和历史模板，生成一个新的、完整的项目模板。

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
- 主题设计

生成模板时，请确保：
1. 所有必要部分都已包含
2. 每个部分都有具体的技术选型和实现方案
3. 技术选型之间要保持一致性
4. 考虑项目的实际需求和规模
5. 遵循最佳实践和行业标准

如果上下文信息不足，请使用合理的默认值，但必须确保生成的模板包含所有必要部分。"""

PROMPT_OPTIMIZATION_SYSTEM_PROMPT = """你是一个专业的prompt优化助手。
请根据提供的上下文信息和模板，优化用户的prompt。
优化后的prompt应该：
1. 包含完整的页面信息
2. 明确技术要求
3. 描述交互细节
4. 考虑性能和用户体验
""" 