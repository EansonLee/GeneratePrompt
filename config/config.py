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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("请在.env文件中设置OPENAI_API_KEY环境变量")
OPENAI_MODEL = "gpt-4-turbo-preview"

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
    "persist_directory": VECTOR_DB_PATH
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
PROMPT_OPTIMIZATION_TEMPERATURE = 0.5  # 平衡创造性和一致性
PROMPT_OPTIMIZATION_MAX_TOKENS = 4000  # 增加token限制以支持更详细的输出

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
DEFAULT_MODEL_NAME = OPENAI_MODEL
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2000
REACT_AGENT_TEMPERATURE = 0.7
REACT_AGENT_MAX_TOKENS = 4000

# 模板生成配置
TEMPLATE_GENERATION_CONFIG = {
    "temperature": 0.7,  # 控制创造性
    "max_tokens": 4000,  # 生成的模板最大长度
    "max_contexts": 5,   # 最大上下文数量
    "max_templates": 3   # 最大历史模板数量
}

# 模板生成系统提示词
TEMPLATE_SYSTEM_PROMPT = """你是一个专业的prompt工程师，擅长生成高质量的prompt模板。

你的任务是基于提供的上下文信息和历史模板，生成一个新的、更好的prompt模板。

在生成模板时，你需要：
1. 分析上下文信息，理解用户的需求和场景
2. 参考历史模板的优点，避免其缺点
3. 确保模板结构清晰，易于理解
4. 包含必要的技术细节和要求
5. 考虑性能和可维护性
6. 使模板具有通用性，适应不同场景
7. 添加适当的示例和说明

请记住：
- 模板应该是可复用的
- 避免过于具体或过于抽象
- 使用清晰的分类和层次结构
- 包含必要的参数和选项
- 提供充分的上下文信息""" 