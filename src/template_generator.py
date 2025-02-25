import logging
from typing import Dict, Any, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config.config import (
    TEMPLATE_GENERATION_CONFIG,
    OPENAI_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    TEMPLATE_SYSTEM_PROMPT
)
from src.utils.vector_store import VectorStore
from src.agents.template_generation_agent import TemplateGenerationAgent
from datetime import datetime

logger = logging.getLogger(__name__)

class TemplateGenerator:
    """模板生成器"""
    
    DEFAULT_TEMPLATE = """# Prompt模板

## 角色（Role）
你是一个{role}

## 背景（Context）
{context}

## 任务（Task）
{task}

## 格式要求（Format）
{format_requirements}

## 示例（Example）
{example}

## 注意事项（Notes）
{notes}
"""

    REQUIRED_FIELDS = [
        "前端技术",
        "后端技术",
        "数据库技术",
        "API设计",
        "导航设计",
        "响应式设计",
        "用户交互流程",
        "状态管理方案",
        "数据流设计",
        "组件设计"
    ]

    def __init__(
            self,
            model_name: str = OPENAI_MODEL,
            max_contexts: int = TEMPLATE_GENERATION_CONFIG["max_contexts"],
            max_templates: int = TEMPLATE_GENERATION_CONFIG["max_templates"],
            temperature: float = TEMPLATE_GENERATION_CONFIG["temperature"],
            max_tokens: int = TEMPLATE_GENERATION_CONFIG["max_tokens"],
            vector_store: Optional[VectorStore] = None
        ):
        """初始化生成器
        
        Args:
            model_name: 模型名称
            max_contexts: 最大上下文数量
            max_templates: 最大模板数量
            temperature: 温度参数
            max_tokens: 最大token数
            vector_store: 向量存储实例
        """
        try:
            self.max_contexts = max_contexts
            self.max_templates = max_templates
            
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL
            )
        except Exception as e:
            logger.error(f"初始化ChatOpenAI失败: {str(e)}")
            self.llm = None
            
        # 使用传入的向量存储实例，如果没有传入则创建新实例
        self.vector_store = vector_store or VectorStore()
        self.agent = TemplateGenerationAgent()
        
    def _get_context(self) -> Dict[str, Any]:
        """获取上下文信息
        
        Returns:
            Dict[str, Any]: 包含上下文和模板的字典
        
        Raises:
            Exception: 获取上下文失败时抛出异常
        """
        try:
            contexts = self.vector_store.search_contexts(
                limit=self.max_contexts
            )
            templates = self.vector_store.search_templates(
                limit=self.max_templates
            )
            
            if not contexts or not templates:
                raise Exception("无法获取足够的上下文或模板信息")
            
            return {
                "contexts": contexts,
                "templates": templates
            }
        except Exception as e:
            logger.error(f"获取上下文失败: {str(e)}")
            raise Exception(f"获取上下文失败: {str(e)}")
            
    def validate_template(self, template: str) -> Tuple[bool, List[str]]:
        """验证模板是否包含所有必要信息
        
        Args:
            template: 生成的模板
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 缺失的字段列表)
        """
        missing_fields = []
        for field in self.REQUIRED_FIELDS:
            if field not in template:
                missing_fields.append(field)
                
        return len(missing_fields) == 0, missing_fields
        
    async def generate(self, project_type: str = None, project_description: str = None) -> str:
        """生成项目模板
        
        Args:
            project_type: 项目类型
            project_description: 项目描述
            
        Returns:
            str: 生成的模板内容
        """
        try:
            # 等待向量数据库就绪
            if not self.vector_store:
                raise ValueError("向量数据库未初始化")
                
            if not await self.vector_store.wait_until_ready():
                error = self.vector_store.get_initialization_error()
                raise ValueError(f"向量数据库未就绪: {error}")
            
            # 构建提示词
            messages = [
                {"role": "system", "content": TEMPLATE_SYSTEM_PROMPT},
                {"role": "user", "content": f"""
请根据以下信息生成项目模板：

项目类型：{project_type or '未指定'}
项目描述：{project_description or '未指定'}

请生成一个完整的项目模板，包含所有必要的技术选型和实现方案。
"""}
            ]
            
            # 调用语言模型
            response = await self.llm.ainvoke(messages)
            
            if not response or not response.content:
                raise ValueError("模板生成失败：未获得有效响应")
                
            template_content = response.content
            
            # 保存到向量数据库
            if self.vector_store:
                # 使用add_template方法而不是add_texts
                self.vector_store.add_template(template_content)
                logger.info("模板已保存到向量数据库")
                
            return template_content
            
        except Exception as e:
            logger.error(f"生成模板失败: {str(e)}")
            raise

    def _extract_tech_info(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """从上下文中提取技术信息
        
        Args:
            context: 上下文信息
            
        Returns:
            Dict[str, Any]: 提取的技术信息
        """
        tech_info = {
            "frontend_tech": [],
            "backend_tech": [],
            "database_tech": [],
            "api_design": [],
            "navigation": [],
            "responsive": [],
            "user_flow": [],
            "state_management": [],
            "data_flow": [],
            "component_design": []
        }
        
        # 从上下文中提取信息
        for ctx in context.get("contexts", []):
            content = ctx.get("content", "")
            metadata = ctx.get("metadata", {})
            
            # 从元数据中提取
            for key in tech_info.keys():
                if key in metadata:
                    tech_info[key].append(metadata[key])
            
            # 从内容中提取（使用FileProcessor中相同的模式）
            extracted = self._extract_from_content(content)
            for key, value in extracted.items():
                if value and value not in tech_info[key]:
                    tech_info[key].append(value)
        
        return tech_info
        
    def _format_context(self, context: Dict[str, Any]) -> str:
        """格式化上下文信息
        
        Args:
            context: 上下文信息
            
        Returns:
            str: 格式化后的上下文
        """
        result = []
        for ctx in context.get("contexts", []):
            result.append(f"- {ctx.get('content', '')}")
        return "\n".join(result)
        
    def _format_tech_info(self, tech_info: Dict[str, Any]) -> str:
        """格式化技术信息
        
        Args:
            tech_info: 技术信息
            
        Returns:
            str: 格式化后的技术信息
        """
        result = []
        for key, values in tech_info.items():
            if values:
                result.append(f"{key}: {', '.join(values)}")
        return "\n".join(result)
        
    def _extract_from_content(self, content: str) -> Dict[str, str]:
        """从内容中提取信息（使用与FileProcessor相同的模式）
        
        Args:
            content: 文本内容
            
        Returns:
            Dict[str, str]: 提取的信息
        """
        # 使用与FileProcessor中相同的正则表达式模式
        patterns = {
            "frontend_tech": [
                r"前端技术[：:]\s*(.*?)(?:\n|$)",
                r"frontend[：:]\s*(.*?)(?:\n|$)",
                r"(?:react|vue|angular).*版本[：:]\s*(.*?)(?:\n|$)"
            ],
            "backend_tech": [
                r"后端技术[：:]\s*(.*?)(?:\n|$)",
                r"backend[：:]\s*(.*?)(?:\n|$)",
                r"(?:python|java|node).*版本[：:]\s*(.*?)(?:\n|$)"
            ],
            "database_tech": [
                r"数据库[：:]\s*(.*?)(?:\n|$)",
                r"database[：:]\s*(.*?)(?:\n|$)",
                r"(?:mysql|postgresql|mongodb)[：:]\s*(.*?)(?:\n|$)"
            ],
            "api_design": [
                r"API设计[：:]\s*(.*?)(?:\n|$)",
                r"接口设计[：:]\s*(.*?)(?:\n|$)",
                r"(?:rest|graphql)[：:]\s*(.*?)(?:\n|$)"
            ],
            "navigation": [
                r"导航[：:]\s*(.*?)(?:\n|$)",
                r"navigation[：:]\s*(.*?)(?:\n|$)",
                r"菜单设计[：:]\s*(.*?)(?:\n|$)"
            ],
            "responsive": [
                r"响应式[：:]\s*(.*?)(?:\n|$)",
                r"responsive[：:]\s*(.*?)(?:\n|$)",
                r"自适应[：:]\s*(.*?)(?:\n|$)"
            ],
            "user_flow": [
                r"用户流程[：:]\s*(.*?)(?:\n|$)",
                r"user.*flow[：:]\s*(.*?)(?:\n|$)",
                r"交互流程[：:]\s*(.*?)(?:\n|$)"
            ],
            "state_management": [
                r"状态管理[：:]\s*(.*?)(?:\n|$)",
                r"state.*management[：:]\s*(.*?)(?:\n|$)",
                r"(?:redux|mobx|vuex)[：:]\s*(.*?)(?:\n|$)"
            ],
            "data_flow": [
                r"数据流[：:]\s*(.*?)(?:\n|$)",
                r"data.*flow[：:]\s*(.*?)(?:\n|$)",
                r"数据流转[：:]\s*(.*?)(?:\n|$)"
            ],
            "component_design": [
                r"组件设计[：:]\s*(.*?)(?:\n|$)",
                r"component[：:]\s*(.*?)(?:\n|$)",
                r"组件架构[：:]\s*(.*?)(?:\n|$)"
            ]
        }
        
        info = {}
        import re
        for key, patterns_list in patterns.items():
            for pattern in patterns_list:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    info[key] = matches[0].strip()
                    break
        
        return info
        
    def _generate_default_template(self) -> str:
        """生成默认模板
        
        Returns:
            str: 默认模板
        """
        return """# 项目模板

## 前端技术
- React 18
- TypeScript
- Ant Design
- TailwindCSS

## 后端技术
- FastAPI
- Python 3.11
- SQLAlchemy
- Pydantic

## 数据库技术
- PostgreSQL
- Redis

## API设计
- RESTful API
- OpenAPI/Swagger
- JWT认证

## 导航设计
- 响应式顶部导航
- 侧边栏菜单
- 面包屑导航

## 响应式设计
- 移动优先设计
- 断点适配
- Flex布局

## 用户交互流程
- 登录/注册
- 仪表盘
- 功能页面
- 设置页面

## 状态管理方案
- Redux Toolkit
- React Query
- Context API

## 数据流设计
- 单向数据流
- 状态下推
- 组件通信

## 组件设计
- 原子设计
- 组件复用
- 主题定制
""" 