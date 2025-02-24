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
            max_tokens: int = TEMPLATE_GENERATION_CONFIG["max_tokens"]
        ):
        """初始化生成器
        
        Args:
            model_name: 模型名称
            max_contexts: 最大上下文数量
            max_templates: 最大模板数量
            temperature: 温度参数
            max_tokens: 最大token数
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
            
        self.vector_store = VectorStore()
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
        
    def generate(self, project_type: str = None, project_description: str = None) -> str:
        """生成模板
        
        Args:
            project_type: 项目类型
            project_description: 项目描述
            
        Returns:
            str: 生成的模板
            
        Raises:
            Exception: 模板生成失败时抛出异常
        """
        try:
            logger.info("开始生成模板...")
            
            if self.llm is None:
                logger.warning("使用默认模板")
                return self._generate_default_template()
                
            # 获取上下文
            context = self._get_context()
            
            # 提取已有的技术信息
            tech_info = self._extract_tech_info(context)
            
            # 如果提供了项目类型和描述，添加到上下文中
            if project_type or project_description:
                context["contexts"].append({
                    "content": f"项目类型: {project_type}\n项目描述: {project_description}",
                    "type": "project_info"
                })
            
            # 构建提示
            prompt = ChatPromptTemplate.from_messages([
                ("system", TEMPLATE_SYSTEM_PROMPT),
                ("user", f"""请根据以下信息生成一个完整的项目模板：

上下文信息：
{self._format_context(context)}

已提取的技术信息：
{self._format_tech_info(tech_info)}

请确保生成的模板包含所有必要的部分，并与已有的技术选型保持一致。
如果某些信息缺失，请使用合理的默认值补充。""")
            ])
            
            # 生成模板
            messages = prompt.format_messages()
            response = self.llm.invoke(messages)
            template = response.content
            
            if not template or len(template.strip()) == 0:
                logger.warning("生成的模板为空，使用默认模板")
                return self._generate_default_template()
            
            # 验证模板
            logger.info("验证模板...")
            is_valid, missing_fields = self.validate_template(template)
            
            if not is_valid:
                error_msg = f"生成的模板缺少以下必要信息: {', '.join(missing_fields)}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
            logger.info("模板验证通过")
            
            # 保存到向量数据库
            logger.info("保存模板到向量数据库...")
            self.vector_store.add_template(template)
            
            # 验证保存
            if not self.vector_store.verify_insertion(template, "templates"):
                logger.warning("模板可能未成功保存到向量数据库")
            
            return template
            
        except Exception as e:
            logger.error(f"生成模板失败: {str(e)}")
            return self._generate_default_template()

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