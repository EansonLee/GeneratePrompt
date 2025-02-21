import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from config.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_BASE_URL,
    TEMPLATE_SYSTEM_PROMPT
)

logger = logging.getLogger(__name__)

class TemplateGenerationAgent:
    """模板生成Agent"""
    
    def __init__(self):
        """初始化Agent"""
        self.llm = ChatOpenAI(
            model_name=OPENAI_MODEL,
            temperature=0.7,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        
        # 初始化提示模板
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", TEMPLATE_SYSTEM_PROMPT),
            ("human", """
            请基于以下信息生成一个完整的项目模板：
            
            上下文信息：
            {contexts}
            
            历史模板：
            {templates}
            
            生成的模板必须包含以下所有信息，缺一不可：
            
            1. 项目基本信息
               - 项目名称：具体的项目名称
               - 项目描述：详细的项目功能和目标描述
               - 项目架构：清晰的架构设计说明
            
            2. 技术栈信息
               - 前端技术：具体使用的前端框架和库（如React、Vue等）
               - UI框架：选用的UI组件库（如Ant Design、Material-UI等）
               - 后端技术：后端框架和主要库
               - 数据库技术：数据库选型和设计
               - API设计：API架构和规范
               
            3. 页面信息
               - 页面列表：详细的页面清单
               - 导航设计：导航结构和交互方式
               - 响应式设计：响应式布局方案
               - 用户交互流程：主要用户操作流程
               - 状态管理方案：状态管理工具和策略
               - 数据流设计：数据流转和处理方案
               - 组件设计：
                 * 组件层次：组件结构和层次关系
                 * 组件通信：组件间通信方式
                 * 组件复用：复用策略和最佳实践
            """),
            ("assistant", "我将基于提供的信息生成一个完整的模板，确保包含所有必要的信息。"),
            ("human", "请生成模板")
        ])
        
        # 创建生成链
        self.chain = self.prompt_template | self.llm
        
    def generate(self, contexts: List[Dict[str, Any]], templates: List[str]) -> str:
        """生成模板
        
        Args:
            contexts: 上下文信息列表
            templates: 历史模板列表
            
        Returns:
            str: 生成的模板
            
        Raises:
            Exception: 模板生成失败时抛出异常
        """
        try:
            # 格式化上下文
            formatted_contexts = "\n".join(
                f"- {ctx.get('content', '')}"
                for ctx in contexts
            )
            
            # 格式化历史模板
            formatted_templates = "\n".join(
                f"模板 {i+1}:\n{template}"
                for i, template in enumerate(templates)
            )
            
            # 执行生成
            result = self.chain.invoke({
                "contexts": formatted_contexts,
                "templates": formatted_templates
            })
            
            # 验证生成的模板
            template = result.content
            if not template or len(template.strip()) < 100:
                raise Exception("生成的模板内容不完整")
                
            return template
            
        except Exception as e:
            logger.error(f"模板生成失败: {str(e)}")
            raise Exception(f"模板生成失败: {str(e)}") 