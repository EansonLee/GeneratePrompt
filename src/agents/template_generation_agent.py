"""模板生成代理模块"""
from typing import Optional, List, Dict, Any
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from src.utils.vector_store import VectorStore
from config.config import settings

# 设置日志级别
logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

class TemplateGenerationAgent:
    """模板生成代理类"""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        """初始化代理
        
        Args:
            vector_store: 向量存储实例
            temperature: 温度参数
            max_tokens: 最大token数
        """
        try:
            self.vector_store = vector_store
            
            # 使用传入的参数或默认配置
            self.temperature = temperature or settings.TEMPLATE_GENERATION_CONFIG["temperature"]
            self.max_tokens = max_tokens or settings.TEMPLATE_GENERATION_CONFIG["max_tokens"]
            
            self.llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL
            )
            
            logger.info(f"模板生成代理初始化成功，使用模型: {settings.OPENAI_MODEL}")
            
        except Exception as e:
            logger.error(f"模板生成代理初始化失败: {str(e)}")
            raise
        
    async def generate(self, contexts: List[Dict[str, Any]], templates: List[str]) -> str:
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
            
            # 构建消息
            messages = [
                {"role": "system", "content": settings.TEMPLATE_SYSTEM_PROMPT},
                {"role": "user", "content": f"""
请基于以下信息生成一个完整的项目模板：

上下文信息：
{formatted_contexts}

历史模板：
{formatted_templates}

请确保生成的模板包含所有必要的技术选型和实现方案。
"""}
            ]
            
            # 异步调用语言模型
            response = await self.llm.ainvoke(messages)
            
            if not response or not response.content:
                raise ValueError("模板生成失败：未获得有效响应")
                
            template = response.content
            
            # 验证生成的模板
            if not template or len(template.strip()) < 100:
                raise Exception("生成的模板内容不完整")
                
            return template
            
        except Exception as e:
            logger.error(f"模板生成失败: {str(e)}")
            raise Exception(f"模板生成失败: {str(e)}")