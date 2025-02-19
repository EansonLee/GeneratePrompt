import logging
from typing import List, Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config.config import (
    OPENAI_MODEL,
    TEMPLATE_GENERATION_CONFIG,
    TEMPLATE_SYSTEM_PROMPT
)

logger = logging.getLogger(__name__)

class TemplateGenerationAgent:
    """模板生成Agent类"""
    
    def __init__(self):
        """初始化模板生成Agent"""
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=TEMPLATE_GENERATION_CONFIG['temperature'],
            max_tokens=TEMPLATE_GENERATION_CONFIG['max_tokens']
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", TEMPLATE_SYSTEM_PROMPT),
            ("human", self._get_human_prompt_template())
        ])
        
    def _get_human_prompt_template(self) -> str:
        """获取人类提示模板
        
        Returns:
            str: 提示模板
        """
        return """请基于以下信息生成一个高质量的prompt模板：

上下文信息：
{contexts}

历史模板：
{templates}

请生成一个新的、更好的模板。模板应该：
1. 保持清晰和结构化
2. 包含必要的技术细节
3. 考虑性能和可维护性
4. 适应不同场景
5. 易于理解和使用

请输出生成的模板。"""
        
    def generate(self, contexts: List[Dict[str, Any]], templates: List[str]) -> str:
        """生成模板
        
        Args:
            contexts: 上下文信息列表
            templates: 历史模板列表
            
        Returns:
            str: 生成的模板
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
            
            # 生成提示
            prompt = self.prompt_template.format_messages(
                contexts=formatted_contexts,
                templates=formatted_templates
            )
            
            # 调用LLM生成模板
            response = self.llm.invoke(prompt)
            
            return response.content
            
        except Exception as e:
            logger.error(f"模板生成失败: {str(e)}")
            raise 