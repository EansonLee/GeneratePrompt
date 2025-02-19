import logging
from typing import Optional, Dict, Any
from config.config import TEMPLATE_GENERATION_CONFIG
from utils.vector_store import VectorStore
from agents.template_generation_agent import TemplateGenerationAgent

logger = logging.getLogger(__name__)

class TemplateGenerator:
    """模板prompt生成器类"""
    
    def __init__(self):
        """初始化模板生成器"""
        self.vector_store = VectorStore()
        self.agent = TemplateGenerationAgent()
        
    def _get_context(self) -> Dict[str, Any]:
        """获取上下文信息
        
        Returns:
            Dict[str, Any]: 上下文信息
        """
        try:
            # 从向量数据库获取相关上下文
            contexts = self.vector_store.search_contexts(
                limit=TEMPLATE_GENERATION_CONFIG['max_contexts']
            )
            
            # 获取历史模板
            templates = self.vector_store.search_templates(
                limit=TEMPLATE_GENERATION_CONFIG['max_templates']
            )
            
            return {
                'contexts': contexts,
                'templates': templates
            }
            
        except Exception as e:
            logger.error(f"获取上下文失败: {str(e)}")
            raise
            
    def generate(self) -> str:
        """生成模板prompt
        
        Returns:
            str: 生成的模板prompt
        """
        try:
            # 获取上下文
            context = self._get_context()
            
            # 使用Agent生成模板
            template = self.agent.generate(
                contexts=context['contexts'],
                templates=context['templates']
            )
            
            return template
            
        except Exception as e:
            logger.error(f"生成模板失败: {str(e)}")
            raise 