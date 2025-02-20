import logging
from typing import Dict, Any, List
from config.config import TEMPLATE_GENERATION_CONFIG
from src.utils.vector_store import VectorStore
from src.agents.template_generation_agent import TemplateGenerationAgent

logger = logging.getLogger(__name__)

class TemplateGenerator:
    """模板生成器"""
    
    def __init__(self, use_mock: bool = False):
        """初始化生成器
        
        Args:
            use_mock: 是否使用mock数据
        """
        self.vector_store = VectorStore(use_mock=use_mock)
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
                limit=TEMPLATE_GENERATION_CONFIG.get('max_contexts', 3)
            )
            templates = self.vector_store.search_templates(
                limit=TEMPLATE_GENERATION_CONFIG.get('max_templates', 2)
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
            # 获取上下文
            context = self._get_context()
            
            # 如果提供了项目类型和描述，添加到上下文中
            if project_type or project_description:
                context["contexts"].append({
                    "content": f"项目类型: {project_type}\n项目描述: {project_description}",
                    "type": "project_info"
                })
            
            # 生成模板
            template = self.agent.generate(
                context["contexts"],
                context["templates"]
            )
            
            # 验证模板是否包含所需信息
            required_sections = [
                "项目名称",
                "项目描述",
                "项目架构",
                "前端技术",
                "后端技术",
                "数据库技术",
                "API设计",
                "页面列表",
                "导航设计",
                "响应式设计",
                "用户交互流程",
                "状态管理方案",
                "数据流设计",
                "组件设计"
            ]
            
            missing_sections = [
                section for section in required_sections
                if section not in template
            ]
            
            if missing_sections:
                raise Exception(f"生成的模板缺少以下必要信息: {', '.join(missing_sections)}")
            
            return template
            
        except Exception as e:
            logger.error(f"模板生成失败: {str(e)}")
            raise Exception(f"模板生成失败: {str(e)}") 