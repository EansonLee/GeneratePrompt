import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools import Tool
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    SYSTEM_TEMPLATE,
    AGENT_CONFIG,
    SEARCH_CONFIG,
    PROMPT_OPTIMIZATION_TEMPERATURE,
    PROMPT_OPTIMIZATION_MAX_TOKENS,
    PROMPT_OPTIMIZATION_SYSTEM_PROMPT
)
from src.utils.vector_store import VectorStore
from src.agents.prompt_optimization_agent import PromptOptimizationAgent

logger = logging.getLogger(__name__)

class PromptOptimizer:
    """提示词优化器"""

    def __init__(
            self,
            model_name: str = OPENAI_MODEL,
            use_mock: bool = False,
            temperature: float = PROMPT_OPTIMIZATION_TEMPERATURE,
            max_tokens: int = PROMPT_OPTIMIZATION_MAX_TOKENS
        ):
        """初始化优化器

        Args:
            model_name: 模型名称
            use_mock: 是否使用mock数据
            temperature: 温度参数
            max_tokens: 最大token数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.vector_store = VectorStore(use_mock=use_mock)
        self.agent = PromptOptimizationAgent(
            vector_store=self.vector_store,
            is_testing=use_mock,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def optimize(self, prompt: str) -> str:
        """优化提示词

        Args:
            prompt: 原始提示词

        Returns:
            str: 优化后的提示词
        """
        if not prompt:
            raise Exception("提示词不能为空")

        try:
            result = self.agent.optimize_prompt(prompt)
            if isinstance(result, dict):
                return result.get("optimized_prompt", "")
            return result
        except Exception as e:
            logger.error(f"优化提示词失败: {str(e)}")
            raise

    def get_optimization_history(self) -> List[Dict[str, str]]:
        """获取优化历史"""
        return self.agent.get_optimization_history()

    def set_model_parameters(self, temperature: float, max_tokens: int):
        """设置模型参数"""
        self.agent.set_model_parameters(temperature, max_tokens)

    def detect_file_type(self, file_path: str) -> str:
        """检测文件类型
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 文件类型
        """
        if file_path.endswith('.test.tsx') or file_path.endswith('.test.jsx'):
            return 'test'
        elif file_path.endswith(('.tsx', '.jsx')):
            return 'react_component'
        elif file_path.endswith('.py'):
            return 'python'
        elif file_path.endswith('.md'):
            return 'doc'
        elif file_path.endswith('.json'):
            return 'config'
        else:
            return 'other'

    def should_ignore(self, file_path: str) -> bool:
        """检查是否应该忽略该文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否应该忽略
        """
        ignore_patterns = [
            'node_modules/',
            'build/',
            'dist/',
            '__pycache__/',
            '.git/',
            '.pytest_cache/',
            '.vscode/',
            '.idea/'
        ]
        return any(pattern in file_path for pattern in ignore_patterns)

    def add_react_code(self, code: str, metadata: Dict[str, Any]) -> bool:
        """添加React代码示例
        
        Args:
            code: 代码内容
            metadata: 元数据
            
        Returns:
            bool: 是否添加成功
        """
        try:
            self.vector_store.add_react_code(code, metadata)
            return True
        except Exception as e:
            logger.error(f"添加React代码失败: {str(e)}")
            raise
            
    def add_best_practice(self, practice: str, category: str):
        """添加最佳实践到向量数据库"""
        try:
            self.vector_store.add_texts([practice], [{"category": category}])
            logger.info(f"成功添加最佳实践，类别: {category}")
        except Exception as e:
            logger.error(f"添加最佳实践失败: {str(e)}")
            raise
            
    def add_reference_content(self, content: str, metadata: Dict[str, Any]):
        """添加参考内容到向量数据库"""
        try:
            self.vector_store.add_texts([content], [metadata])
            logger.info(f"成功添加参考内容，类型: {metadata.get('type')}")
        except Exception as e:
            logger.error(f"添加参考内容失败: {str(e)}")
            raise
            
    def clear_history(self):
        """清除历史记录"""
        self.agent.clear_memory()
        
    def optimize_prompt(self, prompt: str) -> str:
        """优化提示"""
        # 使用专门的Agent进行提示优化
        optimized_prompt = self.agent.execute(prompt)
        
        # 获取优化历史
        optimization_history = self.agent.get_optimization_history()
        
        # 构建详细的优化报告
        report = ["提示优化报告", "=" * 50]
        
        # 添加优化后的提示
        report.append("\n优化后的提示：")
        report.append("-" * 40)
        report.append(optimized_prompt)
        
        # 添加优化过程分析
        if optimization_history:
            report.append("\n优化过程分析：")
            report.append("-" * 40)
            for step in optimization_history:
                if "分析" in step["input"]:
                    report.append(f"\n分析结果：\n{step['output']}")
        
        return "\n".join(report)

    def process_project_file(self, file_path: Union[str, Path], file_type: str = None) -> bool:
        """处理项目文件
        
        Args:
            file_path: 文件路径
            file_type: 文件类型
            
        Returns:
            bool: 是否处理成功
            
        Raises:
            FileNotFoundError: 当文件不存在时抛出
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"文件不存在: {file_path}")
                raise FileNotFoundError(f"文件不存在: {file_path}")
                
            # 读取文件内容
            content = file_path.read_text(encoding='utf-8')
            
            # 根据文件扩展名和内容特征判断文件类型
            if file_type is None:
                file_type = self.detect_file_type(str(file_path))
                
            # 根据文件类型进行处理
            if file_type == 'react_component':
                self.add_react_code(content, {
                    "type": file_type,
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "extension": file_path.suffix
                })
                return True
            elif file_type == 'test':
                # 处理测试文件
                return True
            elif file_type == 'config':
                # 处理配置文件
                return True
            elif file_type == 'doc':
                # 处理文档文件
                self.add_reference_content(content, {
                    "type": file_type,
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "extension": file_path.suffix
                })
                return True
            else:
                return False
            
        except FileNotFoundError as e:
            raise e
        except Exception as e:
            logger.error(f"处理文件失败: {str(e)}")
            return False
            
    def process_project_directory(self, directory: Union[str, Path]) -> Dict[str, int]:
        """处理项目目录
        
        Args:
            directory: 项目目录路径
            
        Returns:
            Dict[str, int]: 处理统计信息
        """
        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"目录不存在: {directory}")
            
        stats = {
            "react_component": 0,
            "test": 0,
            "config": 0,
            "doc": 0,
            "other": 0,
            "failed": 0
        }
        
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file() and not self.should_ignore(str(file_path)):
                    file_type = self.detect_file_type(str(file_path))
                    success = self.process_project_file(file_path, file_type)
                    if success:
                        stats[file_type] += 1
                    else:
                        stats["failed"] += 1
                        
            return stats
        except Exception as e:
            logger.error(f"处理项目目录失败: {str(e)}")
            raise
            
    def get_optimization_requirements(self) -> Dict[str, Any]:
        """获取优化要求
        
        Returns:
            Dict[str, Any]: 优化要求配置
        """
        return {
            "accessibility": True,
            "responsive": True,
            "theme": "dark",
            "language": "zh-CN"
        }

    def optimize_with_template(self, template: str, params: Dict[str, str]) -> Dict[str, Any]:
        """使用模板优化提示
        
        Args:
            template: 提示模板
            params: 模板参数
            
        Returns:
            Dict[str, Any]: 优化结果
        """
        try:
            # 格式化模板
            prompt = template.format(**params)
            
            # 使用标准优化流程
            return self.optimize(prompt)
        except Exception as e:
            logger.error(f"使用模板优化提示失败: {str(e)}")
            raise 