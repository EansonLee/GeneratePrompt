from typing import List, Dict, Any, Union
from pathlib import Path
import os
import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain
from utils.vector_store import VectorStore
from config.config import OPENAI_MODEL as OPENAI_MODEL_NAME, PROMPT_OPTIMIZATION_TEMPERATURE, PROMPT_OPTIMIZATION_MAX_TOKENS, DEFAULT_MODEL_NAME
from .agents.prompt_optimization_agent import PromptOptimizationAgent

logger = logging.getLogger(__name__)

class PromptOptimizer:
    """Prompt优化器，用于优化用户输入的prompt"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """初始化优化器"""
        self.vector_store = VectorStore()
        self.agent = PromptOptimizationAgent(self.vector_store)
        
    def add_react_code(self, code: str, metadata: Dict[str, Any]):
        """添加React代码示例到向量数据库"""
        try:
            self.vector_store.add_react_code(code, metadata)
            logger.info(f"成功添加React代码示例，描述: {metadata.get('description')}")
        except Exception as e:
            logger.error(f"添加React代码示例失败: {str(e)}")
            raise
            
    def add_best_practice(self, practice: str, category: str):
        """添加最佳实践到向量数据库"""
        try:
            self.vector_store.add_best_practice(practice, category)
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
            
    def optimize(self, prompt: str) -> Dict[str, str]:
        """优化提示
        
        Args:
            prompt: 原始提示
            
        Returns:
            Dict[str, str]: 包含原始提示和优化后提示的字典
        """
        try:
            # 使用agent优化提示
            result = self.agent.optimize_prompt(prompt)
            
            # 构建结果
            output = {
                "original_prompt": prompt,
                "optimized_prompt": result["optimized_prompt"]
            }
            
            # 保存优化历史
            self.vector_store.add_prompt_history(prompt, result["optimized_prompt"])
            
            return output
        except Exception as e:
            logger.error(f"优化提示失败: {str(e)}")
            raise
            
    def get_optimization_history(self) -> List[Dict[str, str]]:
        """获取优化历史"""
        return self.agent.get_optimization_history()
        
    def clear_history(self):
        """清除历史记录"""
        self.agent.clear_memory()
        
    def set_model_parameters(self, temperature: float = 0.7, max_tokens: int = 150):
        """设置模型参数"""
        self.agent.set_model_parameters(temperature, max_tokens)

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
            file_type: 文件类型（'react_component', 'test', 'config', 'doc'等）
            
        Returns:
            bool: 是否处理成功
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"文件不存在: {file_path}")
                return False
                
            # 读取文件内容
            content = file_path.read_text(encoding='utf-8')
            
            # 根据文件扩展名和内容特征判断文件类型
            if file_type is None:
                file_type = self._detect_file_type(file_path, content)
            
            # 构建元数据
            metadata = {
                "type": file_type,
                "file_name": file_path.name,
                "file_path": str(file_path),
                "extension": file_path.suffix
            }
            
            # 根据文件类型处理
            if file_type == 'react_component':
                self.add_react_code(content, {
                    **metadata,
                    "description": f"React组件: {file_path.stem}"
                })
            else:
                self.add_reference_content(content, metadata)
                
            logger.info(f"成功处理文件: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {str(e)}")
            return False
            
    def process_project_directory(self, directory_path: Union[str, Path]) -> Dict[str, int]:
        """处理整个项目目录
        
        Args:
            directory_path: 项目目录路径
            
        Returns:
            Dict[str, int]: 各类型文件处理统计
        """
        directory_path = Path(directory_path)
        if not directory_path.exists():
            raise ValueError(f"目录不存在: {directory_path}")
            
        stats = {
            "react_component": 0,
            "test": 0,
            "config": 0,
            "doc": 0,
            "other": 0,
            "failed": 0
        }
        
        # 遍历目录
        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and not self._should_ignore(file_path):
                success = self.process_project_file(file_path)
                if success:
                    file_type = self._detect_file_type(file_path)
                    stats[file_type] = stats.get(file_type, 0) + 1
                else:
                    stats["failed"] += 1
                    
        logger.info(f"目录处理完成: {directory_path}, 统计: {stats}")
        return stats
            
    def _detect_file_type(self, file_path: Path, content: str = None) -> str:
        """检测文件类型
        
        Args:
            file_path: 文件路径
            content: 文件内容（可选）
            
        Returns:
            str: 文件类型
        """
        # 根据文件扩展名判断
        ext = file_path.suffix.lower()
        if ext in {'.jsx', '.tsx'}:
            return 'react_component'
        elif ext == '.test.tsx' or ext == '.test.jsx' or ext == '.spec.tsx' or ext == '.spec.jsx':
            return 'test'
        elif ext in {'.json', '.yaml', '.yml'}:
            return 'config'
        elif ext in {'.md', '.mdx', '.txt'}:
            return 'doc'
        
        # 如果提供了内容，进一步分析
        if content:
            if 'import React' in content or 'from "react"' in content:
                return 'react_component'
            elif 'describe(' in content or 'test(' in content:
                return 'test'
                
        return 'other'
        
    def _should_ignore(self, file_path: Path) -> bool:
        """判断是否应该忽略该文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否应该忽略
        """
        # 忽略的目录
        ignore_dirs = {
            'node_modules', 'build', 'dist', '.git', 
            '__pycache__', '.idea', '.vscode'
        }
        
        # 忽略的文件扩展名
        ignore_extensions = {
            '.pyc', '.pyo', '.pyd', '.so', '.dll', '.class',
            '.log', '.env', '.lock'
        }
        
        # 检查目录
        for parent in file_path.parents:
            if parent.name in ignore_dirs:
                return True
                
        # 检查文件扩展名
        if file_path.suffix.lower() in ignore_extensions:
            return True
            
        return False 