from typing import Dict, Any, List, Optional
import logging
import os
from datetime import datetime
from unittest.mock import Mock, MagicMock
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from config.config import OPENAI_API_KEY, VECTOR_STORE_CONFIG
import numpy as np

logger = logging.getLogger(__name__)

class VectorStore:
    """向量数据库管理类"""
    
    def __init__(self, use_mock: bool = False):
        """初始化向量存储
        
        Args:
            use_mock: 是否使用mock数据
        """
        self.is_testing = os.getenv("TESTING", "False").lower() == "true"
        
        if use_mock or self.is_testing:
            # 创建mock embeddings
            mock_embeddings = MagicMock()
            mock_embeddings.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]
            mock_embeddings.embed_query.return_value = [0.1] * 1536
            self.embeddings = mock_embeddings
            
            # 创建mock stores
            mock_contexts = MagicMock()
            mock_contexts.similarity_search.return_value = [
                Document(page_content="测试上下文1"),
                Document(page_content="测试上下文2")
            ]
            self.contexts_store = mock_contexts
            
            mock_templates = MagicMock()
            mock_templates.similarity_search.return_value = [
                Document(page_content="测试模板1"),
                Document(page_content="测试模板2")
            ]
            self.templates_store = mock_templates
        else:
            self.embeddings = OpenAIEmbeddings()
            self.contexts_store = FAISS.from_texts(
                ["示例上下文"],
                self.embeddings,
                metadatas=[{"type": "context"}]
            )
            self.templates_store = FAISS.from_texts(
                ["示例模板"],
                self.embeddings,
                metadatas=[{"type": "template"}]
            )
            
        self.prompt_history = []
    
    def add_react_code(self, code: str, metadata: Dict[str, Any]) -> None:
        """添加React代码
        
        Args:
            code: 代码内容
            metadata: 元数据
        """
        try:
            # 确保metadata包含必要的字段
            metadata = {
                **metadata,
                "type": "react_code",
                "timestamp": datetime.now().isoformat()
            }
            
            # 创建文档
            doc = Document(
                page_content=code,
                metadata=metadata
            )
            
            # 添加到向量存储
            self.contexts_store.add_documents([doc])
            
            logger.info(f"成功添加React代码示例: {metadata.get('description', '')}")
            
        except Exception as e:
            logger.error(f"添加React代码失败: {str(e)}")
            raise Exception(f"添加React代码失败: {str(e)}")
    
    def add_prompt_history(self, original: str, optimized: str) -> None:
        """添加提示优化历史
        
        Args:
            original: 原始提示
            optimized: 优化后的提示
        """
        try:
            doc = Document(
                page_content=optimized,
                metadata={
                    "type": "prompt_history",
                    "original": original
                }
            )
            self.contexts_store.add_documents([doc])
        except Exception as e:
            logger.error(f"添加提示历史失败: {str(e)}")
            raise
    
    def add_best_practice(self, practice: str, category: str) -> List[str]:
        """添加最佳实践
        
        Args:
            practice: 最佳实践内容
            category: 实践类别（如'架构'、'性能'、'测试'等）
            
        Returns:
            List[str]: 添加的文档ID列表
        """
        try:
            metadata = {
                "type": "best_practices",
                "category": category
            }
            return self.contexts_store.add_texts([practice], [metadata])
        except Exception as e:
            logger.error(f"添加最佳实践失败: {str(e)}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """增强的相似度搜索
        
        Args:
            query: 搜索查询
            k: 返回结果数量
            filter_dict: 过滤条件
            
        Returns:
            List[str]: 相似文档内容列表
        """
        try:
            docs = self.contexts_store.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"相似度搜索失败: {str(e)}")
            raise
    
    def search_prompt_history(self, query: str, k: int = 4) -> List[str]:
        """搜索提示优化历史
        
        Args:
            query: 搜索查询
            k: 返回结果数量
            
        Returns:
            List[str]: 历史优化提示列表
        """
        return self.similarity_search(
            query,
            k=k,
            filter_dict={"type": "prompt_history"}
        )
    
    def search_react_code(self, query: str, k: int = 4) -> List[str]:
        """搜索React代码示例
        
        Args:
            query: 搜索查询
            k: 返回结果数量
            
        Returns:
            List[str]: 相关代码示例列表
        """
        return self.similarity_search(
            query,
            k=k,
            filter_dict={"type": "react_code"}
        )
    
    def search_best_practices(self, query: str, k: int = 4) -> List[str]:
        """搜索最佳实践
        
        Args:
            query: 搜索查询
            k: 返回结果数量
            
        Returns:
            List[str]: 相关最佳实践列表
        """
        return self.similarity_search(
            query,
            k=k,
            filter_dict={"type": "best_practices"}
        )
    
    def search_contexts(self, limit: int = 5) -> List[Dict[str, str]]:
        """搜索上下文
        
        Args:
            limit: 返回结果数量限制
            
        Returns:
            List[Dict[str, str]]: 上下文列表
            
        Raises:
            Exception: 搜索失败时抛出异常
        """
        try:
            results = self.contexts_store.similarity_search(
                "项目上下文",
                k=limit
            )
            
            if not results:
                raise Exception("未找到相关上下文")
                
            return [{"content": doc.page_content} for doc in results]
            
        except Exception as e:
            logger.error(f"搜索上下文失败: {str(e)}")
            raise Exception(f"搜索上下文失败: {str(e)}")
        
    def search_templates(self, limit: int = 5) -> List[str]:
        """搜索模板
        
        Args:
            limit: 返回结果数量限制
            
        Returns:
            List[str]: 模板列表
            
        Raises:
            Exception: 搜索失败时抛出异常
        """
        try:
            results = self.templates_store.similarity_search(
                "项目模板",
                k=limit
            )
            
            if not results:
                raise Exception("未找到相关模板")
                
            return [doc.page_content for doc in results]
            
        except Exception as e:
            logger.error(f"搜索模板失败: {str(e)}")
            raise Exception(f"搜索模板失败: {str(e)}")
        
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        """添加文本到向量存储
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
        """
        try:
            self.contexts_store.add_texts(texts, metadatas=metadatas)
        except Exception as e:
            logger.error(f"添加文本失败: {str(e)}")
            raise
        
    def get_prompt_history(self) -> List[Dict[str, str]]:
        """获取prompt优化历史
        
        Returns:
            List[Dict[str, str]]: 历史记录列表
        """
        return self.prompt_history
        
    def clear_prompt_history(self):
        """清除prompt优化历史"""
        try:
            self.prompt_history.clear()
        except Exception as e:
            logger.error(f"清除优化历史失败: {str(e)}")
            raise
            
    def get_project_context(self) -> Dict[str, Any]:
        """获取项目上下文
        
        Returns:
            Dict[str, Any]: 项目上下文信息
        """
        try:
            results = self.contexts_store.similarity_search(
                "项目信息",
                k=1
            )
            if results:
                return {
                    "content": results[0].page_content,
                    "metadata": results[0].metadata
                }
            return {}
        except Exception as e:
            logger.error(f"获取项目上下文失败: {str(e)}")
            raise 
    
    def add_template(self, template: str) -> None:
        """添加模板到向量存储
        
        Args:
            template: 模板内容
        """
        try:
            doc = Document(
                page_content=template,
                metadata={
                    "type": "template",
                    "timestamp": datetime.now().isoformat()
                }
            )
            self.templates_store.add_documents([doc])
            logger.info("成功添加模板")
        except Exception as e:
            logger.error(f"添加模板失败: {str(e)}")
            raise 