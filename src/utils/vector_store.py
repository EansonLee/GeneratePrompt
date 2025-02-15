from typing import Dict, Any, List, Optional
import logging
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from config.config import OPENAI_API_KEY, VECTOR_STORE_CONFIG

logger = logging.getLogger(__name__)

class VectorStore:
    """向量数据库管理类"""
    
    def __init__(self):
        """初始化向量存储"""
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.vector_store = FAISS.from_texts(
            texts=[],
            embedding=self.embeddings,
            metadatas=[],
        )
    
    def add_react_code(self, code: str, metadata: Dict[str, Any]) -> List[str]:
        """添加React代码示例
        
        Args:
            code: React代码
            metadata: 代码元数据，包含描述、作者等信息
            
        Returns:
            List[str]: 添加的文档ID列表
        """
        try:
            metadata.update({"type": "react_code"})
            return self.vector_store.add_texts([code], [metadata])
        except Exception as e:
            logger.error(f"添加React代码示例失败: {str(e)}")
            raise
    
    def add_prompt_history(self, original_prompt: str, optimized_prompt: str) -> List[str]:
        """添加提示优化历史
        
        Args:
            original_prompt: 原始提示
            optimized_prompt: 优化后的提示
            
        Returns:
            List[str]: 添加的文档ID列表
        """
        try:
            metadata = {
                "type": "prompt_history",
                "original_prompt": original_prompt
            }
            return self.vector_store.add_texts([optimized_prompt], [metadata])
        except Exception as e:
            logger.error(f"添加提示历史记录失败: {str(e)}")
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
            return self.vector_store.add_texts([practice], [metadata])
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
            docs = self.vector_store.similarity_search(
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