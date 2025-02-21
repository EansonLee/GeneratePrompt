from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config.config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OPENAI_API_KEY,
    OPENAI_BASE_URL
)
import logging
import os

logger = logging.getLogger(__name__)

class VectorStore:
    """向量数据库管理类"""
    
    def __init__(self):
        """初始化向量数据库"""
        # 确保设置 OpenAI API Key
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,  # 显式传递 API Key
            base_url=OPENAI_BASE_URL  # 添加代理URL
        )
        self.vector_store = FAISS.from_documents(
            [Document(page_content="初始化文档", metadata={})],
            self.embeddings
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> List[str]:
        """添加文本到向量数据库
        
        Args:
            texts: 要添加的文本列表
            metadatas: 文本对应的元数据列表
            
        Returns:
            List[str]: 添加的文档ID列表
        """
        try:
            documents = self.text_splitter.create_documents(texts, metadatas=metadatas)
            self.vector_store.add_documents(documents)
            logger.info(f"成功添加{len(texts)}条文本到向量数据库")
            return ["doc_" + str(i) for i in range(len(texts))]
        except Exception as e:
            logger.error(f"添加文本到向量数据库失败: {str(e)}")
            raise

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
            return self.add_texts([code], [metadata])
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
            return self.add_texts([optimized_prompt], [metadata])
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
            return self.add_texts([practice], [metadata])
        except Exception as e:
            logger.error(f"添加最佳实践失败: {str(e)}")
            raise

    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter_dict: Dict[str, Any] = None
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

    def search_best_practices(
        self, 
        query: str, 
        category: str = None, 
        k: int = 4
    ) -> List[str]:
        """搜索最佳实践
        
        Args:
            query: 搜索查询
            category: 实践类别（可选）
            k: 返回结果数量
            
        Returns:
            List[str]: 相关最佳实践列表
        """
        filter_dict = {"type": "best_practices"}
        if category:
            filter_dict["category"] = category
        return self.similarity_search(query, k=k, filter_dict=filter_dict)

    def delete_collection(self):
        """删除并重新初始化集合"""
        try:
            self.vector_store = FAISS.from_documents(
                [Document(page_content="初始化文档", metadata={})],
                self.embeddings
            )
            logger.info("成功删除并重新初始化向量数据库集合")
        except Exception as e:
            logger.error(f"删除集合失败: {str(e)}")
            raise 