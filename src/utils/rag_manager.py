"""RAG管理器模块"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.config import settings
from src.utils.vector_store import VectorStore
import asyncio

logger = logging.getLogger(__name__)

class RAGManager:
    """RAG管理器，用于增强提示词生成"""
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """初始化RAG管理器
        
        Args:
            vector_store: 向量存储实例
        """
        try:
            self.vector_store = vector_store or VectorStore()
            
            # 等待向量存储初始化完成
            if not self.vector_store.is_initialized():
                logger.warning("等待向量存储初始化...")
                if not asyncio.run(self.vector_store.wait_until_ready(timeout=30)):
                    raise ValueError("向量存储初始化超时")
            
            # 验证必要的存储是否存在
            if not hasattr(self.vector_store, 'context_store'):
                raise ValueError("向量存储缺少context_store")
            
            # 初始化文本分割器
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.VECTOR_STORE_CONFIG["chunk_size"],
                chunk_overlap=settings.VECTOR_STORE_CONFIG["chunk_overlap"],
                separators=settings.VECTOR_STORE_CONFIG["separators"]
            )
            
            # 初始化检索器
            self.retriever = self.vector_store.context_store.as_retriever(
                search_kwargs={"k": 5}
            )
            
            logger.info("RAG管理器初始化成功")
            
        except Exception as e:
            logger.error(f"RAG管理器初始化失败: {str(e)}")
            raise
            
    async def enhance_prompt(self, prompt: str) -> Dict[str, Any]:
        """使用RAG增强提示词
        
        Args:
            prompt: 原始提示词
            
        Returns:
            Dict[str, Any]: 增强结果
        """
        try:
            # 1. 检索相关文档
            relevant_docs = await self.retriever.aget_relevant_documents(prompt)
            
            # 2. 提取上下文信息
            context = self._extract_context(relevant_docs)
            
            # 3. 分析相关性
            relevance = self._analyze_relevance(prompt, relevant_docs)
            
            # 4. 构建增强提示
            enhanced_prompt = self._build_enhanced_prompt(prompt, context)
            
            return {
                "enhanced_prompt": enhanced_prompt,
                "context": context,
                "relevance": relevance,
                "sources": [self._get_doc_metadata(doc) for doc in relevant_docs]
            }
            
        except Exception as e:
            logger.error(f"增强提示词失败: {str(e)}")
            return {
                "enhanced_prompt": prompt,
                "error": str(e)
            }
            
    def _extract_context(self, docs: List[Document]) -> Dict[str, List[str]]:
        """从文档中提取上下文信息
        
        Args:
            docs: 相关文档列表
            
        Returns:
            Dict[str, List[str]]: 分类的上下文信息
        """
        context = {
            "technical": [],  # 技术相关
            "business": [],   # 业务相关
            "examples": [],   # 示例
            "best_practices": []  # 最佳实践
        }
        
        for doc in docs:
            doc_type = doc.metadata.get("type", "unknown")
            content = doc.page_content.strip()
            
            if doc_type == "technical":
                context["technical"].append(content)
            elif doc_type == "business":
                context["business"].append(content)
            elif doc_type == "example":
                context["examples"].append(content)
            elif doc_type == "best_practices":
                context["best_practices"].append(content)
                
        return context
        
    def _analyze_relevance(self, prompt: str, docs: List[Document]) -> List[Dict[str, Any]]:
        """分析文档与提示词的相关性
        
        Args:
            prompt: 提示词
            docs: 相关文档列表
            
        Returns:
            List[Dict[str, Any]]: 相关性分析结果
        """
        relevance = []
        
        for doc in docs:
            score = self.vector_store.context_store.similarity_search_with_score(
                query=prompt,
                k=1,
                filter={"id": doc.metadata.get("id")}
            )[0][1]
            
            relevance.append({
                "doc_type": doc.metadata.get("type", "unknown"),
                "score": float(score),
                "timestamp": doc.metadata.get("timestamp", "unknown")
            })
            
        return sorted(relevance, key=lambda x: x["score"], reverse=True)
        
    def _build_enhanced_prompt(self, prompt: str, context: Dict[str, List[str]]) -> str:
        """构建增强的提示词
        
        Args:
            prompt: 原始提示词
            context: 上下文信息
            
        Returns:
            str: 增强后的提示词
        """
        # 构建技术上下文
        tech_context = "\n".join(context["technical"][:2]) if context["technical"] else ""
        
        # 构建业务上下文
        business_context = "\n".join(context["business"][:2]) if context["business"] else ""
        
        # 构建示例
        examples = "\n".join(context["examples"][:2]) if context["examples"] else ""
        
        # 构建最佳实践
        best_practices = "\n".join(context["best_practices"][:2]) if context["best_practices"] else ""
        
        # 组装增强提示词
        enhanced_prompt = f"""基于以下上下文信息优化提示词：

## 技术上下文
{tech_context}

## 业务上下文
{business_context}

## 相关示例
{examples}

## 最佳实践
{best_practices}

原始提示词：
{prompt}
"""
        
        return enhanced_prompt
        
    def _get_doc_metadata(self, doc: Document) -> Dict[str, Any]:
        """获取文档元数据
        
        Args:
            doc: 文档对象
            
        Returns:
            Dict[str, Any]: 处理后的元数据
        """
        return {
            "type": doc.metadata.get("type", "unknown"),
            "timestamp": doc.metadata.get("timestamp", "unknown"),
            "source": doc.metadata.get("source", "unknown"),
            "id": doc.metadata.get("id", "unknown")
        }
        
    async def add_context(
        self,
        content: str,
        content_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """添加上下文内容
        
        Args:
            content: 上下文内容
            content_type: 内容类型（technical/business/example/best_practice）
            metadata: 额外的元数据
        """
        try:
            # 准备元数据
            metadata = metadata or {}
            metadata.update({
                "type": content_type,
                "timestamp": datetime.now().isoformat(),
                "source": "manual_input"
            })
            
            # 分割文本
            docs = self.text_splitter.create_documents(
                texts=[content],
                metadatas=[metadata]
            )
            
            # 添加到向量存储
            await self.vector_store.add_documents(docs)
            
            logger.info(f"成功添加{content_type}类型的上下文内容")
            
        except Exception as e:
            logger.error(f"添加上下文内容失败: {str(e)}")
            raise
            
    async def get_context_stats(self) -> Dict[str, int]:
        """获取上下文统计信息
        
        Returns:
            Dict[str, int]: 各类型上下文的数量统计
        """
        try:
            stats = {
                "technical": 0,
                "business": 0,
                "example": 0,
                "best_practice": 0,
                "unknown": 0
            }
            
            # 获取所有文档
            docs = await self.vector_store.get_all_documents()
            
            # 统计各类型数量
            for doc in docs:
                doc_type = doc.metadata.get("type", "unknown")
                if doc_type in stats:
                    stats[doc_type] += 1
                else:
                    stats["unknown"] += 1
                    
            return stats
            
        except Exception as e:
            logger.error(f"获取上下文统计信息失败: {str(e)}")
            return {} 