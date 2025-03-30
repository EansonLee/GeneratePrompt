from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import os
from datetime import datetime
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import numpy as np
from config.config import settings
import asyncio
import faiss
from tqdm import tqdm
import time
from src.utils.performance_monitor import performance_monitor
import json

logger = logging.getLogger(__name__)

class VectorStore:
    """向量存储类，使用FAISS实现高效的向量检索"""
    
    def __init__(self, storage_dir: Optional[Union[str, Path]] = None):
        """初始化向量存储
        
        Args:
            storage_dir: 向量存储目录
        """
        self.storage_dir = Path(storage_dir or settings.DATA_DIR) / "vector_store"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化嵌入模型
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_BASE_URL,
            openai_organization=None  # 如果需要组织ID，可以从settings中添加
        )
        
        # 初始化向量存储
        self.stores = self._init_stores()
        
        logger.info(f"向量存储初始化完成，使用目录: {self.storage_dir}")
    
    def _init_stores(self) -> Dict[str, FAISS]:
        """初始化各类型向量存储
        
        Returns:
            Dict[str, FAISS]: 向量存储字典
        """
        stores = {}
        store_types = ['contexts', 'templates', 'prompts']
        
        for store_type in store_types:
            index_path = self.storage_dir / store_type
            
            try:
                if index_path.exists():
                    stores[store_type] = FAISS.load_local(
                        str(index_path),
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                else:
                    # 创建新的向量存储
                    stores[store_type] = self._create_new_store()
                    # 保存
                    stores[store_type].save_local(str(index_path))
                    
            except Exception as e:
                logger.error(f"初始化向量存储 {store_type} 失败: {str(e)}")
                # 创建新的作为后备
                stores[store_type] = self._create_new_store()
                
        return stores
        
    def _create_new_store(self) -> FAISS:
        """创建新的向量存储
        
        Returns:
            FAISS: 新的向量存储实例
        """
        # 使用简单示例文本初始化
        texts = ["初始化存储示例"]
        return FAISS.from_texts(texts, self.embeddings)
    
    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        store_type: str = "contexts"
    ) -> List[str]:
        """添加文本到向量存储
        
        Args:
            texts: 要添加的文本列表
            metadatas: 元数据列表
            store_type: 存储类型
            
        Returns:
            List[str]: 文档ID列表
        """
        if not texts:
            return []
                
        store = self.stores.get(store_type)
        if not store:
            raise ValueError(f"未找到向量存储: {store_type}")
            
        # 获取嵌入向量
        embeddings = await self._get_embeddings_batch(texts)
        
        # 添加到向量存储
        try:
            ids = store.add_embeddings(
                text_embeddings=list(zip(texts, embeddings)),
                metadatas=metadatas
            )
            
            # 保存更新后的索引
            store.save_local(str(self.storage_dir / store_type))
            
            return ids
            
        except Exception as e:
            logger.error(f"添加文本到向量存储失败: {str(e)}")
            raise
            
    async def similarity_search(
        self, 
        query: str, 
        top_k: int = 5,
        store_type: str = "contexts",
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """使用ANN进行相似度搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            store_type: 存储类型
            threshold: 相似度阈值
            
        Returns:
            List[Dict[str, Any]]: 搜索结果列表
        """
        with performance_monitor.measure_time(
            "vector_search", 
            query=query,
            store_type=store_type
        ):
            store = self.stores.get(store_type)
            if not store:
                raise ValueError(f"未找到向量存储: {store_type}")
                
            # 获取查询向量
            query_embedding = await self._get_embedding(query)
            
            # 使用FAISS的ANN搜索
            docs_and_scores = store.similarity_search_with_score_by_vector(
                embedding=query_embedding,
                k=top_k
            )
            
            results = []
            for doc, score in docs_and_scores:
                # 转换余弦相似度到0-1范围
                similarity = 1 - score / 2
                
                if similarity >= threshold:
                    results.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity': similarity
                    })
            
            # 更新性能监控的结果数量
            performance_monitor.metrics["vector_search"][-1]["result_count"] = len(results)
                
            return results
            
    async def _get_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 嵌入向量
        """
        try:
            embedding = await asyncio.to_thread(
                self.embeddings.embed_query,
                text
            )
            return embedding
        except Exception as e:
            logger.error(f"获取文本嵌入向量失败: {str(e)}")
            raise

    async def _get_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 20
    ) -> List[List[float]]:
        """批量获取文本的嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = await asyncio.to_thread(
                    self.embeddings.embed_documents,
                    batch
                )
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"批量获取嵌入向量失败: {str(e)}")
                raise

        return embeddings

    def is_initialized(self) -> bool:
        """检查向量存储是否已初始化
        
        Returns:
            bool: 向量存储是否已初始化
        """
        return hasattr(self, 'stores') and bool(self.stores)
        
    def is_ready(self) -> bool:
        """检查向量存储是否就绪，可以处理请求
        
        Returns:
            bool: 向量存储是否就绪
        """
        # 检查是否已初始化并且stores中至少有一个store可用
        if not self.is_initialized():
            return False
            
        # 检查是否至少有一个存储类型可用
        for store_type in self.stores:
            if self.stores[store_type] is not None:
                return True
                
        return False
        
    def get_initialization_error(self) -> str:
        """获取初始化过程中的错误信息
        
        Returns:
            str: 错误信息
        """
        if not hasattr(self, 'stores'):
            return "向量存储未正确初始化"
        if not self.stores:
            return "向量存储为空"
        return ""
        
    async def get_store_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取各存储的统计信息
        
        Returns:
            Dict[str, Dict[str, Any]]: 各存储的统计信息
        """
        stats = {}
        
        for store_type, store in self.stores.items():
            if store is not None:
                # 获取实例的一些基本统计信息
                doc_count = 0
                index_size = 0
                
                # 尝试获取文档数量
                if hasattr(store, "docstore") and hasattr(store.docstore, "docs"):
                    doc_count = len(store.docstore.docs)
                    
                # 尝试获取索引大小
                if hasattr(store, "index"):
                    try:
                        # 尝试使用ntotal属性（FAISS索引通常有这个属性）
                        if hasattr(store.index, "ntotal"):
                            index_size = store.index.ntotal
                        # 尝试使用len()
                        else:
                            try:
                                index_size = len(store.index)
                            except (TypeError, AttributeError):
                                # 如果都不支持，则使用0作为默认值
                                index_size = 0
                    except Exception as e:
                        logger.warning(f"获取索引大小失败: {str(e)}")
                        index_size = 0
                
                stats[store_type] = {
                    "available": True,
                    "document_count": doc_count,
                    "index_size": index_size
                }
            else:
                stats[store_type] = {
                    "available": False,
                    "error": "存储实例为空"
                }
                
        return stats
        
    async def wait_until_ready(self, timeout: int = 10) -> bool:
        """等待向量存储初始化完成
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            bool: 是否初始化成功
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_ready():
                return True
            await asyncio.sleep(0.5)
        return False
        
    def optimize_index(self, store_type: str = "contexts"):
        """优化向量索引
        
        Args:
            store_type: 存储类型
        """
        store = self.stores.get(store_type)
        if not store:
            raise ValueError(f"未找到向量存储: {store_type}")
            
        try:
            # 获取原始索引
            index = store.index
            
            # 训练量化器
            quantizer = faiss.IndexFlatL2(index.d)
            index = faiss.IndexIVFFlat(quantizer, index.d, min(index.ntotal, 100))
            
            # 训练索引
            if index.is_trained:
                logger.info("索引已训练，跳过训练步骤")
            else:
                logger.info("开始训练索引...")
                train_vectors = index.reconstruct_n(0, index.ntotal)
                index.train(train_vectors)
                
            # 添加向量到新索引
            vectors = [index.reconstruct(i) for i in range(index.ntotal)]
            index.add(np.array(vectors))
            
            # 更新存储的索引
            store.index = index
            
            # 保存优化后的索引
            store.save_local(str(self.storage_dir / store_type))
            
            logger.info(f"向量索引 {store_type} 优化完成")
        except Exception as e:
            logger.error(f"优化向量索引失败: {str(e)}")
            raise
            
    async def get_relevant_context(
        self,
        query: str,
        store_type: str = "contexts",
        top_k: int = 5,
        max_tokens: int = 2000
    ) -> str:
        """获取相关上下文
        
        Args:
            query: 查询文本
            store_type: 存储类型
            top_k: 返回结果数量
            max_tokens: 最大token数量
            
        Returns:
            str: 相关上下文
        """
        try:
            # 搜索相似内容
            results = await self.similarity_search(
                query=query,
                top_k=top_k,
                store_type=store_type
            )
            
            if not results:
                return ""
                
            # 组合上下文
            context = ""
            for i, result in enumerate(results):
                content = result.get('content', '')
                similarity = result.get('similarity', 0)
                source = result.get('metadata', {}).get('source', 'unknown')
                
                # 简单的token估算
                if len(context) + len(content) > max_tokens * 4:  # 粗略估计字符数是token数的4倍
                    break
                    
                context += f"\n\n来源 {i+1} ({source}, 相似度: {similarity:.2f}):\n{content}"
                
            return context.strip()
            
        except Exception as e:
            logger.error(f"获取相关上下文失败: {str(e)}")
            return ""
            
    def save(self):
        """保存所有向量存储"""
        for store_type, store in self.stores.items():
            try:
                store.save_local(str(self.storage_dir / store_type))
                logger.info(f"向量存储 {store_type} 已保存")
            except Exception as e:
                logger.error(f"保存向量存储 {store_type} 失败: {str(e)}")
                
    def reset(self, store_type: Optional[str] = None):
        """重置向量存储
        
        Args:
            store_type: 要重置的存储类型，如果为None则重置所有
        """
        if store_type:
            store_types = [store_type]
        else:
            store_types = list(self.stores.keys())
            
        for st in store_types:
            try:
                # 创建新的存储
                self.stores[st] = self._create_new_store()
                
                # 保存
                self.stores[st].save_local(str(self.storage_dir / st))
                
                logger.info(f"向量存储 {st} 已重置")
            except Exception as e:
                logger.error(f"重置向量存储 {st} 失败: {str(e)}")
                
    async def add_prompt(self, prompt: str) -> Optional[str]:
        """添加prompt到向量存储
        
        Args:
            prompt: 要添加的prompt
            
        Returns:
            Optional[str]: 文档ID，如果添加失败则返回None
        """
        try:
            # 检查store是否存在
            if "prompts" not in self.stores or self.stores["prompts"] is None:
                logger.warning("prompts存储不存在，尝试初始化")
                self.stores["prompts"] = self._create_new_store()
                
            # 添加到向量存储
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "source": "user_submitted",
                "type": "prompt"
            }
            
            doc_ids = await self.add_texts(
                texts=[prompt],
                metadatas=[metadata],
                store_type="prompts"
            )
            
            if doc_ids and len(doc_ids) > 0:
                logger.info(f"成功添加prompt到向量存储，ID: {doc_ids[0]}")
                return doc_ids[0]
            else:
                logger.warning("添加prompt未返回文档ID")
                return None
                
        except Exception as e:
            logger.error(f"添加prompt到向量存储失败: {str(e)}")
            if settings.DEBUG:
                logger.debug("详细错误信息:", exc_info=True)
            return None
            
    def add_prompt_history(self, original_prompt: str, optimized_prompt: str) -> bool:
        """记录prompt历史到简单存储（非向量）
        
        这是一个降级方法，当向量存储不可用时使用
        
        Args:
            original_prompt: 原始prompt
            optimized_prompt: 优化后的prompt
            
        Returns:
            bool: 是否成功记录
        """
        try:
            # 创建历史记录文件
            history_dir = self.storage_dir.parent / "prompt_history"
            history_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_file = history_dir / f"prompt_history_{timestamp}.json"
            
            # 保存历史记录
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "original_prompt": original_prompt,
                    "optimized_prompt": optimized_prompt
                }, f, ensure_ascii=False, indent=2)
                
            logger.info(f"成功记录prompt历史到文件: {history_file}")
            return True
            
        except Exception as e:
            logger.error(f"记录prompt历史失败: {str(e)}")
            return False
            
    async def add_content(
        self,
        content: str,
        content_type: str = "contexts",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """添加单个内容到向量存储
        
        Args:
            content: 要添加的内容
            content_type: 内容类型
            metadata: 元数据
            
        Returns:
            Optional[str]: 文档ID，如果添加失败则返回None
        """
        try:
            # 添加到向量存储
            doc_ids = await self.add_texts(
                texts=[content],
                metadatas=[metadata] if metadata else None,
                store_type=content_type
            )
            
            if doc_ids and len(doc_ids) > 0:
                logger.info(f"成功添加内容到向量存储，ID: {doc_ids[0]}")
                return doc_ids[0]
            else:
                logger.warning(f"添加内容到向量存储未返回文档ID")
                return None
                
        except Exception as e:
            logger.error(f"添加内容到向量存储失败: {str(e)}")
            if settings.DEBUG:
                logger.debug("详细错误信息:", exc_info=True)
            return None 