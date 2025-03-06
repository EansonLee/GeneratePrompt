from typing import Dict, Any, List, Optional
import logging
import os
from datetime import datetime
from unittest.mock import Mock, MagicMock
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from config.config import settings
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import json
from pathlib import Path
import time
import asyncio
import uuid
from .cache_manager import CacheManager

# 设置日志级别
logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

class VectorStore:
    """增强的向量存储类"""
    
    _instance = None  # 单例实例
    _is_initialized = False  # 初始化标志
    _initialization_error = None  # 初始化错误信息
    _embedding_cache = {}  # 嵌入缓存
    _embedding_cache_file = Path(settings.VECTOR_DB_PATH) / "embedding_cache.json"
    
    def __new__(cls, use_mock: bool = False):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, use_mock: bool = False):
        """初始化向量存储
        
        Args:
            use_mock: 是否使用mock数据
        """
        # 避免重复初始化
        if self._is_initialized:
            return
            
        self.is_testing = settings.TESTING
        self.is_ready = False
        
        # 加载嵌入缓存
        self._load_embedding_cache()
        
        # 初始化文档缓存
        self._doc_cache = {}
        self._doc_cache_file = Path(settings.VECTOR_DB_PATH) / "doc_cache.json"
        self._load_doc_cache()
        
        # 打印配置信息
        logger.info("==================== 当前配置信息 ====================")
        logger.info(f"配置文件中的 API Key: {settings.OPENAI_API_KEY}")
        logger.info(f"配置文件中的 Base URL: {settings.OPENAI_BASE_URL}")
        logger.info(f"配置文件中的 Embedding Model: {settings.EMBEDDING_MODEL}")
        logger.info(f"调试模式: {settings.DEBUG}")
        logger.info("====================================================")
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.VECTOR_STORE_CONFIG["chunk_size"],
            chunk_overlap=settings.VECTOR_STORE_CONFIG["chunk_overlap"],
            length_function=len,
            separators=settings.VECTOR_STORE_CONFIG["separators"]
        )
        logger.info("文本分割器初始化完成")
        
        try:
            if use_mock:
                self._init_mock_data()
            else:
                self._init_real_api()
            self.is_ready = True
            self._is_initialized = True
            self._initialization_error = None
        except Exception as e:
            self._initialization_error = str(e)
            logger.error(f"向量数据库初始化失败: {str(e)}")
            if settings.DEBUG:
                logger.debug("详细错误信息:", exc_info=True)
            if use_mock or self.is_testing:
                logger.warning("回退到mock模式")
                self.is_ready = True
            else:
                raise
        
        self.prompt_history = []
    
    def is_initialized(self) -> bool:
        """检查向量数据库是否已初始化完成
        
        Returns:
            bool: 是否初始化完成
        """
        try:
            # 检查必要的属性是否存在
            required_attrs = ['embeddings', 'context_store', 'template_store']
            for attr in required_attrs:
                if not hasattr(self, attr):
                    logger.warning(f"向量存储缺少必要属性: {attr}")
                    return False
                
            # 检查存储是否可用
            if not self.context_store or not self.template_store:
                logger.warning("向量存储实例未正确初始化")
                return False
            
            return self.is_ready and self._is_initialized
            
        except Exception as e:
            logger.error(f"检查向量存储状态失败: {str(e)}")
            return False

    def get_initialization_error(self) -> Optional[str]:
        """获取初始化错误信息
        
        Returns:
            Optional[str]: 错误信息，如果没有错误则返回None
        """
        if not self._initialization_error:
            return None
        
        # 添加更多上下文信息
        error_context = {
            "error": self._initialization_error,
            "is_ready": self.is_ready,
            "is_initialized": self._is_initialized,
            "has_embeddings": hasattr(self, 'embeddings'),
            "has_context_store": hasattr(self, 'context_store'),
            "has_template_store": hasattr(self, 'template_store')
        }
        
        return json.dumps(error_context, ensure_ascii=False)

    async def wait_until_ready(self, timeout: int = 30) -> bool:
        """等待向量数据库就绪
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            bool: 是否就绪
        """
        try:
            start_time = time.time()
            check_interval = 1  # 检查间隔（秒）
            
            while not self.is_initialized() and time.time() - start_time < timeout:
                await asyncio.sleep(check_interval)
                
                # 如果有初始化错误，立即返回
                if self._initialization_error:
                    logger.error(f"向量存储初始化失败: {self._initialization_error}")
                    return False
                
                # 记录等待时间
                elapsed = time.time() - start_time
                if elapsed > timeout / 2:
                    logger.warning(f"向量存储初始化时间较长: {elapsed:.1f}秒")
                
            is_ready = self.is_initialized()
            if not is_ready:
                logger.error(f"向量存储初始化超时（{timeout}秒）")
            
            return is_ready
            
        except Exception as e:
            logger.error(f"等待向量存储就绪失败: {str(e)}")
            return False

    def _init_real_api(self):
        """初始化真实API"""
        try:
            logger.info("初始化向量数据库...")
            logger.info(f"使用的 API Base URL: {settings.OPENAI_BASE_URL}")
            logger.info(f"使用的 Embedding Model: {settings.EMBEDDING_MODEL}")
            
            # 验证API配置
            if not settings.OPENAI_API_KEY:
                raise ValueError("未设置 OPENAI_API_KEY")
                
            if not settings.OPENAI_BASE_URL:
                raise ValueError("未设置 OPENAI_BASE_URL")
            
            # 初始化嵌入模型，使用较小维度
            self.embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                openai_api_base=settings.OPENAI_BASE_URL,
                dimensions=settings.VECTOR_STORE_CONFIG.get("embedding_dimensions", 64),
                encoding_format="float"
            )
            
            # 检查是否需要测试嵌入功能
            cache_status_file = Path(settings.VECTOR_DB_PATH) / "embedding_test_status.json"
            if not cache_status_file.exists() or not hasattr(self, '_embedding_tested'):
                self._test_embeddings()
                self._embedding_tested = True
            
            # 初始化存储
            self._init_stores()
            logger.info("向量数据库初始化完成")
            
        except Exception as e:
            logger.error(f"向量数据库初始化失败: {str(e)}")
            if settings.DEBUG:
                logger.debug(f"详细错误信息: {str(e)}", exc_info=True)
            raise
    
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
            self.context_store.add_documents([doc])
            
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
            self.context_store.add_documents([doc])
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
            return self.context_store.add_texts([practice], [metadata])
        except Exception as e:
            logger.error(f"添加最佳实践失败: {str(e)}")
            raise
    
    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        search_type: str = "similarity",
        cache_key: Optional[str] = None,
        **kwargs
    ) -> List[Document]:
        """执行相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            search_type: 搜索类型 ("similarity" 或 "mmr")
            cache_key: 缓存键
            **kwargs: 额外参数
            
        Returns:
            List[Document]: 搜索结果文档列表
        """
        start_time = time.time()
        
        # 使用智能缓存管理
        if not hasattr(self, '_search_cache'):
            self._search_cache = CacheManager(
                max_size=settings.VECTOR_STORE_CONFIG.get("search_cache_size", 1000),
                ttl=settings.VECTOR_STORE_CONFIG.get("search_cache_ttl", 3600)  # 1小时缓存
            )
        
        # 检查缓存
        if cache_key:
            cached_results = self._search_cache.get(cache_key)
            if cached_results is not None:
                logger.info("使用缓存的搜索结果")
                return cached_results
            
        try:
            # 获取查询文本的嵌入向量（使用缓存）
            query_embedding = await self._get_embedding_with_cache(query)
            
            # 使用异步信号量限制并发请求数
            max_concurrent = settings.VECTOR_STORE_CONFIG.get("max_concurrent_requests", 5)
            if not hasattr(self, '_semaphore'):
                self._semaphore = asyncio.Semaphore(max_concurrent)
                
            async with self._semaphore:
                # 根据搜索类型选择搜索方法
                if search_type == "mmr":
                    results = await asyncio.to_thread(
                        self.context_store.max_marginal_relevance_search,
                        query,
                        k=min(k, settings.VECTOR_STORE_CONFIG.get("max_results", 5)),
                        fetch_k=min(2*k, settings.VECTOR_STORE_CONFIG.get("max_fetch", 10)),
                        lambda_mult=settings.VECTOR_STORE_CONFIG.get("mmr_lambda", 0.7),
                        **kwargs
                    )
                else:
                    results = await asyncio.to_thread(
                        self.context_store.similarity_search,
                        query,
                        k=min(k, settings.VECTOR_STORE_CONFIG.get("max_results", 5)),
                        **kwargs
                    )
                    
                # 检查查询时间
                query_time = time.time() - start_time
                if query_time > settings.VECTOR_STORE_CONFIG.get("slow_query_threshold", 5.0):
                    logger.warning(f"检测到慢查询: {query_time:.2f}s, query={query}")
                    
                # 缓存结果
                if cache_key:
                    self._search_cache.set(cache_key, results)
                    logger.info(f"已缓存搜索结果，键值: {cache_key}")
                    
                return results
                
        except Exception as e:
            logger.error(f"相似度搜索失败: {str(e)}")
            return []
    
    def search_prompt_history(
        self,
        query: str,
        k: int = 4,
        search_type: str = "similarity"
    ) -> List[str]:
        """搜索提示优化历史
        
        Args:
            query: 搜索查询
            k: 返回结果数量
            search_type: 搜索类型，可选 "similarity" 或 "mmr"
            
        Returns:
            List[str]: 历史优化提示列表
        """
        return self.similarity_search(
            query,
            k=k,
            filter_dict={"type": "prompt_history"},
            search_type=search_type
        )
    
    def search_react_code(
        self,
        query: str,
        k: int = 4,
        search_type: str = "similarity"
    ) -> List[str]:
        """搜索React代码示例
        
        Args:
            query: 搜索查询
            k: 返回结果数量
            search_type: 搜索类型，可选 "similarity" 或 "mmr"
            
        Returns:
            List[str]: 相关代码示例列表
        """
        return self.similarity_search(
            query,
            k=k,
            filter_dict={"type": "react_code"},
            search_type=search_type
        )
    
    def search_best_practices(
        self,
        query: str,
        k: int = 4,
        search_type: str = "similarity"
    ) -> List[str]:
        """搜索最佳实践
        
        Args:
            query: 搜索查询
            k: 返回结果数量
            search_type: 搜索类型，可选 "similarity" 或 "mmr"
            
        Returns:
            List[str]: 相关最佳实践列表
        """
        return self.similarity_search(
            query,
            k=k,
            filter_dict={"type": "best_practices"},
            search_type=search_type
        )
    
    async def search_contexts(self, query: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索上下文
        
        Args:
            query: 搜索查询
            limit: 返回结果数量限制
            
        Returns:
            List[Dict[str, Any]]: 搜索结果列表
        """
        try:
            if not self.context_store:
                raise ValueError("上下文存储未初始化")
                
            if query:
                results = await asyncio.to_thread(
                    self.context_store.similarity_search_with_score,
                    query,
                    k=limit
                )
                return [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score
                    }
                    for doc, score in results
                ]
            else:
                # 使用同步方法获取所有文档
                docs = list(self.context_store.docstore._dict.values())
                return [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in docs[:limit]
                ]
                
        except Exception as e:
            logger.error(f"搜索上下文失败: {str(e)}")
            raise
            
    async def search_templates(self, query: str = None, limit: int = 3) -> List[str]:
        """搜索模板
        
        Args:
            query: 搜索查询
            limit: 返回结果数量限制
            
        Returns:
            List[str]: 模板列表
        """
        try:
            if not self.template_store:
                raise ValueError("模板存储未初始化")
                
            if query:
                results = await asyncio.to_thread(
                    self.template_store.similarity_search_with_score,
                    query,
                    k=limit
                )
                return [doc.page_content for doc, _ in results]
            else:
                # 使用同步方法获取所有文档
                docs = list(self.template_store.docstore._dict.values())
                return [doc.page_content for doc in docs[:limit]]
                
        except Exception as e:
            logger.error(f"搜索模板失败: {str(e)}")
            raise
        
    def _load_doc_cache(self):
        """加载文档缓存"""
        try:
            # 使用智能缓存管理
            if not hasattr(self, '_doc_cache_manager'):
                self._doc_cache_manager = CacheManager(
                    max_size=settings.VECTOR_STORE_CONFIG.get("doc_cache_max_size", 5000),
                    ttl=settings.VECTOR_STORE_CONFIG.get("doc_cache_ttl", 86400 * 30)  # 30天缓存
                )
            
            if self._doc_cache_file.exists():
                with open(self._doc_cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    for key, value in cache_data.items():
                        self._doc_cache_manager.set(key, value)
                logger.info(f"已加载{len(cache_data)}条文档缓存")
        except Exception as e:
            logger.warning(f"加载文档缓存失败: {str(e)}")
        
    def _save_doc_cache(self):
        """保存文档缓存"""
        try:
            if hasattr(self, '_doc_cache_manager'):
                self._doc_cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_data = {
                    key: value for key, value in self._doc_cache_manager.cache.items()
                    if isinstance(value, tuple) and len(value) == 2
                }
                with open(self._doc_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f)
                logger.info(f"已保存{len(cache_data)}条文档缓存")
        except Exception as e:
            logger.warning(f"保存文档缓存失败: {str(e)}")

    def _get_doc_cache_key(self, text: str, metadata: Optional[Dict] = None) -> str:
        """生成文档缓存键
        
        Args:
            text: 文本内容
            metadata: 元数据
            
        Returns:
            str: 缓存键
        """
        import hashlib
        # 组合文本和元数据生成唯一键
        cache_data = f"{text}_{json.dumps(metadata, sort_keys=True) if metadata else ''}"
        return hashlib.md5(cache_data.encode('utf-8')).hexdigest()

    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """添加文本到向量存储
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
            ids: ID列表
            
        Returns:
            List[str]: 添加的文档ID列表
        """
        try:
            if not texts:
                logger.warning("没有要添加的文本")
                return []

            # 检查文档缓存
            doc_ids = []
            uncached_texts = []
            uncached_metadatas = []
            uncached_indices = []

            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas else None
                cache_key = self._get_doc_cache_key(text, metadata)
                
                if cache_key in self._doc_cache:
                    # 使用缓存的ID
                    doc_ids.append(self._doc_cache[cache_key])
                    logger.debug(f"使用缓存的文档ID: {cache_key}")
                else:
                    # 收集未缓存的文本
                    uncached_texts.append(text)
                    if metadatas:
                        uncached_metadatas.append(metadata)
                    uncached_indices.append(i)

            # 如果有未缓存的文本，则处理它们
            if uncached_texts:
                if settings.VECTOR_STORE_CONFIG.get("enable_semantic_deduplication", False):
                    # 执行语义去重
                    uncached_texts, uncached_metadatas = await self._semantic_deduplication(
                        uncached_texts, 
                        uncached_metadatas if uncached_metadatas else None
                    )

                # 生成新的ID
                new_ids = [str(uuid.uuid4()) for _ in uncached_texts]

                # 添加时间戳到元数据
                if uncached_metadatas:
                    for metadata in uncached_metadatas:
                        metadata["timestamp"] = datetime.now().isoformat()
                else:
                    uncached_metadatas = [{"timestamp": datetime.now().isoformat()} for _ in uncached_texts]

                # 使用批量嵌入处理
                embeddings = await self._get_embeddings_batch_with_cache(uncached_texts)
                
                # 批量添加到FAISS
                new_doc_ids = await asyncio.to_thread(
                    self.context_store.add_embeddings,
                    embeddings=embeddings,
                    texts=uncached_texts,
                    metadatas=uncached_metadatas,
                    ids=new_ids
                )

                # 更新文档缓存
                for i, text in enumerate(uncached_texts):
                    cache_key = self._get_doc_cache_key(
                        text, 
                        uncached_metadatas[i] if uncached_metadatas else None
                    )
                    self._doc_cache[cache_key] = new_doc_ids[i]

                # 异步保存文档缓存
                asyncio.create_task(self._async_save_doc_cache())

                # 合并ID列表
                for i, idx in enumerate(uncached_indices):
                    doc_ids.insert(idx, new_doc_ids[i])

            # 保存向量存储
            await asyncio.to_thread(
                self.context_store.save_local,
                folder_path=str(Path(settings.VECTOR_DB_PATH) / "contexts")
            )
            
            logger.info(f"成功添加 {len(texts)} 个文本到向量存储")
            return doc_ids
            
        except Exception as e:
            logger.error(f"添加文本到向量存储失败: {str(e)}")
            raise

    async def _async_save_doc_cache(self):
        """异步保存文档缓存"""
        try:
            await asyncio.to_thread(self._save_doc_cache)
        except Exception as e:
            logger.error(f"异步保存文档缓存失败: {str(e)}")

    async def _get_embeddings_batch_with_cache(self, texts: List[str]) -> List[List[float]]:
        """批量获取带缓存的嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        # 使用智能缓存管理
        if not hasattr(self, '_cache_manager'):
            self._cache_manager = CacheManager(
                max_size=settings.VECTOR_STORE_CONFIG.get("cache_max_size", 10000),
                ttl=settings.VECTOR_STORE_CONFIG.get("cache_ttl", 86400 * 7)  # 7天缓存
            )
        
        # 检查哪些文本需要新的嵌入
        cache_keys = [self._get_cache_key(text) for text in texts]
        uncached_indices = []
        cached_embeddings = []
        
        # 批量检查缓存
        for i, key in enumerate(cache_keys):
            cached_value = self._cache_manager.get(key)
            if cached_value is None:
                uncached_indices.append(i)
                cached_embeddings.append(None)
            else:
                cached_embeddings[i] = cached_value
        
        if uncached_indices:
            # 优化批处理大小
            batch_size = min(
                settings.VECTOR_STORE_CONFIG.get("batch_size", 32),
                len(uncached_indices)
            )
            
            # 收集未缓存的文本
            uncached_texts = [texts[i] for i in uncached_indices]
            all_new_embeddings = []
            
            # 使用动态批处理大小
            for i in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[i:i + batch_size]
                
                # 对批次中的文本进行预处理
                processed_texts = [
                    text[:settings.VECTOR_STORE_CONFIG.get("max_text_length", 1000)]
                    for text in batch_texts
                ]
                
                # 批量获取嵌入向量
                try:
                    batch_embeddings = self.embeddings.embed_documents(
                        processed_texts
                    )
                    all_new_embeddings.extend(batch_embeddings)
                    
                    # 更新缓存
                    for text, embedding in zip(batch_texts, batch_embeddings):
                        cache_key = self._get_cache_key(text)
                        self._cache_manager.set(cache_key, embedding)
                        
                except Exception as e:
                    logger.error(f"批量获取嵌入向量失败: {str(e)}")
                    # 使用零向量作为后备
                    dim = settings.VECTOR_STORE_CONFIG.get("embedding_dimensions", 64)
                    zero_embeddings = [[0.0] * dim for _ in batch_texts]
                    all_new_embeddings.extend(zero_embeddings)
            
            # 将新的嵌入向量插入到正确的位置
            for i, idx in enumerate(uncached_indices):
                cached_embeddings[idx] = all_new_embeddings[i]
        
        return cached_embeddings

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的相似度
        
        Args:
            text1: 第一段文本
            text2: 第二段文本
            
        Returns:
            float: 相似度分数 (0-1)
        """
        try:
            # 添加相似度阈值
            threshold = settings.VECTOR_STORE_CONFIG.get("similarity_threshold", 0.95)
            if threshold < 0.5:
                return 0.0  # 避免不必要的计算
            
            # 检查缓存
            cache_key = f"sim_{self._get_cache_key(text1)}_{self._get_cache_key(text2)}"
            
            # 使用智能缓存管理
            if not hasattr(self, '_similarity_cache'):
                self._similarity_cache = CacheManager(
                    max_size=settings.VECTOR_STORE_CONFIG.get("similarity_cache_size", 5000),
                    ttl=settings.VECTOR_STORE_CONFIG.get("similarity_cache_ttl", 86400 * 3)  # 3天缓存
                )
            
            cached_similarity = self._similarity_cache.get(cache_key)
            if cached_similarity is not None:
                return cached_similarity
            
            # 获取嵌入向量（使用缓存）
            embedding1 = self.embeddings.embed_query(text1)
            embedding2 = self.embeddings.embed_query(text2)
            
            # 计算余弦相似度
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            
            # 缓存相似度结果
            self._similarity_cache.set(cache_key, float(similarity))
            return float(similarity)
            
        except Exception as e:
            logger.error(f"计算相似度失败: {str(e)}")
            return 0.0

    def _test_embeddings(self):
        """测试嵌入功能"""
        try:
            # 检查缓存文件是否存在且未过期
            cache_status_file = Path(settings.VECTOR_DB_PATH) / "embedding_test_status.json"
            current_time = time.time()
            
            if cache_status_file.exists():
                try:
                    with open(cache_status_file, 'r', encoding='utf-8') as f:
                        status = json.load(f)
                    # 检查缓存是否在24小时内
                    if current_time - status.get('last_test_time', 0) < 86400:
                        logger.info("使用缓存的嵌入测试状态")
                        return True
                except Exception:
                    pass
            
            # 使用最短的测试文本和批处理
            test_texts = ["a", "b", "c"]  # 使用多个短文本进行批量测试
            
            # 使用dimension_reduction参数减少维度
            test_embeddings = self.embeddings.embed_documents(test_texts)
            
            # 验证所有嵌入向量
            for embedding in test_embeddings:
                if len(embedding) <= 0:
                    raise ValueError("嵌入维度无效")
                
            # 更新测试状态缓存
            cache_status_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_status_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'last_test_time': current_time,
                    'test_dim': len(test_embeddings[0]),
                    'model': settings.EMBEDDING_MODEL,
                    'batch_size': len(test_texts)
                }, f)
            
            logger.info(f"嵌入测试成功，使用{len(test_texts)}个文本进行批量测试")
            return True
            
        except Exception as e:
            logger.error(f"嵌入测试失败: {str(e)}")
            raise

    def _init_stores(self):
        """初始化向量存储"""
        try:
            # 配置 OpenAI Embeddings，使用较小维度
            self.embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                openai_api_base=settings.OPENAI_BASE_URL,
                dimensions=settings.VECTOR_STORE_CONFIG.get("embedding_dimensions", 64),
                encoding_format="float"
            )

            # 初始化上下文存储
            context_path = Path(settings.VECTOR_DB_PATH) / "contexts"
            context_path.mkdir(parents=True, exist_ok=True)
            
            try:
                self.context_store = FAISS.load_local(
                    str(context_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("成功加载现有的上下文向量存储")
            except Exception as e:
                logger.warning(f"加载上下文向量存储失败，创建新的存储: {str(e)}")
                # 使用较小的初始文档
                self.context_store = FAISS.from_texts(
                    ["init"],
                    self.embeddings,
                    metadatas=[{"type": "context", "timestamp": datetime.now().isoformat()}]
                )
                self.context_store.save_local(str(context_path))

            # 初始化模板存储
            template_path = Path(settings.VECTOR_DB_PATH) / "templates"
            template_path.mkdir(parents=True, exist_ok=True)
            
            try:
                self.template_store = FAISS.load_local(
                    str(template_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("成功加载现有的模板向量存储")
            except Exception as e:
                logger.warning(f"加载模板向量存储失败，创建新的存储: {str(e)}")
                # 使用较小的初始文档
                self.template_store = FAISS.from_texts(
                    ["init"],
                    self.embeddings,
                    metadatas=[{"type": "template", "timestamp": datetime.now().isoformat()}]
                )
                self.template_store.save_local(str(template_path))

            self._is_initialized = True
            logger.info("向量存储初始化成功")
            
        except Exception as e:
            self._initialization_error = f"初始化向量存储失败: {str(e)}"
            logger.error(self._initialization_error)
            raise
        
    def _init_mock_data(self):
        """初始化mock数据"""
        logger.info("初始化mock数据...")
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        self.embeddings = mock_embeddings
        
        mock_contexts = MagicMock()
        mock_contexts.similarity_search.return_value = [
            Document(page_content="测试上下文1"),
            Document(page_content="测试上下文2")
        ]
        self.context_store = mock_contexts
        
        mock_templates = MagicMock()
        mock_templates.similarity_search.return_value = [
            Document(page_content="测试模板1"),
            Document(page_content="测试模板2")
        ]
        self.templates_store = mock_templates
        logger.info("Mock数据初始化完成")

    async def _semantic_deduplication(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> tuple[List[str], Optional[List[Dict[str, Any]]]]:
        """语义去重
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
            
        Returns:
            Tuple[List[str], Optional[List[Dict[str, Any]]]]: 去重后的文本和元数据
        """
        if not texts:
            return texts, metadatas
            
        # 使用较低的阈值进行快速去重
        threshold = settings.VECTOR_STORE_CONFIG.get("deduplication_threshold", 0.95)
        min_length = settings.VECTOR_STORE_CONFIG.get("min_dedup_length", 10)
        
        # 对短文本跳过去重
        if all(len(text) < min_length for text in texts):
            return texts, metadatas
        
        # 使用批量嵌入处理
        embeddings = await self._get_embeddings_batch_with_cache(texts)
        
        # 使用numpy进行向量化计算
        embeddings_array = np.array(embeddings)
        norms = np.linalg.norm(embeddings_array, axis=1)
        similarity_matrix = np.dot(embeddings_array, embeddings_array.T) / np.outer(norms, norms)
        
        # 使用掩码避免自身比较
        np.fill_diagonal(similarity_matrix, 0)
        
        # 找出需要保留的索引
        keep_indices = []
        for i in range(len(texts)):
            # 如果当前文本与已保留的文本相似度都低于阈值，则保留
            if not keep_indices or not any(similarity_matrix[i][j] > threshold for j in keep_indices):
                keep_indices.append(i)
        
        # 保留未重复的文本和元数据
        unique_texts = [texts[i] for i in keep_indices]
        unique_metadatas = [metadatas[i] for i in keep_indices] if metadatas else None
        
        # 记录去重结果
        removed_count = len(texts) - len(unique_texts)
        if removed_count > 0:
            logger.info(f"语义去重移除了 {removed_count} 个重复文本")
        
        return unique_texts, unique_metadatas
        
    async def _rerank_results(
        self,
        query: str,
        results: List[Any],
        top_k: int = 3
    ) -> List[Any]:
        """重排序搜索结果
        
        Args:
            query: 查询文本
            results: 原始结果列表
            top_k: 重排序后保留的结果数量
            
        Returns:
            List[Any]: 重排序后的结果
        """
        if not results or len(results) <= 1:
            return results
            
        # 计算查询向量
        query_embedding = self.embeddings.embed_query(query)
        
        # 计算每个结果的得分
        scored_results = []
        for doc in results:
            doc_embedding = self.embeddings.embed_query(doc.page_content)
            
            # 计算语义相似度
            semantic_score = self.calculate_similarity(query_embedding, doc_embedding)
            
            # 计算时间衰减因子（如果有时间戳）
            time_decay = 1.0
            if doc.metadata and "timestamp" in doc.metadata:
                age = (datetime.now() - datetime.fromisoformat(doc.metadata["timestamp"])).total_seconds()
                time_decay = 1.0 / (1.0 + age / 86400)  # 24小时衰减
                
            # 计算最终得分
            final_score = semantic_score * 0.7 + time_decay * 0.3
            
            scored_results.append((doc, final_score))
            
        # 按得分排序并返回top_k结果
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_results[:top_k]]
        
    @staticmethod
    def calculate_similarity(vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            float: 相似度得分
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _load_embedding_cache(self):
        """加载嵌入缓存"""
        try:
            if self._embedding_cache_file.exists():
                import gzip
                with gzip.open(str(self._embedding_cache_file) + '.gz', 'rt', encoding='utf-8') as f:
                    self._embedding_cache = json.load(f)
                logger.info(f"已加载{len(self._embedding_cache)}条嵌入缓存")
                
                # 验证缓存数据的维度
                if self._embedding_cache:
                    first_key = next(iter(self._embedding_cache))
                    embedding_dim = len(self._embedding_cache[first_key])
                    expected_dim = settings.VECTOR_STORE_CONFIG.get("embedding_dimensions", 64)
                    if embedding_dim != expected_dim:
                        logger.warning(f"缓存的嵌入维度({embedding_dim})与配置维度({expected_dim})不匹配，清空缓存")
                        self._embedding_cache = {}
                    
        except Exception as e:
            logger.warning(f"加载嵌入缓存失败: {str(e)}")
            self._embedding_cache = {}
            
    def _save_embedding_cache(self):
        """保存嵌入缓存"""
        try:
            # 确保目录存在
            self._embedding_cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用gzip压缩存储
            import gzip
            with gzip.open(str(self._embedding_cache_file) + '.gz', 'wt', encoding='utf-8') as f:
                # 转换numpy数组为列表以便JSON序列化
                cache_data = {
                    key: embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                    for key, embedding in self._embedding_cache.items()
                }
                json.dump(cache_data, f)
            logger.info(f"已压缩保存{len(self._embedding_cache)}条嵌入缓存")
        except Exception as e:
            logger.warning(f"保存嵌入缓存失败: {str(e)}")
            
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键
        
        Args:
            text: 文本内容
            
        Returns:
            str: 缓存键
        """
        import hashlib
        # 使用文本的hash作为缓存键，限制文本长度以提高效率
        max_length = settings.VECTOR_STORE_CONFIG.get("max_cache_key_length", 1000)
        text = text[:max_length]
        return hashlib.md5(text.encode('utf-8')).hexdigest()
        
    async def _get_embedding_with_cache(self, text: str) -> List[float]:
        """获取带缓存的嵌入向量
        
        Args:
            text: 文本内容
            
        Returns:
            List[float]: 嵌入向量
        """
        cache_key = self._get_cache_key(text)
        
        # 检查缓存
        if cache_key in self._embedding_cache:
            embedding = self._embedding_cache[cache_key]
            # 验证维度
            if len(embedding) == settings.VECTOR_STORE_CONFIG.get("embedding_dimensions", 64):
                logger.debug(f"使用缓存的嵌入向量: {cache_key}")
                return embedding
            else:
                # 维度不匹配，删除缓存
                del self._embedding_cache[cache_key]
        
        # 获取新的嵌入向量
        embedding = self.embeddings.embed_query(
            text
        )
        
        # 更新缓存
        self._embedding_cache[cache_key] = embedding
        
        # 使用LRU策略管理缓存大小
        max_size = settings.VECTOR_STORE_CONFIG["embedding_cache"]["max_size"]
        if len(self._embedding_cache) > max_size:
            # 删除最早添加的20%条目
            num_to_delete = max_size // 5
            keys_to_delete = list(self._embedding_cache.keys())[:num_to_delete]
            for key in keys_to_delete:
                del self._embedding_cache[key]
        
        # 异步保存缓存
        if len(self._embedding_cache) % 100 == 0:  # 每100次更新保存一次
            asyncio.create_task(self._async_save_cache())
        
        return embedding
        
    async def _async_save_cache(self):
        """异步保存缓存"""
        try:
            await asyncio.to_thread(self._save_embedding_cache)
        except Exception as e:
            logger.error(f"异步保存缓存失败: {str(e)}")
        
    def add_template(self, template: str) -> str:
        """添加模板到向量存储
        
        Args:
            template: 模板内容
            
        Returns:
            str: 添加的文档ID
        """
        try:
            # 创建元数据
            metadata = {
                "type": "template",
                "timestamp": datetime.now().isoformat()
            }
            
            # 添加到向量存储
            doc_ids = self.template_store.add_texts(
                [template],
                [metadata]
            )
            
            # 保存到本地
            template_path = Path(settings.VECTOR_DB_PATH) / "templates"
            self.template_store.save_local(str(template_path))
            
            logger.info("成功添加模板到向量存储")
            return doc_ids[0]
            
        except Exception as e:
            logger.error(f"添加模板失败: {str(e)}")
            raise

    def verify_insertion(self, content: str, store_type: str = "templates") -> bool:
        """验证内容是否成功插入到向量存储
        
        Args:
            content: 要验证的内容
            store_type: 存储类型 ("templates" 或 "contexts")
            
        Returns:
            bool: 是否成功插入
        """
        try:
            # 选择存储
            store = self.template_store if store_type == "templates" else self.context_store
            
            # 搜索刚插入的内容
            results = store.similarity_search(
                content,
                k=1
            )
            
            # 检查是否找到匹配的内容
            if results and len(results) > 0:
                # 获取嵌入向量
                content_embedding = self.embeddings.embed_query(content)
                result_embedding = self.embeddings.embed_query(results[0].page_content)
                
                # 计算余弦相似度
                similarity = np.dot(content_embedding, result_embedding) / (
                    np.linalg.norm(content_embedding) * np.linalg.norm(result_embedding)
                )
                
                # 使用较高的阈值确保精确匹配
                return float(similarity) > 0.98
                
            return False
            
        except Exception as e:
            logger.error(f"验证插入失败: {str(e)}")
            return False
        