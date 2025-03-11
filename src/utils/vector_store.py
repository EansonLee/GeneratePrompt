from typing import Dict, Any, List, Optional, Tuple
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
    """增强的向量存储类，支持多模态处理"""
    
    _instance = None  # 单例实例
    _is_initialized = False  # 初始化标志
    _is_ready = False  # 就绪标志
    _initialization_error = None  # 初始化错误信息
    _embedding_cache = {}  # 文本嵌入缓存
    _embedding_cache_file = Path(settings.VECTOR_DB_PATH) / "embedding_cache.json"
    _image_embedding_cache = {}  # 图片嵌入缓存
    _image_embedding_cache_file = Path(settings.VECTOR_DB_PATH) / "image_embedding_cache.json"
    _vision_model = None  # 视觉模型实例
    
    def __new__(cls, use_mock: bool = False):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, use_mock: bool = False):
        """初始化向量存储
        
        Args:
            use_mock: 是否使用模拟数据
        """
        # 如果已经初始化过，直接返回
        if hasattr(self, '_stores'):
            return
            
        self.use_mock = use_mock
        self._stores = {}
        
        try:
            # 确保向量存储目录存在
            Path(settings.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)
            
            # 初始化文本嵌入
            self._init_text_embeddings()
            
            # 初始化存储（必须在文本嵌入初始化之后）
            self._init_stores()
            
            # 验证存储是否正确初始化
            if not all([
                hasattr(self, 'context_store'),
                hasattr(self, 'template_store'),
                hasattr(self, 'prompts_store'),
                hasattr(self, 'designs_store')
            ]):
                raise ValueError("存储初始化失败")
            
            # 初始化视觉模型
            self._init_vision_model()
            
            # 加载缓存
            self._load_embedding_cache()
            self._load_image_embedding_cache()
            self._load_doc_cache()
            
            if use_mock:
                self._init_mock_data()
                
            VectorStore._is_initialized = True
            VectorStore._is_ready = True
            logger.info("向量存储初始化完成")
            
        except Exception as e:
            VectorStore._initialization_error = str(e)
            logger.error(f"向量存储初始化失败: {str(e)}", exc_info=True)
            raise

    def is_initialized(self) -> bool:
        """检查向量数据库是否已初始化完成
        
        Returns:
            bool: 是否初始化完成
        """
        try:
            # 检查必要的属性是否存在
            required_attrs = ['embeddings', 'context_store', 'template_store', 'prompts_store', 'designs_store']
            for attr in required_attrs:
                if not hasattr(self, attr):
                    logger.warning(f"向量存储缺少必要属性: {attr}")
                    return False
                
            # 检查存储实例是否可用
            stores = [self.context_store, self.template_store, self.prompts_store, self.designs_store]
            if not all(stores):
                logger.warning("一个或多个向量存储实例未正确初始化")
                return False
            
            # 同时检查初始化和就绪状态
            return self._is_initialized and self._is_ready
            
        except Exception as e:
            logger.error(f"检查向量存储状态失败: {str(e)}")
            return False

    def get_initialization_error(self) -> Optional[str]:
        """获取初始化错误信息
        
        Returns:
            Optional[str]: 错误信息，如果没有错误则返回None
        """
        try:
            if not self._initialization_error:
                return None
            
            # 添加更多上下文信息
            error_context = {
                "error": self._initialization_error,
                "is_initialized": self._is_initialized,
                "is_ready": self._is_ready,
                "has_embeddings": hasattr(self, 'embeddings'),
                "has_context_store": hasattr(self, 'context_store'),
                "has_template_store": hasattr(self, 'template_store'),
                "has_prompts_store": hasattr(self, 'prompts_store'),
                "has_designs_store": hasattr(self, 'designs_store')
            }
            
            return json.dumps(error_context, ensure_ascii=False)
        except Exception as e:
            logger.error(f"获取初始化错误信息失败: {str(e)}")
            return str(self._initialization_error) if self._initialization_error else None

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

    def _init_text_embeddings(self):
        """初始化文本嵌入模型"""
        try:
            # 初始化文本嵌入模型
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",  # 使用稳定的文本嵌入模型
                openai_api_key=settings.OPENAI_API_KEY,
                openai_api_base=settings.OPENAI_BASE_URL,
                chunk_size=settings.VECTOR_STORE_CONFIG["chunk_size"],
                max_retries=settings.VECTOR_STORE_CONFIG["max_retries"],
                timeout=settings.VECTOR_STORE_CONFIG["timeout"]
            )
            logger.info("文本嵌入模型初始化成功")
        except Exception as e:
            logger.error(f"初始化文本嵌入模型失败: {str(e)}")
            raise
    
    def _init_vision_model(self):
        """初始化视觉模型"""
        try:
            from openai import OpenAI
            import time
            
            if self.use_mock:
                self._vision_model = Mock()
                logger.info("使用Mock视觉模型")
                return
                
            # 初始化视觉模型
            self._vision_model = OpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL
            )
            
            # 测试模型连接
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    test_response = self._vision_model.models.list()
                    if not test_response:
                        raise ValueError("无法连接到OpenAI API")
                    
                    # 验证是否支持gpt-4o模型
                    available_models = [model.id for model in test_response]
                    if "gpt-4o" not in available_models:
                        logger.warning("gpt-4o模型不可用，请确保API支持该模型")
                    
                    logger.info("视觉模型初始化成功")
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"视觉模型连接尝试 {attempt + 1} 失败: {str(e)}")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避
                    else:
                        raise
                        
        except Exception as e:
            logger.error(f"初始化视觉模型失败: {str(e)}", exc_info=True)
            raise
    
    async def process_content(self, content: Any, content_type: str = "text") -> List[float]:
        """处理内容并返回嵌入向量
        
        Args:
            content: 要处理的内容
            content_type: 内容类型 ("text" 或 "image")
            
        Returns:
            List[float]: 内容的嵌入向量
        """
        try:
            if content_type == "image":
                if not isinstance(content, bytes):
                    raise ValueError("图片内容必须是bytes类型")
                    
                logger.info("开始处理图片内容")
                return await self._get_image_embedding(content)
                
            elif content_type == "text":
                if not isinstance(content, str):
                    raise ValueError("文本内容必须是字符串类型")
                    
                logger.info("开始处理文本内容")
                return await self._get_embedding_with_cache(content)
                
            else:
                raise ValueError(f"不支持的内容类型: {content_type}")
                
        except Exception as e:
            logger.error(f"处理内容失败: {str(e)}", exc_info=True)
            raise

    async def add_content(
        self,
        content: Any,
        content_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """添加内容到向量存储
        
        Args:
            content: 要添加的内容
            content_type: 内容类型 ("text" 或 "image")
            metadata: 内容元数据
            
        Returns:
            str: 文档ID
        """
        try:
            # 生成文档ID
            doc_id = str(uuid.uuid4())
            
            # 处理元数据
            if metadata is None:
                metadata = {}
            metadata["content_type"] = content_type
            metadata["timestamp"] = datetime.now().isoformat()
            
            # 获取内容的嵌入向量
            embedding = await self.process_content(content, content_type)
            
            if content_type == "image":
                # 存储图片内容
                self._store_image_content(content, embedding, metadata, doc_id)
                logger.info(f"图片内容已添加到向量存储: {doc_id}")
            else:
                # 存储文本内容
                texts = [content]
                metadatas = [metadata]
                ids = [doc_id]
                
                await self.add_texts(texts, metadatas, ids)
                logger.info(f"文本内容已添加到向量存储: {doc_id}")
            
            return doc_id
            
        except Exception as e:
            logger.error(f"添加内容失败: {str(e)}", exc_info=True)
            raise

    def _store_image_content(
        self,
        image_data: bytes,
        embedding: List[float],
        metadata: Dict[str, Any],
        doc_id: str
    ):
        """存储图片内容
        
        Args:
            image_data: 图片数据
            embedding: 图片的嵌入向量
            metadata: 图片元数据
            doc_id: 文档ID
        """
        try:
            # 计算图片hash
            import hashlib
            image_hash = hashlib.md5(image_data).hexdigest()
            
            # 更新元数据
            metadata.update({
                "image_hash": image_hash,
                "vector_dim": len(embedding),
                "doc_id": doc_id
            })
            
            # 存储到向量数据库
            self.image_store.add_embeddings(
                [embedding],
                [metadata],
                [doc_id]
            )
            
            # 保存到缓存
            self._image_embedding_cache[image_hash] = embedding
            
            # 异步保存缓存
            asyncio.create_task(self._async_save_image_cache())
            
            logger.info(f"图片内容已存储: hash={image_hash}, doc_id={doc_id}")
            
        except Exception as e:
            logger.error(f"存储图片内容失败: {str(e)}", exc_info=True)
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
    
    async def search_contexts(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索上下文
        
        Args:
            query: 查询文本
            limit: 返回结果数量限制
            
        Returns:
            List[Dict[str, Any]]: 搜索结果
        """
        try:
            # 检查向量数据库是否已初始化
            if not self.is_ready:
                logger.warning("向量数据库尚未初始化完成")
                return []
                
            # 搜索向量数据库
            docs = await self.prompts_store.asimilarity_search(query, k=limit)
            
            # 转换结果
            results = []
            for doc in docs:
                results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata
                })
                
            return results
            
        except Exception as e:
            logger.error(f"搜索上下文失败: {str(e)}")
            return []
    
    async def search_texts(
        self, 
        query: str, 
        limit: int = 5,
        search_type: str = "similarity"
    ) -> List[Dict[str, Any]]:
        """搜索文本
        
        Args:
            query: 查询文本
            limit: 返回结果数量限制
            search_type: 搜索类型 (similarity/mmr/hybrid)
            
        Returns:
            List[Dict[str, Any]]: 搜索结果
        """
        try:
            # 检查向量数据库是否已初始化
            if not self.is_ready:
                logger.warning("向量数据库尚未初始化完成")
                return []
            
            # 根据搜索类型选择搜索方法
            docs = []
            if search_type == "similarity":
                # 相似度搜索
                docs = await self.prompts_store.asimilarity_search(query, k=limit)
            elif search_type == "mmr":
                # 最大边际相关性搜索
                docs = await self.prompts_store.amax_marginal_relevance_search(
                    query, k=limit, fetch_k=limit*2
                )
            elif search_type == "hybrid":
                # 混合搜索 (先进行相似度搜索，再进行MMR过滤)
                similarity_docs = await self.prompts_store.asimilarity_search(query, k=limit*2)
                
                # 如果结果数量足够，进行MMR过滤
                if len(similarity_docs) > limit:
                    # 提取文本内容
                    texts = [doc.page_content for doc in similarity_docs]
                    
                    # 计算嵌入
                    embeddings = await self.embeddings.aembed_documents(texts)
                    
                    # 计算查询嵌入
                    query_embedding = await self.embeddings.aembed_query(query)
                    
                    # 进行MMR过滤
                    mmr_indices = self._mmr(
                        query_embedding, embeddings, 
                        k=limit, lambda_mult=0.5
                    )
                    
                    # 根据MMR索引选择文档
                    docs = [similarity_docs[i] for i in mmr_indices]
                else:
                    docs = similarity_docs
            else:
                # 默认使用相似度搜索
                logger.warning(f"未知的搜索类型: {search_type}，使用默认的相似度搜索")
                docs = await self.prompts_store.asimilarity_search(query, k=limit)
            
            # 转换结果
            results = []
            for doc in docs:
                results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata
                })
                
            return results
            
        except Exception as e:
            logger.error(f"搜索文本失败: {str(e)}")
            return []
    
    def _mmr(
        self, 
        query_embedding: List[float], 
        embeddings: List[List[float]], 
        k: int = 5, 
        lambda_mult: float = 0.5
    ) -> List[int]:
        """最大边际相关性算法
        
        Args:
            query_embedding: 查询嵌入
            embeddings: 文档嵌入列表
            k: 返回结果数量
            lambda_mult: 多样性权重
            
        Returns:
            List[int]: 选择的文档索引
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 转换为numpy数组
        query_embedding = np.array(query_embedding).reshape(1, -1)
        embeddings = np.array(embeddings)
        
        # 计算文档与查询的相似度
        sim_query = cosine_similarity(embeddings, query_embedding).reshape(-1)
        
        # 初始化已选择和未选择的索引
        selected_indices = []
        unselected_indices = list(range(len(embeddings)))
        
        # 选择第一个文档（与查询最相似的文档）
        first_idx = np.argmax(sim_query)
        selected_indices.append(first_idx)
        unselected_indices.remove(first_idx)
        
        # 迭代选择剩余的文档
        for _ in range(min(k - 1, len(embeddings) - 1)):
            # 如果没有未选择的文档，退出循环
            if not unselected_indices:
                break
                
            # 计算未选择文档与已选择文档的最大相似度
            sim_selected = np.max(
                cosine_similarity(
                    embeddings[unselected_indices], 
                    embeddings[selected_indices]
                ), 
                axis=1
            )
            
            # 计算MMR得分
            mmr_scores = lambda_mult * sim_query[unselected_indices] - (1 - lambda_mult) * sim_selected
            
            # 选择MMR得分最高的文档
            next_idx = unselected_indices[np.argmax(mmr_scores)]
            selected_indices.append(next_idx)
            unselected_indices.remove(next_idx)
        
        return selected_indices
    
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
        ids: Optional[List[str]] = None,
        store_type: str = "contexts"
    ) -> List[str]:
        """添加文本到向量存储
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
            ids: 文档ID列表
            store_type: 存储类型 ("templates" 或 "contexts")
            
        Returns:
            List[str]: 添加的文档ID列表
        """
        try:
            # 选择存储
            store = self.template_store if store_type == "templates" else self.context_store
            
            # 进行语义去重
            unique_texts, unique_metadatas = await self._semantic_deduplication(texts, metadatas)
            
            if not unique_texts:
                logger.warning("所有文本都被去重过滤，没有新内容需要添加")
                return []
            
            # 添加到向量存储
            try:
                doc_ids = store.add_texts(
                    texts=unique_texts,
                    metadatas=unique_metadatas,
                    ids=ids[:len(unique_texts)] if ids else None
                )
                
                # 保存存储
                store_path = Path(settings.VECTOR_DB_PATH) / store_type
                store.save_local(str(store_path))
                
                logger.info(f"成功添加{len(doc_ids)}条文本到{store_type}向量存储")
                return doc_ids
                
            except Exception as e:
                logger.error(f"添加文本到向量存储失败: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"处理文本添加失败: {str(e)}")
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
        try:
            embeddings = []
            for text in texts:
                try:
                    embedding = await self._get_embedding_with_cache(text)
                    embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"获取嵌入向量失败: {str(e)}")
                    # 使用零向量作为fallback
                    dim = settings.VECTOR_STORE_CONFIG.get("embedding_dimensions", 64)
                    embeddings.append([0.0] * dim)
            return embeddings
        except Exception as e:
            logger.error(f"批量获取嵌入向量失败: {str(e)}")
            # 返回零向量列表作为fallback
            dim = settings.VECTOR_STORE_CONFIG.get("embedding_dimensions", 64)
            return [[0.0] * dim for _ in texts]

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的语义相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            float: 相似度得分
        """
        try:
            # 获取文本的嵌入向量
            embedding1 = self.embeddings.embed_query(text1)
            embedding2 = self.embeddings.embed_query(text2)
            
            # 计算余弦相似度
            return self.calculate_vector_similarity(embedding1, embedding2)
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
            # 确保存储目录存在
            Path(settings.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)
            
            # 初始化基础存储
            base_texts = ["初始化文档"]
            base_metadata = [{"type": "base", "timestamp": datetime.now().isoformat()}]
            
            # 定义所有需要的存储
            store_configs = {
                "context_store": "contexts",
                "template_store": "templates",
                "prompts_store": "prompts",
                "designs_store": "designs"
            }
            
            # 初始化或加载每个存储
            for attr_name, store_name in store_configs.items():
                store_path = Path(settings.VECTOR_DB_PATH) / store_name
                try:
                    if store_path.exists():
                        store = FAISS.load_local(
                            str(store_path),
                            self.embeddings,
                            allow_dangerous_deserialization=True
                        )
                        logger.info(f"成功加载现有的{store_name}向量存储")
                    else:
                        store = FAISS.from_texts(
                            texts=base_texts,
                            embedding=self.embeddings,
                            metadatas=base_metadata
                        )
                        # 立即保存新创建的存储
                        store.save_local(str(store_path))
                        logger.info(f"创建并保存新的{store_name}向量存储")
                    
                    # 设置类属性
                    setattr(self, attr_name, store)
                    
                except Exception as e:
                    logger.error(f"初始化{store_name}存储失败: {str(e)}")
                    raise
            
            # 验证所有存储是否正确初始化
            for attr_name in store_configs.keys():
                if not hasattr(self, attr_name):
                    raise ValueError(f"存储{attr_name}初始化失败")
                if getattr(self, attr_name) is None:
                    raise ValueError(f"存储{attr_name}为空")
            
            logger.info("所有向量存储初始化完成")
            
        except Exception as e:
            logger.error(f"初始化向量存储失败: {str(e)}", exc_info=True)
            raise
        
    def _init_mock_data(self):
        """初始化mock数据"""
        logger.info("初始化mock数据...")
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]
        mock_embeddings.embed_query.return_value = [0.1] * 1536

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
            tuple[List[str], Optional[List[Dict[str, Any]]]]: 去重后的文本和元数据
        """
        try:
            if not texts:
                return texts, metadatas
                
            # 使用较低的阈值进行快速去重
            threshold = settings.VECTOR_STORE_CONFIG.get("deduplication_threshold", 0.95)
            min_length = settings.VECTOR_STORE_CONFIG.get("min_dedup_length", 10)
            
            # 对短文本跳过去重
            if all(len(text) < min_length for text in texts):
                return texts, metadatas
            
            try:
                # 使用批量嵌入处理
                embeddings = await self._get_embeddings_batch_with_cache(texts)
                if not embeddings or len(embeddings) != len(texts):
                    logger.error("获取嵌入向量失败或数量不匹配")
                    return texts, metadatas
                
                # 使用numpy进行向量化计算
                embeddings_array = np.array(embeddings, dtype=np.float32)
                norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                
                # 避免除零错误
                mask = norms > 1e-10
                norms[~mask] = 1e-10
                
                # 计算余弦相似度矩阵
                normalized_embeddings = embeddings_array / norms
                similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
                
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
                
            except np.linalg.LinAlgError as e:
                logger.error(f"向量计算错误: {str(e)}")
                return texts, metadatas
                
        except Exception as e:
            logger.error(f"语义去重失败: {str(e)}")
            return texts, metadatas
        
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
    def calculate_vector_similarity(vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            float: 相似度得分
        """
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
        except Exception as e:
            logger.error(f"计算向量相似度失败: {str(e)}")
            return 0.0

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

    async def validate_insert(self, content: str, store_type: str = "contexts") -> bool:
        """验证内容是否可以插入（避免重复）
        
        Args:
            content: 要验证的内容
            store_type: 存储类型 ("templates" 或 "contexts")
            
        Returns:
            bool: 是否成功插入
        """
        try:
            # 选择存储
            store = self.template_store if store_type == "templates" else self.context_store
            
            # 获取内容的嵌入向量
            content_embedding = await self._get_embedding_with_cache(content)
            
            # 搜索最相似的结果
            results = store.similarity_search_with_score(content, k=1)
            
            if results:
                # 获取最相似结果的向量
                result_embedding = await self._get_embedding_with_cache(results[0][0].page_content)
                
                # 计算余弦相似度
                similarity = np.dot(content_embedding, result_embedding) / (
                    np.linalg.norm(content_embedding) * np.linalg.norm(result_embedding))
                
                # 使用较高的阈值确保精确匹配
                return similarity < settings.VECTOR_STORE_CONFIG["deduplication_threshold"]
            
            return True
            
        except Exception as e:
            logger.error(f"验证插入失败: {str(e)}")
            return False
        
    def _load_or_create_store(self, store_type: str) -> FAISS:
        """加载或创建向量存储
        
        Args:
            store_type: 存储类型 (contexts/templates)
            
        Returns:
            FAISS: 向量存储实例
        """
        store_path = Path(settings.VECTOR_DB_PATH) / store_type
        
        try:
            # 如果存储目录存在，尝试加载
            if store_path.exists():
                store = FAISS.load_local(
                    str(store_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"成功加载现有的{store_type}向量存储")
            else:
                # 创建新的存储
                store = FAISS.from_texts(
                    texts=["初始化文档"],
                    embedding=self.embeddings,
                    metadatas=[{"type": store_type}]
                )
                # 保存存储
                store.save_local(str(store_path))
                logger.info(f"创建并保存新的{store_type}向量存储")
                
            return store
            
        except Exception as e:
            logger.error(f"加载或创建{store_type}向量存储失败: {str(e)}")
            raise
        
    async def save(self):
        """保存向量存储"""
        try:
            # 保存上下文存储
            if hasattr(self, 'context_store'):
                context_path = Path(settings.VECTOR_DB_PATH) / "contexts"
                self.context_store.save_local(str(context_path))
                logger.info("成功保存上下文向量存储")
            
            # 保存模板存储
            if hasattr(self, 'template_store'):
                template_path = Path(settings.VECTOR_DB_PATH) / "templates"
                self.template_store.save_local(str(template_path))
                logger.info("成功保存模板向量存储")
            
            # 保存文本嵌入缓存
            self._save_embedding_cache()
            
            # 保存图片嵌入缓存
            self._save_image_embedding_cache()
            
            logger.info("向量存储保存完成")
            
        except Exception as e:
            logger.error(f"保存向量存储失败: {str(e)}")
            raise
        
    async def _get_image_embedding(self, image_data: bytes) -> List[float]:
        """获取图片的嵌入向量
        
        Args:
            image_data: 图片数据
            
        Returns:
            List[float]: 图片的嵌入向量
        """
        try:
            # 计算图片数据的hash作为缓存键
            import hashlib
            cache_key = hashlib.md5(image_data).hexdigest()
            
            # 检查缓存
            if cache_key in self._image_embedding_cache:
                logger.info("使用缓存的图片嵌入向量")
                return self._image_embedding_cache[cache_key]
            
            # 图片预处理
            try:
                from PIL import Image
                import io
                
                # 打开并验证图片
                img = Image.open(io.BytesIO(image_data))
                img_format = img.format or 'PNG'
                
                # 转换为RGB模式
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    logger.info(f"图片已转换为RGB模式: {img.mode}")
                
                # 调整图片大小（如果太大）
                max_size = (1024, 1024)
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    logger.info(f"图片已调整大小: {img.size}")
                
                # 保存为bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img_format, quality=95)
                img_byte_arr = img_byte_arr.getvalue()
                
                # 转换为base64
                import base64
                image_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
                
                logger.info(f"图片预处理成功: 格式={img_format}, 大小={len(img_byte_arr)}, 尺寸={img.size}")
                
            except Exception as e:
                logger.error(f"图片预处理失败: {str(e)}", exc_info=True)
                raise ValueError(f"图片预处理失败: {str(e)}")
            
            # 使用视觉模型分析图片
            try:
                if not self._vision_model:
                    self._init_vision_model()
                
                # 构建API请求
                messages = [
                    {
                        "role": "system",
                        "content": "分析这个UI设计图的布局结构、视觉元素和交互设计特征。"
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "请提供以下分析：\n1. 布局结构\n2. UI组件\n3. 视觉风格\n4. 交互模式"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{img_format.lower()};base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ]
                
                # 调用API
                max_retries = 3
                retry_delay = 1
                
                for attempt in range(max_retries):
                    try:
                        response = self._vision_model.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=1000,
                            temperature=0.3
                        )
                        
                        if not response or not response.choices:
                            raise ValueError("API返回结果无效")
                        
                        break
                        
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"API调用尝试 {attempt + 1} 失败: {str(e)}")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # 指数退避
                        else:
                            raise
                
                # 获取分析文本
                analysis_text = response.choices[0].message.content
                if not analysis_text or len(analysis_text.strip()) < 50:
                    raise ValueError("分析结果过短或为空")
                
                logger.info(f"获取分析结果: 长度={len(analysis_text)}")
                
                # 生成嵌入向量
                embedding = await self._get_embedding_with_cache(analysis_text)
                
                # 验证向量
                if not embedding or len(embedding) != 1536:
                    raise ValueError(f"嵌入向量维度错误: {len(embedding) if embedding else 0}")
                
                # 缓存结果
                self._image_embedding_cache[cache_key] = embedding
                await self._async_save_image_cache()
                
                logger.info("成功生成并缓存图片嵌入向量")
                return embedding
                
            except Exception as e:
                logger.error(f"图片分析失败: {str(e)}", exc_info=True)
                raise
                
        except Exception as e:
            logger.error(f"获取图片嵌入向量失败: {str(e)}", exc_info=True)
            # 返回零向量作为降级方案
            return [0.0] * 1536
    
    def _load_image_embedding_cache(self):
        """加载图片嵌入缓存"""
        try:
            if self._image_embedding_cache_file.exists():
                import gzip
                import json
                
                with gzip.open(self._image_embedding_cache_file, 'rt', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    # 转换缓存数据中的列表为numpy数组
                    self._image_embedding_cache = {
                        k: np.array(v, dtype=np.float32) 
                        for k, v in cache_data.items()
                    }
                logger.info(f"已加载图片嵌入缓存: {len(self._image_embedding_cache)}条记录")
            else:
                logger.info("图片嵌入缓存文件不存在，将创建新缓存")
                self._image_embedding_cache = {}
                
        except Exception as e:
            logger.error(f"加载图片嵌入缓存失败: {str(e)}", exc_info=True)
            self._image_embedding_cache = {}

    def _save_image_embedding_cache(self):
        """保存图片嵌入缓存"""
        try:
            import gzip
            import json
            
            # 确保缓存目录存在
            self._image_embedding_cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换numpy数组为列表
            cache_data = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self._image_embedding_cache.items()
            }
            
            # 使用gzip压缩保存
            with gzip.open(self._image_embedding_cache_file, 'wt', encoding='utf-8') as f:
                json.dump(cache_data, f)
                
            logger.info(f"已保存图片嵌入缓存: {len(self._image_embedding_cache)}条记录")
            
        except Exception as e:
            logger.error(f"保存图片嵌入缓存失败: {str(e)}", exc_info=True)

    async def _async_save_image_cache(self):
        """异步保存图片嵌入缓存"""
        try:
            # 使用线程池执行同步的保存操作
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_image_embedding_cache)
            
        except Exception as e:
            logger.error(f"异步保存图片嵌入缓存失败: {str(e)}", exc_info=True)
        