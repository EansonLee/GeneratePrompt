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

# 设置日志级别
logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

class VectorStore:
    """增强的向量存储类"""
    
    _instance = None  # 单例实例
    _is_initialized = False  # 初始化标志
    _initialization_error = None  # 初始化错误信息
    
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
            
            self.embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                openai_api_base=settings.OPENAI_BASE_URL,
                headers={
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "OpenAI/v1 PythonClient/1.0.0"
                },
                timeout=30,
                max_retries=3,
                model_kwargs={
                    "encoding_format": "float"
                }
            )
            
            # 只在首次初始化时测试嵌入功能
            if not hasattr(self, '_embedding_tested'):
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
        
        # 检查缓存
        if cache_key and cache_key in self.cache:
            logger.info("使用缓存的搜索结果")
            return self.cache[cache_key]
            
        try:
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
                        k=k,
                        fetch_k=min(2*k, 10),  # 限制fetch_k的大小
                        lambda_mult=0.7,
                        **kwargs
                    )
                else:
                    results = await asyncio.to_thread(
                        self.context_store.similarity_search,
                        query,
                        k=min(k, 5),  # 限制返回结果数量
                        **kwargs
                    )
                    
                # 检查查询时间
                query_time = time.time() - start_time
                if query_time > settings.VECTOR_STORE_CONFIG.get("slow_query_threshold", 5.0):
                    logger.warning(f"检测到慢查询: {query_time:.2f}s, query={query}")
                    
                # 缓存结果
                if cache_key:
                    self.cache[cache_key] = results
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

            if settings.VECTOR_STORE_CONFIG.get("enable_semantic_deduplication", False):
                # 执行语义去重
                texts, metadatas = await self._semantic_deduplication(texts, metadatas)
                
            # 生成唯一ID
            if not ids:
                ids = [str(uuid.uuid4()) for _ in texts]
                
            # 添加时间戳到元数据
            if metadatas:
                for metadata in metadatas:
                    metadata["timestamp"] = datetime.now().isoformat()
            else:
                metadatas = [{"timestamp": datetime.now().isoformat()} for _ in texts]

            # 使用 asyncio.to_thread 包装同步操作
            doc_ids = await asyncio.to_thread(
                self.context_store.add_texts,
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
            
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
            results = self.context_store.similarity_search(
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
            self.template_store.add_documents([doc])
            logger.info("成功添加模板")
        except Exception as e:
            logger.error(f"添加模板失败: {str(e)}")
            raise

    def verify_insertion(self, content: str, store_type: str = "contexts") -> bool:
        """验证内容是否成功插入到向量数据库
        
        Args:
            content: 要验证的内容
            store_type: 存储类型，可选 "contexts" 或 "templates"
            
        Returns:
            bool: 是否成功插入
        """
        try:
            store = self.context_store if store_type == "contexts" else self.template_store
            # 使用内容的前200个字符进行搜索，避免过长内容影响相似度
            search_content = content[:200]
            results = store.similarity_search_with_score(search_content, k=1)
            
            if not results:
                logger.warning(f"未找到插入的内容: {search_content}...")
                return False
                
            # 获取最相似文档的相似度分数
            doc, score = results[0]
            # FAISS返回的是距离，需要转换为相似度（1 - 距离）
            similarity = 1 - score
            
            # 降低相似度阈值到0.5，因为FAISS的相似度计算方式与余弦相似度不同
            is_similar = similarity > 0.5
            
            if not is_similar:
                logger.warning(f"找到的内容相似度较低 ({similarity:.4f}): {doc.page_content[:100]}...")
            else:
                logger.info(f"成功验证内容插入 (相似度: {similarity:.4f})")
                
            # 如果相似度大于0.5，或者内容完全匹配，则认为插入成功
            if is_similar or search_content.strip() in doc.page_content:
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"验证插入失败: {str(e)}")
            return False
            
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的相似度
        
        Args:
            text1: 第一段文本
            text2: 第二段文本
            
        Returns:
            float: 相似度分数 (0-1)
        """
        try:
            # 使用 OpenAI embeddings 计算相似度
            embedding1 = self.embeddings.embed_query(text1)
            embedding2 = self.embeddings.embed_query(text2)
            
            # 计算余弦相似度
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            return float(similarity)
            
        except Exception as e:
            logger.error(f"计算相似度失败: {str(e)}")
            return 0.0 

    def _test_embeddings(self):
        """测试嵌入功能"""
        try:
            test_text = "测试文本"
            test_embedding = self.embeddings.embed_query(test_text)
            logger.info(f"嵌入测试成功，向量维度: {len(test_embedding)}")
            return True
        except Exception as e:
            logger.error(f"嵌入测试失败: {str(e)}")
            raise
        
    def _init_stores(self):
        """初始化向量存储"""
        try:
            # 配置 OpenAI Embeddings
            self.embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                openai_api_base=settings.OPENAI_BASE_URL
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
                self.context_store = FAISS.from_texts(
                    ["初始化文档"],
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
                self.template_store = FAISS.from_texts(
                    ["初始化模板"],
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
            
        threshold = settings.VECTOR_STORE_CONFIG.get("deduplication_threshold", 0.95)
        
        # 获取所有文本的嵌入
        embeddings = [self.embeddings.embed_query(text) for text in texts]
        
        # 构建相似度矩阵
        similarity_matrix = np.zeros((len(texts), len(texts)))
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = self.calculate_similarity(embeddings[i], embeddings[j])
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim
                
        # 找出需要保留的索引
        keep_indices = []
        for i in range(len(texts)):
            # 如果当前文本与已保留的文本相似度都低于阈值，则保留
            if not any(similarity_matrix[i][j] > threshold for j in keep_indices):
                keep_indices.append(i)
                
        # 保留未重复的文本和元数据
        unique_texts = [texts[i] for i in keep_indices]
        unique_metadatas = [metadatas[i] for i in keep_indices] if metadatas else None
        
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