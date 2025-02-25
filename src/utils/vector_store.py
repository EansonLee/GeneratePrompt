from typing import Dict, Any, List, Optional
import logging
import os
from datetime import datetime
from unittest.mock import Mock, MagicMock
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from config.config import (
    OPENAI_API_KEY, 
    OPENAI_BASE_URL, 
    EMBEDDING_MODEL,
    VECTOR_STORE_CONFIG,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    LOG_LEVEL,
    DEBUG
)
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import json
from pathlib import Path

# 设置日志级别
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

class VectorStore:
    """向量数据库管理类"""
    
    _instance = None  # 单例实例
    _is_initialized = False  # 初始化标志
    
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
        # 如果已经初始化过，直接返回
        if self._is_initialized:
            return
            
        self.is_testing = os.getenv("TESTING", "False").lower() == "true"
        self.is_ready = False
        self.initialization_error = None
        
        # 打印配置信息
        logger.info("==================== 当前配置信息 ====================")
        logger.info(f"配置文件中的 API Key: {OPENAI_API_KEY}")
        logger.info(f"配置文件中的 Base URL: {OPENAI_BASE_URL}")
        logger.info(f"配置文件中的 Embedding Model: {EMBEDDING_MODEL}")
        logger.info(f"调试模式: {DEBUG}")
        logger.info("====================================================")
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=VECTOR_STORE_CONFIG["SEPARATORS"]
        )
        logger.info("文本分割器初始化完成")
        
        # 强制使用真实API
        try:
            self._init_real_api()
            self.is_ready = True
        except Exception as e:
            logger.error(f"真实API初始化失败: {str(e)}")
            self.initialization_error = str(e)
            if DEBUG:
                logger.debug("详细错误信息:", exc_info=True)
            if use_mock or self.is_testing:
                logger.warning("回退到mock模式")
                self._init_mock_data()
                self.is_ready = True
            else:
                raise
        
        self.prompt_history = []
        self._is_initialized = True
    
    def is_initialized(self) -> bool:
        """检查向量数据库是否已初始化完成
        
        Returns:
            bool: 是否初始化完成
        """
        return self.is_ready

    def get_initialization_error(self) -> str:
        """获取初始化错误信息
        
        Returns:
            str: 错误信息
        """
        return self.initialization_error

    async def wait_until_ready(self, timeout: int = 30) -> bool:
        """等待向量数据库就绪
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            bool: 是否就绪
        """
        import asyncio
        start_time = asyncio.get_event_loop().time()
        while not self.is_ready:
            if asyncio.get_event_loop().time() - start_time > timeout:
                return False
            await asyncio.sleep(1)
        return True

    def _init_real_api(self):
        """初始化真实API"""
        try:
            logger.info("初始化向量数据库...")
            logger.info(f"使用的 API Base URL: {OPENAI_BASE_URL}")
            logger.info(f"使用的 Embedding Model: text-embedding-ada-002")
            
            # 验证API配置
            if not OPENAI_API_KEY:
                raise ValueError("未设置 OPENAI_API_KEY")
                
            if not OPENAI_BASE_URL:
                raise ValueError("未设置 OPENAI_BASE_URL")
            
            self.embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=OPENAI_BASE_URL,
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
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
            if DEBUG:
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
        
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> List[str]:
        """添加文本到向量数据库"""
        try:
            logger.info(f"开始添加{len(texts)}条文本到向量数据库")
            logger.info("文本内容预览:")
            for i, (text, metadata) in enumerate(zip(texts, metadatas or [{}] * len(texts))):
                logger.info(f"文本 {i+1}:")
                logger.info(f"- 前100字符: {text[:100]}...")
                logger.info(f"- 元数据: {metadata}")
                logger.info(f"- 总字符数: {len(text)}")
            
            documents = self.text_splitter.create_documents(texts, metadatas=metadatas)
            logger.info(f"文本分块完成，共{len(documents)}个块")
            logger.info(f"平均块大小: {sum(len(doc.page_content) for doc in documents) / len(documents):.2f}字符")
            
            self.contexts_store.add_documents(documents)
            logger.info("文档已添加到向量数据库")
            
            # 验证插入
            verification_results = [self.verify_insertion(text) for text in texts]
            success_rate = sum(verification_results) / len(verification_results) * 100
            logger.info(f"文档验证完成，成功率: {success_rate:.2f}%")
            
            if not all(verification_results):
                logger.warning("部分文本可能未成功插入，请检查日志")
            
            return ["doc_" + str(i) for i in range(len(texts))]
            
        except Exception as e:
            logger.error(f"添加文本到向量数据库失败: {str(e)}")
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

    def verify_insertion(self, content: str, store_type: str = "contexts") -> bool:
        """验证内容是否成功插入到向量数据库
        
        Args:
            content: 要验证的内容
            store_type: 存储类型，可选 "contexts" 或 "templates"
            
        Returns:
            bool: 是否成功插入
        """
        try:
            store = self.contexts_store if store_type == "contexts" else self.templates_store
            # 使用内容的前200个字符进行搜索，避免过长内容影响相似度
            search_content = content[:200]
            results = store.similarity_search(search_content, k=1)
            
            if not results:
                logger.warning(f"未找到插入的内容: {search_content}...")
                return False
                
            similarity_score = self.calculate_similarity(search_content, results[0].page_content[:200])
            # 降低相似度阈值到0.7
            is_similar = similarity_score > 0.7
            
            if not is_similar:
                logger.warning(f"找到的内容相似度较低 ({similarity_score}): {results[0].page_content[:100]}...")
            else:
                logger.info(f"成功验证内容插入 (相似度: {similarity_score})")
                
            return is_similar
            
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
        """初始化存储"""
        logger.info("初始化向量存储...")
        
        # 确保向量存储目录存在
        vector_store_path = Path(VECTOR_STORE_CONFIG["persist_directory"])
        vector_store_path.mkdir(parents=True, exist_ok=True)
        
        contexts_path = vector_store_path / "contexts"
        templates_path = vector_store_path / "templates"
        
        try:
            # 尝试加载已存在的向量存储
            if contexts_path.exists():
                self.contexts_store = FAISS.load_local(
                    str(contexts_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("成功加载已存在的contexts向量存储")
            else:
                # 使用最小的初始化数据
                self.contexts_store = FAISS.from_texts(
                    [""],  # 使用空字符串作为初始数据
                    self.embeddings,
                    metadatas=[{"type": "context"}]
                )
                self.contexts_store.save_local(str(contexts_path))
                logger.info("创建并保存新的contexts向量存储")
                
            if templates_path.exists():
                self.templates_store = FAISS.load_local(
                    str(templates_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("成功加载已存在的templates向量存储")
            else:
                # 使用最小的初始化数据
                self.templates_store = FAISS.from_texts(
                    [""],  # 使用空字符串作为初始数据
                    self.embeddings,
                    metadatas=[{"type": "template"}]
                )
                self.templates_store.save_local(str(templates_path))
                logger.info("创建并保存新的templates向量存储")
                
        except Exception as e:
            logger.error(f"初始化向量存储失败: {str(e)}")
            raise
            
        logger.info("向量数据库初始化完成")
        
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
        self.contexts_store = mock_contexts
        
        mock_templates = MagicMock()
        mock_templates.similarity_search.return_value = [
            Document(page_content="测试模板1"),
            Document(page_content="测试模板2")
        ]
        self.templates_store = mock_templates
        logger.info("Mock数据初始化完成") 