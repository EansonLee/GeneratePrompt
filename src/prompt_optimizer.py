"""提示词优化器模块"""

import logging
import json
import time
import asyncio
import statistics
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from config.config import settings
from src.utils.vector_store import VectorStore
from src.utils.cache_manager import CacheManager
from src.templates.prompt_templates import (
    SYSTEM_PROMPT,
    OPTIMIZATION_TEMPLATE,
    RAG_TEMPLATE,
    EVALUATION_TEMPLATE
)

logger = logging.getLogger(__name__)

class PromptOptimizer:
    """提示词优化器"""
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """初始化优化器
        
        Args:
            vector_store: 向量存储实例，如果不传入则使用全局实例
        """
        try:
            # 使用传入的向量存储实例或获取全局实例
            self.vector_store = vector_store or VectorStore()
            
            # 初始化LLM
            self.llm = ChatOpenAI(
                temperature=settings.PROMPT_OPTIMIZATION_CONFIG["temperature"],
                model=settings.OPENAI_MODEL,
                max_tokens=settings.PROMPT_OPTIMIZATION_CONFIG["max_tokens"]
            )
            
            # 初始化提示模板
            self.optimization_prompt = ChatPromptTemplate.from_template(OPTIMIZATION_TEMPLATE)
            self.rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
            self.evaluation_prompt = ChatPromptTemplate.from_template(EVALUATION_TEMPLATE)
            
            # 初始化缓存管理器
            self.cache = CacheManager(
                max_size=settings.PROMPT_OPTIMIZATION_CONFIG["cache_size"],
                ttl=settings.PROMPT_OPTIMIZATION_CONFIG["cache_ttl"]
            )
            
            # 初始化线程池
            self.thread_pool = ThreadPoolExecutor(
                max_workers=settings.PROMPT_OPTIMIZATION_CONFIG["max_workers"]
            )
            
            logger.info("优化器初始化成功")
            
        except Exception as e:
            logger.error(f"初始化优化器失败: {str(e)}")
            raise
            
    async def optimize_batch(
        self,
        prompts: List[str],
        context: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """批量优化提示词
        
        Args:
            prompts: 提示词列表
            context: 可选的上下文信息
            batch_size: 批处理大小
            
        Returns:
            List[Dict[str, Any]]: 优化结果列表
        """
        if not batch_size:
            batch_size = settings.PROMPT_OPTIMIZATION_CONFIG["batch_size"]
            
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = await asyncio.gather(*[
                self.optimize(prompt, context)
                for prompt in batch
            ])
            results.extend(batch_results)
            
            # 动态调整批大小
            if i > 0:
                avg_time = statistics.mean(
                    r.get("optimization_time", 0)
                    for r in batch_results
                )
                batch_size = self._adjust_batch_size(avg_time, batch_size)
                
        return results
        
    async def optimize(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """优化提示词
        
        Args:
            prompt: 原始提示词
            context: 上下文信息
            
        Returns:
            优化结果字典
        """
        start_time = time.time()
        
        try:
            # 检查缓存和历史记录
            cache_key = f"optimize_{prompt}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info("使用缓存的优化结果")
                return json.loads(json.dumps(cached_result))
            
            # 获取最近的历史记录
            latest_history = await self._get_latest_history(prompt)
            
            # 并行执行相似提示词搜索和项目上下文获取
            similar_prompts_task = asyncio.create_task(
                self.vector_store.similarity_search(
                    prompt,
                    k=min(3, settings.PROMPT_OPTIMIZATION_CONFIG.get("max_similar_prompts", 5))
                )
            )
            project_context_task = asyncio.create_task(self._get_project_context())
            
            # 等待所有任务完成
            similar_prompts, project_context = await asyncio.gather(
                similar_prompts_task,
                project_context_task
            )
            
            # 转换为可序列化的格式并限制大小
            similar_prompts_json = []
            total_length = 0
            max_length_per_prompt = 500
            
            for doc in similar_prompts:
                if total_length >= 1500:
                    break
                content = self._truncate_text(str(doc.page_content), max_length_per_prompt)
                total_length += len(content)
                similar_prompts_json.append({
                    "content": content,
                    "metadata": dict(doc.metadata)
                })
            
            # 使用 RAG 增强上下文，包含历史对比信息
            rag_result = await self._enhance_with_rag(
                self._truncate_text(prompt, 500),
                similar_prompts_json,
                latest_history
            )
            
            # 优化提示词
            optimization_result = await self._optimize_prompt(
                self._truncate_text(prompt, 500),
                rag_result,
                latest_history
            )
            
            # 并行执行评估
            evaluation = await self._evaluate_prompt(
                self._truncate_text(optimization_result["content"], 1000)
            )
            
            # 计算优化时间
            optimization_time = time.time() - start_time
            
            # 构建最终结果
            result = {
                "original_prompt": str(prompt),
                "optimized_prompt": str(optimization_result["content"]),
                "rag_info": {
                    "enhanced_context": str(rag_result["enhanced_context"]),
                    "similar_prompts": similar_prompts_json[:2],
                    "latest_history": latest_history
                },
                "evaluation": dict(evaluation),
                "optimization_time": float(optimization_time)
            }
            
            # 异步保存结果到缓存
            asyncio.create_task(self._async_cache_result(cache_key, result))
            
            # 异步记录性能指标
            asyncio.create_task(self._log_performance_metrics({
                "operation": "optimize",
                "prompt_length": len(prompt),
                "optimization_time": optimization_time,
                "cache_hit": False,
                "similar_prompts_count": len(similar_prompts_json),
                "scores": evaluation.get("scores", {})
            }))
            
            return json.loads(json.dumps(result))
            
        except Exception as e:
            logger.error(f"优化提示词失败: {str(e)}")
            error_result = {
                "error": str(e),
                "original_prompt": str(prompt),
                "optimized_prompt": str(prompt),
                "evaluation": {
                    "scores": {
                        "clarity": 0.0,
                        "completeness": 0.0,
                        "relevance": 0.0,
                        "consistency": 0.0,
                        "structure": 0.0
                    },
                    "feedback": [f"优化失败: {str(e)}"],
                    "suggestions": ["请检查输入并重试"]
                }
            }
            return json.loads(json.dumps(error_result))

    async def _async_cache_result(self, key: str, value: Any):
        """异步缓存结果
        
        Args:
            key: 缓存键
            value: 要缓存的值
        """
        try:
            self.cache.set(key, value)
        except Exception as e:
            logger.error(f"缓存结果失败: {str(e)}")

    def _truncate_text(self, text: str, max_length: int = 4000) -> str:
        """截断文本到指定长度
        
        Args:
            text: 要截断的文本
            max_length: 最大长度
            
        Returns:
            str: 截断后的文本
        """
        if len(text) <= max_length:
            return text
        return text[:max_length] + "...(已截断)"

    async def _get_latest_history(self, prompt: str) -> Optional[Dict[str, Any]]:
        """获取最近的历史记录
        
        Args:
            prompt: 原始提示词
            
        Returns:
            Optional[Dict[str, Any]]: 最近的历史记录
        """
        try:
            # 使用相似度搜索找到最相似的历史记录
            results = await self.vector_store.similarity_search(
                prompt,
                k=1,
                filter_dict={"type": "prompt_history"},
                search_type="similarity"
            )
            
            if results and len(results) > 0:
                return {
                    "content": results[0].page_content,
                    "metadata": dict(results[0].metadata),
                    "similarity_score": 0.95  # 假设相似度分数
                }
            return None
            
        except Exception as e:
            logger.error(f"获取历史记录失败: {str(e)}")
            return None

    async def _enhance_with_rag(
        self,
        prompt: str,
        similar_prompts: List[Dict[str, Any]],
        latest_history: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """使用RAG增强提示词
        
        Args:
            prompt: 原始提示词
            similar_prompts: 相似的历史提示词
            latest_history: 最近的历史记录
            
        Returns:
            Dict[str, Any]: RAG增强结果
        """
        try:
            # 限制相似提示词的数量和长度
            truncated_prompts = []
            total_length = 0
            max_prompt_length = 500
            
            # 如果有历史记录，优先使用
            if latest_history:
                truncated_prompts.append({
                    "content": self._truncate_text(latest_history["content"], max_prompt_length),
                    "metadata": latest_history["metadata"],
                    "is_history": True
                })
                total_length += len(latest_history["content"])
            
            # 添加其他相似提示词
            for prompt_dict in similar_prompts[:2]:
                if total_length >= 1500:
                    break
                content = prompt_dict.get("content", "")
                if content:
                    truncated_content = self._truncate_text(content, max_prompt_length)
                    truncated_prompts.append({
                        "content": truncated_content,
                        "metadata": prompt_dict.get("metadata", {}),
                        "is_history": False
                    })
                    total_length += len(truncated_content)
            
            # 使用RAG模板生成增强上下文
            rag_result = await self.llm.apredict_messages([{
                "role": "system",
                "content": self._truncate_text(SYSTEM_PROMPT, 500)
            }, {
                "role": "user",
                "content": self.rag_prompt.format(
                    original_prompt=self._truncate_text(prompt, 500),
                    similar_prompts=json.dumps(truncated_prompts, ensure_ascii=False),
                    latest_history=json.dumps(latest_history, ensure_ascii=False) if latest_history else "null"
                )
            }])
            
            return {
                "enhanced_context": rag_result.content,
                "similar_prompts": truncated_prompts,
                "latest_history": latest_history,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"RAG增强失败: {str(e)}")
            raise

    async def _optimize_prompt(
        self,
        prompt: str,
        rag_result: Dict[str, Any],
        latest_history: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """优化提示词
        
        Args:
            prompt: 原始提示词
            rag_result: RAG增强结果
            latest_history: 最近的历史记录
            
        Returns:
            Dict[str, Any]: 优化结果
        """
        try:
            # 获取项目上下文
            project_context = await self._get_project_context()
            
            # 限制各部分内容长度
            truncated_context = {
                "tech_stack": self._truncate_text(json.dumps(project_context.get("tech_stack", {}), ensure_ascii=False), 500),
                "file_structure": self._truncate_text(json.dumps(project_context.get("file_structure", {}), ensure_ascii=False), 500),
                "output_paths": self._truncate_text(json.dumps(project_context.get("output_paths", {}), ensure_ascii=False), 500)
            }
            
            # 使用优化模板生成优化后的提示词
            optimization_result = await self.llm.apredict_messages([{
                "role": "system",
                "content": self._truncate_text(SYSTEM_PROMPT, 500)
            }, {
                "role": "user",
                "content": self.optimization_prompt.format(
                    original_prompt=self._truncate_text(prompt, 500),
                    tech_stack=truncated_context["tech_stack"],
                    file_structure=truncated_context["file_structure"],
                    output_paths=truncated_context["output_paths"],
                    similar_prompts=self._truncate_text(json.dumps(rag_result["similar_prompts"], ensure_ascii=False), 1000),
                    rag_context=self._truncate_text(rag_result["enhanced_context"], 1000),
                    latest_history=json.dumps(latest_history, ensure_ascii=False) if latest_history else "null"
                )
            }])
            
            return {
                "content": optimization_result.content,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"优化提示词失败: {str(e)}")
            return {
                "content": prompt,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _evaluate_prompt(self, prompt: str) -> Dict[str, Any]:
        """评估优化后的提示词
        
        Args:
            prompt: 优化后的提示词
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            evaluation_result = await self.llm.apredict_messages([{
                "role": "system",
                "content": SYSTEM_PROMPT
            }, {
                "role": "user",
                "content": self.evaluation_prompt.format(prompt=prompt)
            }])
            
            # 解析评估结果
            try:
                # 尝试从内容中提取JSON部分
                content = evaluation_result.content
                # 查找第一个 { 和最后一个 } 的位置
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = content[start:end]
                    result = json.loads(json_str)
                else:
                    raise json.JSONDecodeError("No JSON found in content", content, 0)
                    
            except json.JSONDecodeError:
                logger.warning("评估结果解析失败，使用默认评估结果")
                logger.debug(f"原始评估结果: {evaluation_result.content}")
                result = {
                    "scores": {
                        "clarity": 0.7,
                        "completeness": 0.7,
                        "relevance": 0.7,
                        "consistency": 0.7,
                        "structure": 0.7
                    },
                    "feedback": ["提示词评估结果解析失败"],
                    "suggestions": ["建议重新评估"],
                    "directory_structure_analysis": {
                        "completeness": "无法评估",
                        "organization": "无法评估",
                        "naming": "无法评估",
                        "modularity": "无法评估",
                        "improvements": ["建议重新评估目录结构"]
                    }
                }
            
            return {
                "feedback": result.get("feedback", ["评估反馈解析失败"]),
                "improvement_suggestions": result.get("suggestions", ["建议重新评估"]),
                "scores": {
                    "clarity": result.get("scores", {}).get("clarity", 0.7),
                    "completeness": result.get("scores", {}).get("completeness", 0.7),
                    "relevance": result.get("scores", {}).get("relevance", 0.7),
                    "consistency": result.get("scores", {}).get("consistency", 0.7),
                    "structure": result.get("scores", {}).get("structure", 0.7)
                },
                "directory_structure_analysis": result.get("directory_structure_analysis", {
                    "completeness": "无法评估",
                    "organization": "无法评估",
                    "naming": "无法评估",
                    "modularity": "无法评估",
                    "improvements": ["建议重新评估目录结构"]
                })
            }
            
        except Exception as e:
            logger.error(f"评估提示词失败: {str(e)}")
            return {
                "feedback": [f"评估失败: {str(e)}"],
                "improvement_suggestions": ["建议重新评估"],
                "scores": {
                    "clarity": 0.7,
                    "completeness": 0.7,
                    "relevance": 0.7,
                    "consistency": 0.7,
                    "structure": 0.7
                },
                "directory_structure_analysis": {
                    "completeness": "评估失败",
                    "organization": "评估失败",
                    "naming": "评估失败",
                    "modularity": "评估失败",
                    "improvements": ["评估过程出错，请重试"]
                }
            }
            
    async def _save_optimization_result(
        self,
        original_prompt: str,
        optimized_prompt: str,
        evaluation: Dict[str, Any],
        context: Dict[str, Any],
        optimization_time: float
    ) -> None:
        """保存优化结果
        
        Args:
            original_prompt: 原始提示词
            optimized_prompt: 优化后的提示词
            evaluation: 评估结果
            context: 上下文信息
            optimization_time: 优化耗时
        """
        try:
            # 准备元数据
            metadata = {
                "original_prompt": original_prompt,
                "evaluation": evaluation,
                "context": context,
                "optimization_time": optimization_time,
                "timestamp": datetime.now().isoformat(),
                "model_version": settings.OPENAI_MODEL
            }
            
            # 保存到向量存储
            await self.vector_store.add_texts(
                texts=[optimized_prompt],
                metadatas=[metadata]
            )
            
        except Exception as e:
            logger.error(f"保存优化结果失败: {str(e)}")
            raise
            
    @staticmethod
    def _adjust_batch_size(avg_time: float, current_size: int) -> int:
        """动态调整批处理大小
        
        Args:
            avg_time: 平均处理时间
            current_size: 当前批大小
            
        Returns:
            int: 调整后的批大小
        """
        target_time = 5.0  # 目标平均处理时间（秒）
        min_size = 1
        max_size = 20
        
        if avg_time > target_time * 1.2:  # 处理时间过长，减小批大小
            new_size = max(min_size, int(current_size * 0.8))
        elif avg_time < target_time * 0.8:  # 处理时间较短，增加批大小
            new_size = min(max_size, int(current_size * 1.2))
        else:
            new_size = current_size
            
        return new_size
        
    async def _log_performance_metrics(self, metrics: Dict[str, Any]):
        """记录性能指标
        
        Args:
            metrics: 性能指标数据
        """
        try:
            # 这里可以实现具体的指标记录逻辑
            # 例如：发送到监控系统、写入日志等
            if metrics.get("optimization_time", 0) > settings.PROMPT_OPTIMIZATION_CONFIG["slow_query_threshold"]:
                logger.warning(f"检测到慢查询: {json.dumps(metrics, ensure_ascii=False)}")
                
            logger.info(f"性能指标: {json.dumps(metrics, ensure_ascii=False)}")
            
        except Exception as e:
            logger.error(f"记录性能指标失败: {str(e)}")
            # 不抛出异常，避免影响主流程

    async def _get_project_context(self) -> Dict[str, Any]:
        """获取项目上下文信息"""
        try:
            results = await self.vector_store.similarity_search(
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
            return {}

    async def search_prompt_history(
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
        results = await self.vector_store.similarity_search(
            query,
            k=k,
            filter_dict={"type": "prompt_history"},
            search_type=search_type
        )
        return [doc.page_content for doc in results] 