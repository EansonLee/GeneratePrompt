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
            self._init_cache_managers()
            
            # 初始化线程池
            self.thread_pool = ThreadPoolExecutor(
                max_workers=settings.PROMPT_OPTIMIZATION_CONFIG["max_workers"]
            )
            
            logger.info("优化器初始化成功")
            
        except Exception as e:
            logger.error(f"优化器初始化失败: {str(e)}")
            raise
            
    def _init_cache_managers(self):
        """初始化缓存管理器"""
        try:
            # 优化结果缓存
            self.optimization_cache = CacheManager(
                max_size=settings.PROMPT_OPTIMIZATION_CONFIG.get("cache_size", 1000),
                ttl=settings.PROMPT_OPTIMIZATION_CONFIG.get("cache_ttl", 86400)  # 1天
            )
            
            # 相似提示词缓存
            self.similar_prompts_cache = CacheManager(
                max_size=settings.PROMPT_OPTIMIZATION_CONFIG.get("similar_cache_size", 500),
                ttl=settings.PROMPT_OPTIMIZATION_CONFIG.get("similar_cache_ttl", 3600)  # 1小时
            )
            
            # RAG结果缓存
            self.rag_cache = CacheManager(
                max_size=settings.PROMPT_OPTIMIZATION_CONFIG.get("rag_cache_size", 500),
                ttl=settings.PROMPT_OPTIMIZATION_CONFIG.get("rag_cache_ttl", 3600)  # 1小时
            )
            
        except Exception as e:
            logger.error(f"初始化缓存管理器失败: {str(e)}")
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
        
    async def optimize(self, prompt: str, context_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """优化提示词
        
        Args:
            prompt: 原始提示词
            context_files: 上下文文件列表
            
        Returns:
            Dict[str, Any]: 优化结果
        """
        start_time = time.time()
        
        try:
            # 检查缓存
            cache_key = f"optimize_{prompt}"
            cached_result = self.optimization_cache.get(cache_key)
            if cached_result:
                logger.info("使用缓存的优化结果")
                return cached_result
                
            # 获取相似的历史提示词
            similar_prompts = await self._get_similar_prompts(prompt)
            logger.info(f"找到 {len(similar_prompts)} 个相似的历史提示词")
            
            # 获取项目上下文信息
            project_context = await self._get_project_context()
            logger.info("获取到项目上下文信息")
            
            # 使用RAG增强提示词
            rag_result = await self._enhance_with_rag(prompt, context_files)
            logger.info(f"RAG增强结果: 找到 {len(rag_result.get('contexts', []))} 个相关上下文")
            
            # 组合优化上下文
            optimization_context = {
                "original_prompt": prompt,
                "similar_prompts": similar_prompts,
                "rag_context": rag_result,
                "project_context": project_context,
                "context_files": context_files or [],
                "tech_stack": await self._get_tech_stack(),
                "file_structure": await self._get_file_structure()
            }
            
            # 生成优化后的提示词
            optimized_prompt = await self._generate_optimized_prompt(optimization_context)
            
            # 评估优化结果
            evaluation = await self._evaluate_prompt(optimized_prompt)
            
            # 准备返回结果
            result = {
                "status": "success",
                "original_prompt": prompt,
                "optimized_prompt": optimized_prompt,
                "evaluation": evaluation,
                "optimization_time": time.time() - start_time,
                "context_info": {
                    "similar_prompts_count": len(similar_prompts),
                    "rag_contexts_count": len(rag_result.get("contexts", [])),
                    "project_context_available": bool(project_context)
                }
            }
            
            # 缓存结果
            self.optimization_cache.set(cache_key, result)
            
            # 记录性能指标
            await self._log_performance_metrics(prompt, result)
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"优化提示词失败: {error_msg}")
            return {
                "status": "error",
                "original_prompt": prompt,
                "optimized_prompt": prompt,
                "error": error_msg,
                "optimization_time": time.time() - start_time
            }
            
    async def _get_similar_prompts(self, prompt: str) -> List[str]:
        """获取相似的历史提示词
        
        Args:
            prompt: 当前提示词
            
        Returns:
            List[str]: 相似提示词列表
        """
        try:
            # 检查缓存
            cache_key = f"similar_{prompt}"
            cached_result = self.similar_prompts_cache.get(cache_key)
            if cached_result:
                return cached_result
                
            # 从向量存储中搜索相似提示词
            similar_docs = await self.vector_store.similarity_search(
                prompt,
                k=3,
                filter_dict={"type": "prompt_history"}
            )
            
            # 提取提示词内容
            similar_prompts = [doc.page_content for doc in similar_docs]
            
            # 缓存结果
            self.similar_prompts_cache.set(cache_key, similar_prompts)
            
            return similar_prompts
            
        except Exception as e:
            logger.error(f"获取相似提示词失败: {str(e)}")
            return []
            
    async def _enhance_with_rag(
        self,
        prompt: str,
        context_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """使用RAG增强提示词
        
        Args:
            prompt: 原始提示词
            context_files: 上下文文件列表
            
        Returns:
            Dict[str, Any]: RAG增强结果
        """
        try:
            # 检查缓存
            cache_key = f"rag_{prompt}_{'-'.join(context_files or [])}"
            cached_result = self.rag_cache.get(cache_key)
            if cached_result:
                return cached_result
                
            # 获取相关上下文
            contexts = []
            if context_files:
                for file_path in context_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            contexts.append(f.read())
                    except Exception as e:
                        logger.warning(f"读取上下文文件失败: {str(e)}")
                        
            # 从向量存储中获取相关文档
            relevant_docs = await self.vector_store.similarity_search(
                prompt,
                k=3,
                filter_dict={"type": "context"}
            )
            
            # 合并上下文
            all_contexts = contexts + [doc.page_content for doc in relevant_docs]
            
            # 使用RAG模板生成增强结果
            rag_result = {
                "contexts": all_contexts,
                "file_contexts": contexts,
                "retrieved_contexts": [doc.page_content for doc in relevant_docs]
            }
            
            # 缓存结果
            self.rag_cache.set(cache_key, rag_result)
            
            return rag_result
            
        except Exception as e:
            logger.error(f"RAG增强失败: {str(e)}")
            return {
                "contexts": [],
                "file_contexts": [],
                "retrieved_contexts": []
            }

    async def _get_tech_stack(self) -> Dict[str, Any]:
        """获取项目技术栈信息"""
        try:
            results = await self.vector_store.similarity_search(
                "技术栈 前端框架 后端框架 数据库",
                k=3,
                filter_dict={"type": "tech_stack"}
            )
            tech_stack = {
                "frontend": [],
                "backend": [],
                "database": [],
                "tools": []
            }
            for doc in results:
                content = doc.page_content
                metadata = doc.metadata
                if "frontend" in metadata:
                    tech_stack["frontend"].extend(metadata["frontend"])
                if "backend" in metadata:
                    tech_stack["backend"].extend(metadata["backend"])
                if "database" in metadata:
                    tech_stack["database"].extend(metadata["database"])
                if "tools" in metadata:
                    tech_stack["tools"].extend(metadata["tools"])
            return tech_stack
        except Exception as e:
            logger.error(f"获取技术栈信息失败: {str(e)}")
            return {}

    async def _get_file_structure(self) -> Dict[str, Any]:
        """获取项目文件结构信息"""
        try:
            results = await self.vector_store.similarity_search(
                "项目结构 目录组织",
                k=3,
                filter_dict={"type": "file_structure"}
            )
            structure = {
                "root_dir": "",
                "src_dir": "",
                "components_dir": "",
                "pages_dir": "",
                "api_dir": "",
                "styles_dir": "",
                "assets_dir": ""
            }
            for doc in results:
                content = doc.page_content
                metadata = doc.metadata
                for key in structure.keys():
                    if key in metadata:
                        structure[key] = metadata[key]
            return structure
        except Exception as e:
            logger.error(f"获取文件结构信息失败: {str(e)}")
            return {}

    async def _generate_optimized_prompt(self, optimization_context: Dict[str, Any]) -> str:
        """生成优化后的提示词"""
        try:
            # 构建更详细的提示词模板
            template = """请根据以下信息生成一个详细的优化提示词：

原始需求：
{original_prompt}

技术栈信息：
前端: {frontend_tech}
后端: {backend_tech}
数据库: {database_tech}
工具链: {tools}

文件结构：
根目录: {root_dir}
源码目录: {src_dir}
组件目录: {components_dir}
页面目录: {pages_dir}
API目录: {api_dir}
样式目录: {styles_dir}
资源目录: {assets_dir}

相似历史提示词：
{similar_prompts}

项目上下文：
{project_context}

RAG上下文：
{rag_context}

请生成一个优化后的提示词，确保包含：
1. 详细的技术实现要求
2. 具体的文件结构和组织方式
3. 组件设计和数据流说明
4. UI/UX设计指南
5. API接口定义
6. 数据模型设计
7. 测试要求
8. 性能考虑
"""
            
            tech_stack = optimization_context.get("tech_stack", {})
            file_structure = optimization_context.get("file_structure", {})
            
            messages = [{
                "role": "system",
                "content": self._truncate_text(SYSTEM_PROMPT, 500)
            }, {
                "role": "user",
                "content": template.format(
                    original_prompt=optimization_context["original_prompt"],
                    frontend_tech=", ".join(tech_stack.get("frontend", [])),
                    backend_tech=", ".join(tech_stack.get("backend", [])),
                    database_tech=", ".join(tech_stack.get("database", [])),
                    tools=", ".join(tech_stack.get("tools", [])),
                    root_dir=file_structure.get("root_dir", ""),
                    src_dir=file_structure.get("src_dir", ""),
                    components_dir=file_structure.get("components_dir", ""),
                    pages_dir=file_structure.get("pages_dir", ""),
                    api_dir=file_structure.get("api_dir", ""),
                    styles_dir=file_structure.get("styles_dir", ""),
                    assets_dir=file_structure.get("assets_dir", ""),
                    similar_prompts="\n".join(optimization_context.get("similar_prompts", [])),
                    project_context=json.dumps(optimization_context.get("project_context", {}), ensure_ascii=False),
                    rag_context=json.dumps(optimization_context.get("rag_context", {}), ensure_ascii=False)
                )
            }]
            
            result = await self.llm.ainvoke(messages)
            return result.content
            
        except Exception as e:
            logger.error(f"生成优化后的提示词失败: {str(e)}")
            return optimization_context["original_prompt"]

    async def _evaluate_prompt(self, prompt: str) -> Dict[str, Any]:
        """评估优化后的提示词"""
        try:
            messages = [{
                "role": "system",
                "content": SYSTEM_PROMPT
            }, {
                "role": "user",
                "content": self.evaluation_prompt.format(prompt=prompt)
            }]
            
            evaluation_result = await self.llm.ainvoke(messages)
            
            try:
                # 提取并清理JSON
                content = evaluation_result.content
                # 查找JSON部分
                json_pattern = r'\{[\s\S]*\}'
                import re
                json_match = re.search(json_pattern, content)
                
                if json_match:
                    json_str = json_match.group()
                    # 清理格式问题
                    json_str = json_str.replace('\n', '').replace('    ', '')
                    result = json.loads(json_str)
                else:
                    raise json.JSONDecodeError("No JSON found in content", content, 0)
                    
            except json.JSONDecodeError as e:
                logger.warning(f"评估结果解析失败: {str(e)}")
                logger.debug(f"原始评估结果: {evaluation_result.content}")
                result = self._get_default_evaluation()
            
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
            return self._get_default_evaluation()

    def _get_default_evaluation(self) -> Dict[str, Any]:
        """获取默认的评估结果"""
        return {
            "feedback": ["评估过程出错"],
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

    async def save_optimization_result(
        self,
        original_prompt: str,
        optimized_prompt: str,
        evaluation: Dict[str, Any],
        context: Dict[str, Any],
        optimization_time: float
    ) -> bool:
        """保存优化结果到向量数据库
        
        Args:
            original_prompt: 原始提示词
            optimized_prompt: 优化后的提示词
            evaluation: 评估结果
            context: 上下文信息
            optimization_time: 优化耗时
            
        Returns:
            bool: 是否保存成功
        """
        try:
            # 准备元数据
            metadata = {
                "original_prompt": original_prompt,
                "evaluation": evaluation,
                "context": context,
                "optimization_time": optimization_time,
                "timestamp": datetime.now().isoformat(),
                "model_version": settings.OPENAI_MODEL,
                "type": "prompt_history"  # 添加类型标记
            }
            
            # 保存到向量存储
            await self.vector_store.add_texts(
                texts=[optimized_prompt],
                metadatas=[metadata]
            )
            
            logger.info("优化结果已成功保存到向量数据库")
            return True
            
        except Exception as e:
            logger.error(f"保存优化结果失败: {str(e)}")
            return False
            
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
        
    async def _log_performance_metrics(self, prompt: str, result: Dict[str, Any]):
        """记录性能指标
        
        Args:
            prompt: 原始提示词
            result: 优化结果
        """
        try:
            # 这里可以实现具体的指标记录逻辑
            # 例如：发送到监控系统、写入日志等
            if result.get("optimization_time", 0) > settings.PROMPT_OPTIMIZATION_CONFIG["slow_query_threshold"]:
                logger.warning(f"检测到慢查询: {json.dumps(result, ensure_ascii=False)}")
                
            logger.info(f"性能指标: {json.dumps(result, ensure_ascii=False)}")
            
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

    def _truncate_text(self, text: str, max_length: int) -> str:
        """截断文本到指定长度
        
        Args:
            text: 要截断的文本
            max_length: 最大长度
            
        Returns:
            str: 截断后的文本
        """
        if not text:
            return ""
            
        if len(text) <= max_length:
            return text
            
        # 保留前后文，中间用省略号
        half_length = (max_length - 3) // 2
        return f"{text[:half_length]}...{text[-half_length:]}" 