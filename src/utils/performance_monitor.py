"""性能监控模块"""
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path
import statistics
import asyncio
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """性能监控类，用于跟踪和分析嵌入分析的性能"""
    
    def __init__(self, log_dir: Optional[Path] = None):
        """初始化性能监控器
        
        Args:
            log_dir: 日志目录，如果为None则使用默认目录
        """
        self.log_dir = Path(log_dir) if log_dir else Path("logs") / "performance"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化性能指标
        self.metrics = {
            "file_processing": [],
            "embedding_generation": [],
            "vector_search": [],
            "cache_hits": 0,
            "cache_misses": 0,
            "total_files_processed": 0,
            "skipped_files": 0,
            "total_chars_processed": 0,
            "total_tokens_processed": 0
        }
        
        # 记录最近的调用时间
        self.last_save_time = time.time()
        self.save_interval = 300  # 5分钟保存一次
        
        logger.info(f"性能监控器初始化完成，日志目录: {self.log_dir}")
        
    def record_file_processing(self, file_path: str, processing_time: float, 
                               file_size: int, is_cached: bool):
        """记录文件处理性能
        
        Args:
            file_path: 文件路径
            processing_time: 处理时间（秒）
            file_size: 文件大小（字节）
            is_cached: 是否命中缓存
        """
        self.metrics["file_processing"].append({
            "timestamp": datetime.now().isoformat(),
            "file_path": file_path,
            "processing_time": processing_time,
            "file_size": file_size,
            "is_cached": is_cached,
            "processing_speed": file_size / processing_time if processing_time > 0 else 0
        })
        
        self.metrics["total_files_processed"] += 1
        self.metrics["total_chars_processed"] += file_size
        
        if is_cached:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1
            
        # 自动保存
        self._auto_save()
        
    def record_embedding_generation(self, text_length: int, token_count: int, 
                                    generation_time: float, model_name: str):
        """记录嵌入生成性能
        
        Args:
            text_length: 文本长度
            token_count: Token数量
            generation_time: 生成时间（秒）
            model_name: 模型名称
        """
        self.metrics["embedding_generation"].append({
            "timestamp": datetime.now().isoformat(),
            "text_length": text_length,
            "token_count": token_count,
            "generation_time": generation_time,
            "model_name": model_name,
            "tokens_per_second": token_count / generation_time if generation_time > 0 else 0
        })
        
        self.metrics["total_tokens_processed"] += token_count
        
        # 自动保存
        self._auto_save()
        
    def record_vector_search(self, query: str, result_count: int, search_time: float, store_type: str):
        """记录向量搜索性能
        
        Args:
            query: 查询文本
            result_count: 结果数量
            search_time: 搜索时间（秒）
            store_type: 存储类型
        """
        self.metrics["vector_search"].append({
            "timestamp": datetime.now().isoformat(),
            "query_length": len(query),
            "result_count": result_count,
            "search_time": search_time,
            "store_type": store_type
        })
        
        # 自动保存
        self._auto_save()
        
    def record_skipped_file(self, file_path: str, reason: str):
        """记录跳过的文件
        
        Args:
            file_path: 文件路径
            reason: 跳过原因
        """
        self.metrics["skipped_files"] += 1
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取性能统计数据
        
        Returns:
            Dict[str, Any]: 性能统计
        """
        stats = {
            "cache_hit_ratio": 0,
            "avg_file_processing_time": 0,
            "avg_embedding_generation_time": 0,
            "avg_vector_search_time": 0,
            "total_files_processed": self.metrics["total_files_processed"],
            "skipped_files": self.metrics["skipped_files"],
            "total_chars_processed": self.metrics["total_chars_processed"],
            "total_tokens_processed": self.metrics["total_tokens_processed"]
        }
        
        # 计算缓存命中率
        total_cache_requests = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        if total_cache_requests > 0:
            stats["cache_hit_ratio"] = self.metrics["cache_hits"] / total_cache_requests
            
        # 计算平均文件处理时间
        if self.metrics["file_processing"]:
            stats["avg_file_processing_time"] = statistics.mean(
                [item["processing_time"] for item in self.metrics["file_processing"]]
            )
            
        # 计算平均嵌入生成时间
        if self.metrics["embedding_generation"]:
            stats["avg_embedding_generation_time"] = statistics.mean(
                [item["generation_time"] for item in self.metrics["embedding_generation"]]
            )
            
        # 计算平均向量搜索时间
        if self.metrics["vector_search"]:
            stats["avg_vector_search_time"] = statistics.mean(
                [item["search_time"] for item in self.metrics["vector_search"]]
            )
            
        return stats
        
    def save_metrics(self):
        """保存性能指标到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = self.log_dir / f"performance_metrics_{timestamp}.json"
        
        try:
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(self.metrics, f, ensure_ascii=False, indent=2)
            logger.info(f"性能指标已保存到: {metrics_file}")
        except Exception as e:
            logger.error(f"保存性能指标失败: {str(e)}")
            
    def _auto_save(self):
        """自动保存性能指标"""
        current_time = time.time()
        if current_time - self.last_save_time > self.save_interval:
            self.save_metrics()
            self.last_save_time = current_time
            
    def reset(self):
        """重置性能指标"""
        self.metrics = {
            "file_processing": [],
            "embedding_generation": [],
            "vector_search": [],
            "cache_hits": 0,
            "cache_misses": 0,
            "total_files_processed": 0,
            "skipped_files": 0,
            "total_chars_processed": 0,
            "total_tokens_processed": 0
        }
        
    @contextmanager
    def measure_time(self, metric_type: str, **kwargs):
        """计时上下文管理器
        
        Args:
            metric_type: 指标类型
            **kwargs: 额外参数
        """
        start_time = time.time()
        yield
        elapsed_time = time.time() - start_time
        
        if metric_type == "file_processing":
            self.record_file_processing(
                file_path=kwargs.get("file_path", "unknown"),
                processing_time=elapsed_time,
                file_size=kwargs.get("file_size", 0),
                is_cached=kwargs.get("is_cached", False)
            )
        elif metric_type == "embedding_generation":
            self.record_embedding_generation(
                text_length=kwargs.get("text_length", 0),
                token_count=kwargs.get("token_count", 0),
                generation_time=elapsed_time,
                model_name=kwargs.get("model_name", "unknown")
            )
        elif metric_type == "vector_search":
            self.record_vector_search(
                query=kwargs.get("query", ""),
                result_count=kwargs.get("result_count", 0),
                search_time=elapsed_time,
                store_type=kwargs.get("store_type", "unknown")
            )
            
# 创建全局性能监控实例
performance_monitor = PerformanceMonitor() 