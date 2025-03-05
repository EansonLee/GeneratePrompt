"""缓存管理模块"""
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """智能缓存管理器"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 7200):
        """初始化缓存管理器
        
        Args:
            max_size: 最大缓存条目数
            ttl: 缓存生存时间（秒）
        """
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.max_size = max_size
        self.ttl = ttl
        self.hit_counts: Dict[str, int] = {}
        self.last_cleanup = datetime.now()
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[Any]: 缓存的值，如果不存在或过期则返回None
        """
        if key not in self.cache:
            return None
            
        value, timestamp = self.cache[key]
        if datetime.now() - timestamp > timedelta(seconds=self.ttl):
            self._remove_expired(key)
            return None
            
        # 更新访问统计
        self.hit_counts[key] = self.hit_counts.get(key, 0) + 1
        return value
        
    def set(self, key: str, value: Any):
        """设置缓存值
        
        Args:
            key: 缓存键
            value: 要缓存的值
        """
        # 检查是否需要清理过期缓存
        self._cleanup_if_needed()
        
        # 如果达到最大大小，执行淘汰
        if len(self.cache) >= self.max_size:
            self._evict()
            
        self.cache[key] = (value, datetime.now())
        self.hit_counts[key] = 0
        
    def _remove_expired(self, key: str):
        """移除过期的缓存项
        
        Args:
            key: 要移除的缓存键
        """
        if key in self.cache:
            del self.cache[key]
        if key in self.hit_counts:
            del self.hit_counts[key]
            
    def _cleanup_if_needed(self):
        """定期清理过期缓存"""
        if (datetime.now() - self.last_cleanup).total_seconds() > 3600:  # 每小时清理一次
            logger.info("执行定期缓存清理")
            current_time = datetime.now()
            expired_keys = [
                key for key, (_, timestamp) in self.cache.items()
                if (current_time - timestamp).total_seconds() > self.ttl
            ]
            for key in expired_keys:
                self._remove_expired(key)
            self.last_cleanup = current_time
            
    def _evict(self):
        """智能缓存淘汰
        
        使用访问频率和时间的混合策略进行淘汰
        """
        if not self.cache:
            return
            
        # 计算每个缓存项的得分
        current_time = datetime.now()
        scores = {}
        for key in self.cache:
            hits = self.hit_counts.get(key, 0)
            age = (current_time - self.cache[key][1]).total_seconds()
            # 得分 = 访问次数 / (年龄 + 1)
            scores[key] = hits / (age + 1)
            
        # 淘汰得分最低的项
        key_to_evict = min(scores.keys(), key=lambda k: scores[k])
        self._remove_expired(key_to_evict)
        logger.info(f"缓存淘汰: {key_to_evict}")
        
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.hit_counts.clear()
        self.last_cleanup = datetime.now()
        
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息
        
        Returns:
            Dict[str, Any]: 缓存统计信息
        """
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "total_hits": sum(self.hit_counts.values()),
            "items_with_hits": len([k for k, v in self.hit_counts.items() if v > 0]),
            "last_cleanup": self.last_cleanup.isoformat()
        } 