"""缓存管理模块"""
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os
import json
from pathlib import Path
from hashlib import sha256

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

class FileHashCache:
    """基于文件哈希的缓存管理器"""
    
    def __init__(self, cache_dir: Path):
        """初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "file_hash_cache.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()
        
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """加载缓存数据
        
        Returns:
            Dict[str, Dict[str, Any]]: 缓存数据
        """
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"加载缓存文件失败: {str(e)}")
        return {}
        
    def _save_cache(self):
        """保存缓存数据"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存缓存文件失败: {str(e)}")
            
    def get_file_hash(self, file_path: Path) -> str:
        """计算文件的SHA256哈希值
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 文件的哈希值
        """
        try:
            with open(file_path, 'rb') as f:
                return sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"计算文件哈希失败 {file_path}: {str(e)}")
            return ""
            
    def is_file_changed(self, file_path: Path) -> bool:
        """检查文件是否已更改
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 如果文件已更改或未缓存则返回True
        """
        if not file_path.exists():
            return True
            
        current_hash = self.get_file_hash(file_path)
        if not current_hash:
            return True
            
        cached_info = self.cache.get(str(file_path))
        if not cached_info:
            return True
            
        return cached_info['hash'] != current_hash
        
    def get_cached_embeddings(self, file_path: Path) -> Optional[Any]:
        """获取文件的缓存嵌入向量
        
        Args:
            file_path: 文件路径
            
        Returns:
            Optional[Any]: 缓存的嵌入向量，如果不存在则返回None
        """
        cached_info = self.cache.get(str(file_path))
        if cached_info and not self.is_file_changed(file_path):
            return cached_info.get('embeddings')
        return None
        
    def update_cache(self, file_path: Path, embeddings: Any):
        """更新文件的缓存信息
        
        Args:
            file_path: 文件路径
            embeddings: 文件的嵌入向量
        """
        self.cache[str(file_path)] = {
            'hash': self.get_file_hash(file_path),
            'last_processed': str(datetime.now()),
            'embeddings': embeddings
        }
        self._save_cache()
        
    def clear_expired_cache(self, max_age_days: int = 7):
        """清理过期的缓存数据
        
        Args:
            max_age_days: 缓存最大保留天数
        """
        now = datetime.now()
        expired_files = []
        
        for file_path, info in self.cache.items():
            try:
                last_processed = datetime.fromisoformat(info['last_processed'])
                age_days = (now - last_processed).days
                
                if age_days > max_age_days:
                    expired_files.append(file_path)
            except Exception as e:
                logger.warning(f"处理缓存条目失败 {file_path}: {str(e)}")
                expired_files.append(file_path)
                
        for file_path in expired_files:
            del self.cache[file_path]
            
        if expired_files:
            self._save_cache()
            logger.info(f"已清理 {len(expired_files)} 个过期缓存条目") 