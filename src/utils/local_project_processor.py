"""本地项目处理器"""
import os
import time
import re
import logging
import asyncio
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
import aiofiles
import platform
from src.utils.vector_store import VectorStore

logger = logging.getLogger(__name__)

class LocalProjectProcessor:
    """本地项目处理器，用于处理本地项目路径"""
    
    # 支持的文件扩展名
    SUPPORTED_EXTENSIONS = {
        # 文本文件
        '.txt': 'text',
        '.md': 'markdown',
        '.markdown': 'markdown',
        # 代码文件
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'react',
        '.ts': 'typescript',
        '.tsx': 'react',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.less': 'less',
        '.java': 'java',
        '.kt': 'kotlin',
        '.swift': 'swift',
        '.c': 'c',
        '.cpp': 'cpp',
        '.h': 'h',
        '.cs': 'csharp',
        '.go': 'go',
        '.php': 'php',
        '.rb': 'ruby',
        '.rs': 'rust',
        # 配置文件
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.xml': 'xml',
        '.ini': 'ini',
        '.env': 'env',
        # 脚本文件
        '.sh': 'shell',
        '.bat': 'batch',
        '.ps1': 'powershell',
    }
    
    # 忽略的目录 - 扩展更多忽略的目录
    IGNORED_DIRS = {
        # 常规忽略目录
        'node_modules', 'venv', '.venv', '__pycache__', '.git',
        'dist', 'build', '.idea', '.vscode', 'target', 'bin',
        'obj', '.next', '.nuxt', 'coverage', 'htmlcov',
        'logs', 'log', 'tmp', 'temp', 'cache', '.cache',
        # 额外忽略目录
        'out', 'output', 'public', 'static', 'assets', 'images', 
        'vendor', 'packages', 'bower_components', 'jspm_packages',
        'typings', 'tsd_typings', 'cordova', 'platforms', 'plugins',
        # 构建产物和中间文件目录
        'generated', 'gen', 'release', 'debug', 'bin-debug', 'bin-release',
        # 测试相关目录
        'test-output', 'mochawesome-report', 'jest-coverage'
    }
    
    # 忽略的文件模式 (新增)
    IGNORED_FILE_PATTERNS = [
        r'.*\.min\.js$',          # 压缩JS
        r'.*\.min\.css$',         # 压缩CSS
        r'.*\.d\.ts$',            # TypeScript声明文件
        r'.*\.test\..*$',         # 测试文件
        r'.*\.spec\..*$',         # 测试文件
        r'.*\.bundle\.js$',       # 打包JS
        r'.*\.map$',              # 源码映射文件
        r'.*\.lock$',             # 锁文件
        r'.*-lock\.json$',        # 包锁文件
        r'.*\.svg$',              # SVG图像文件
        r'.*\.ttf$',              # 字体文件
        r'.*\.woff2?$',           # 网页字体文件
        r'.*\.eot$',              # 网页字体文件
        r'.*\.ai$',               # Adobe Illustrator文件
        r'.*\.psd$'               # Photoshop文件
    ]
    
    # 特殊关注的文件
    SPECIAL_FILES = {
        'package.json': 'npm_config',
        'tsconfig.json': 'typescript_config',
        'README.md': 'documentation',
        '.env': 'environment',
        'vite.config.ts': 'vite_config',
        'vite.config.js': 'vite_config',
        'next.config.js': 'next_config',
        'next.config.ts': 'next_config',
        'pom.xml': 'maven_config',
        'build.gradle': 'gradle_config',
        'pubspec.yaml': 'flutter_config',
        'Gemfile': 'ruby_config',
        'Cargo.toml': 'rust_config',
        'requirements.txt': 'python_dependencies',
        'Dockerfile': 'docker_config',
    }
    
    # 不同操作系统的路径分隔符
    PATH_SEPARATORS = {
        'Windows': '\\',
        'Darwin': '/',  # macOS
        'Linux': '/',
    }
    
    # 文件大小限制 (新增)
    MAX_FILE_SIZE = 1024 * 1024  # 1MB
    
    # 缓存实现 (新增)
    _cache = {}
    _cache_file = None
    _cache_ttl = 24 * 60 * 60  # 24小时
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """初始化本地项目处理器
        
        Args:
            vector_store: 向量存储实例
        """
        self.vector_store = vector_store
        self.os_name = platform.system()
        self.path_separator = self.PATH_SEPARATORS.get(self.os_name, os.path.sep)
        
        # 初始化缓存文件路径
        if LocalProjectProcessor._cache_file is None:
            data_dir = os.getenv("DATA_DIR", "data")
            os.makedirs(data_dir, exist_ok=True)
            LocalProjectProcessor._cache_file = os.path.join(data_dir, "project_analysis_cache.json")
            self._load_cache()
            
        logger.info(f"本地项目处理器初始化完成，系统: {self.os_name}, 路径分隔符: {self.path_separator}")
    
    def _load_cache(self):
        """加载缓存数据"""
        try:
            if os.path.exists(LocalProjectProcessor._cache_file):
                with open(LocalProjectProcessor._cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                    # 清理过期缓存
                    current_time = time.time()
                    valid_cache = {}
                    for key, entry in cache_data.items():
                        if current_time - entry.get('timestamp', 0) < self._cache_ttl:
                            valid_cache[key] = entry
                    
                    LocalProjectProcessor._cache = valid_cache
                    logger.info(f"已加载项目分析缓存: {len(valid_cache)}条有效记录")
            else:
                logger.info("缓存文件不存在，使用空缓存")
                LocalProjectProcessor._cache = {}
        except Exception as e:
            logger.error(f"加载缓存失败: {str(e)}")
            LocalProjectProcessor._cache = {}
    
    def _save_cache(self):
        """保存缓存数据"""
        try:
            # 限制缓存大小
            if len(LocalProjectProcessor._cache) > 100:
                # 按时间戳排序，保留最新的100条
                sorted_cache = sorted(
                    LocalProjectProcessor._cache.items(),
                    key=lambda x: x[1].get('timestamp', 0)
                )
                LocalProjectProcessor._cache = dict(sorted_cache[-100:])
                
            with open(LocalProjectProcessor._cache_file, 'w', encoding='utf-8') as f:
                json.dump(LocalProjectProcessor._cache, f, ensure_ascii=False)
                logger.info(f"已保存项目分析缓存: {len(LocalProjectProcessor._cache)}条记录")
        except Exception as e:
            logger.error(f"保存缓存失败: {str(e)}")
    
    def is_supported_file(self, filename: str) -> bool:
        """检查文件是否支持
        
        Args:
            filename: 文件名
            
        Returns:
            bool: 是否支持该文件类型
        """
        # 检查是否匹配忽略模式
        for pattern in self.IGNORED_FILE_PATTERNS:
            if re.match(pattern, filename):
                return False
                
        ext = Path(filename).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS
    
    def is_file_size_acceptable(self, file_path: str) -> bool:
        """检查文件大小是否可接受
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 文件大小是否可接受
        """
        try:
            return os.path.getsize(file_path) <= self.MAX_FILE_SIZE
        except Exception:
            return False
    
    def get_file_type(self, filename: str) -> str:
        """获取文件类型
        
        Args:
            filename: 文件名
            
        Returns:
            str: 文件类型
        """
        # 检查是否为特殊文件
        if filename in self.SPECIAL_FILES:
            return self.SPECIAL_FILES[filename]
        
        # 检查扩展名
        ext = Path(filename).suffix.lower()
        return self.SUPPORTED_EXTENSIONS.get(ext, 'unknown')
    
    def normalize_path(self, path: str) -> str:
        """规范化路径，处理不同操作系统的路径分隔符
        
        Args:
            path: 路径
            
        Returns:
            str: 规范化后的路径
        """
        # 转换为当前系统的路径格式
        return str(Path(path))
    
    def _compute_project_hash(self, project_path: str) -> str:
        """计算项目哈希值，用于缓存查找
        
        Args:
            project_path: 项目路径
            
        Returns:
            str: 项目哈希值
        """
        # 使用项目路径和最后修改时间计算哈希
        try:
            path_hash = hashlib.md5(project_path.encode()).hexdigest()
            
            # 获取根目录下重要文件的最后修改时间
            key_files = ['package.json', 'requirements.txt', 'pom.xml', 'build.gradle', 'pubspec.yaml']
            mod_times = []
            
            for key_file in key_files:
                file_path = os.path.join(project_path, key_file)
                if os.path.exists(file_path):
                    mod_times.append(str(os.path.getmtime(file_path)))
            
            # 组合哈希
            if mod_times:
                combined = path_hash + ''.join(sorted(mod_times))
                return hashlib.md5(combined.encode()).hexdigest()
            
            return path_hash
        except Exception as e:
            logger.error(f"计算项目哈希失败: {str(e)}")
            return hashlib.md5(project_path.encode()).hexdigest()
    
    async def _read_file_content(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """读取文件内容
        
        Args:
            file_path: 文件路径
            
        Returns:
            Tuple[Optional[str], Optional[str]]: (文件内容, 错误信息)
        """
        try:
            # 检查文件大小
            if not self.is_file_size_acceptable(file_path):
                return None, f"文件过大: {os.path.getsize(file_path)} 字节"
                
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()
                return content, None
        except Exception as e:
            error = f"读取文件 {file_path} 失败: {str(e)}"
            logger.warning(error)
            return None, error
    
    async def analyze_project(self, project_path: str, branch: str = "main") -> Dict[str, Any]:
        """分析本地项目
        
        Args:
            project_path: 本地项目路径
            branch: 分支名（如果是Git仓库）
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        start_time = time.time()
        
        try:
            # 规范化路径
            normalized_path = self.normalize_path(project_path)
            logger.info(f"开始分析本地项目: {normalized_path}")
            
            # 检查路径是否存在
            if not os.path.exists(normalized_path):
                return {
                    "status": "error",
                    "error": f"项目路径不存在: {normalized_path}"
                }
            
            # 检查是否为目录
            if not os.path.isdir(normalized_path):
                return {
                    "status": "error",
                    "error": f"指定的路径不是目录: {normalized_path}"
                }
            
            # 计算项目哈希值
            project_hash = self._compute_project_hash(normalized_path)
            logger.info(f"项目哈希值: {project_hash}")
            
            # 检查缓存
            if project_hash in LocalProjectProcessor._cache:
                cache_entry = LocalProjectProcessor._cache[project_hash]
                cache_time = cache_entry.get('timestamp', 0)
                
                # 检查缓存是否过期
                if time.time() - cache_time < self._cache_ttl:
                    logger.info(f"使用缓存的项目分析结果，缓存时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cache_time))}")
                    cache_entry['from_cache'] = True
                    
                    # 更新缓存时间戳
                    cache_entry['timestamp'] = time.time()
                    self._save_cache()
                    
                    # 返回缓存的结果
                    return cache_entry.get('result', {})
            
            # 获取项目文件结构
            file_structure = self._get_file_structure(normalized_path)
            
            # 获取支持的文件列表
            supported_files = []
            self._collect_supported_files(normalized_path, supported_files)
            
            # 智能采样：如果文件超过阈值，进行采样处理
            max_files = 100  # 最大处理文件数
            file_sample = supported_files
            sample_rate = 1.0
            
            if len(supported_files) > max_files:
                sample_rate = max_files / len(supported_files)
                
                # 确保包含关键文件
                key_files = [f for f in supported_files if os.path.basename(f) in self.SPECIAL_FILES]
                
                # 从其余文件中随机采样
                remaining = [f for f in supported_files if f not in key_files]
                import random
                random.seed(42)  # 固定随机种子，保证结果可重现
                
                # 按文件类型分组，确保每种类型都有代表性样本
                file_types = {}
                for f in remaining:
                    file_type = self.get_file_type(os.path.basename(f))
                    if file_type not in file_types:
                        file_types[file_type] = []
                    file_types[file_type].append(f)
                
                # 从每种类型中按比例选择文件
                sampled_remaining = []
                for file_type, files in file_types.items():
                    count = max(1, int(len(files) * sample_rate))
                    sampled_remaining.extend(random.sample(files, min(count, len(files))))
                
                # 组合关键文件和采样文件
                file_sample = key_files + sampled_remaining
                
                # 如果仍然超过最大文件数，进行截断
                if len(file_sample) > max_files:
                    file_sample = file_sample[:max_files]
                
                logger.info(f"文件数量超过阈值，进行采样: {len(supported_files)} -> {len(file_sample)} 文件")
            
            # 读取文件内容
            files_content = {}
            file_errors = {}
            total_files = len(file_sample)
            processed_files = 0
            
            # 控制并发数量
            semaphore = asyncio.Semaphore(10)  # 最多同时处理10个文件
            
            async def process_file(file_path: str):
                async with semaphore:
                    nonlocal processed_files
                    rel_path = os.path.relpath(file_path, normalized_path)
                    content, error = await self._read_file_content(file_path)
                    
                    if content is not None:
                        file_type = self.get_file_type(os.path.basename(file_path))
                        files_content[rel_path] = {
                            "content": content,
                            "file_type": file_type,
                            "size": len(content)
                        }
                    else:
                        file_errors[rel_path] = error
                    
                    processed_files += 1
                    if processed_files % 10 == 0:
                        logger.info(f"已处理 {processed_files}/{total_files} 个文件...")
            
            # 并发处理文件
            tasks = [process_file(file_path) for file_path in file_sample]
            await asyncio.gather(*tasks)
            
            # 添加到向量数据库
            if self.vector_store and files_content:
                logger.info(f"向向量数据库添加 {len(files_content)} 个文件内容...")
                
                # 批量添加以提高性能
                batch_size = 10
                texts = []
                metadatas = []
                
                for rel_path, file_info in files_content.items():
                    # 准备元数据
                    metadata = {
                        "file_name": os.path.basename(rel_path),
                        "file_path": rel_path,
                        "file_type": file_info["file_type"],
                        "project_path": normalized_path,
                        "timestamp": time.time()
                    }
                    
                    texts.append(file_info["content"])
                    metadatas.append(metadata)
                    
                    # 达到批量大小，添加到向量数据库
                    if len(texts) >= batch_size:
                        await self.vector_store.add_texts(texts=texts, metadatas=metadatas)
                        texts = []
                        metadatas = []
                
                # 添加剩余的文件
                if texts:
                    await self.vector_store.add_texts(texts=texts, metadatas=metadatas)
            
            # 准备返回结果
            result = {
                "status": "success",
                "project_path": normalized_path,
                "total_files": len(supported_files),  # 原始文件总数
                "processed_files": processed_files,
                "sampled_files": len(file_sample),    # 采样后的文件数
                "sample_rate": sample_rate,
                "supported_files": len(files_content),
                "error_files": len(file_errors),
                "file_structure": file_structure,
                "file_errors": file_errors,
                "processing_time": time.time() - start_time
            }
            
            # 保存到缓存
            LocalProjectProcessor._cache[project_hash] = {
                "result": result,
                "timestamp": time.time()
            }
            self._save_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"分析本地项目失败: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": f"分析本地项目失败: {str(e)}"
            }
    
    def _get_file_structure(self, project_path: str) -> Dict[str, Any]:
        """获取项目文件结构
        
        Args:
            project_path: 项目路径
            
        Returns:
            Dict[str, Any]: 文件结构
        """
        try:
            structure = {}
            
            def process_dir(dir_path: str, parent_dict: Dict[str, Any]):
                """递归处理目录"""
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    
                    # 跳过忽略的目录
                    if os.path.isdir(item_path) and item in self.IGNORED_DIRS:
                        continue
                    
                    if os.path.isdir(item_path):
                        parent_dict[item] = {'type': 'directory', 'items': {}}
                        process_dir(item_path, parent_dict[item]['items'])
                    elif os.path.isfile(item_path) and self.is_supported_file(item):
                        # 检查文件大小
                        if self.is_file_size_acceptable(item_path):
                            parent_dict[item] = {
                                'type': 'file',
                                'path': os.path.relpath(item_path, project_path),
                                'size': os.path.getsize(item_path),
                                'file_type': self.get_file_type(item)
                            }
            
            # 开始处理根目录
            root = {'type': 'directory', 'items': {}}
            process_dir(project_path, root['items'])
            
            return root
            
        except Exception as e:
            logger.error(f"获取文件结构失败: {str(e)}")
            return {'type': 'directory', 'items': {}, 'error': str(e)}
    
    def _collect_supported_files(self, project_path: str, file_list: List[str]) -> None:
        """收集所有支持的文件
        
        Args:
            project_path: 项目路径
            file_list: 文件列表，用于存储收集到的文件路径
        """
        try:
            for root, dirs, files in os.walk(project_path):
                # 跳过忽略的目录
                dirs[:] = [d for d in dirs if d not in self.IGNORED_DIRS]
                
                # 添加支持的文件
                for filename in files:
                    if self.is_supported_file(filename):
                        file_path = os.path.join(root, filename)
                        # 检查文件大小
                        if self.is_file_size_acceptable(file_path):
                            file_list.append(file_path)
                        
        except Exception as e:
            logger.error(f"收集支持的文件失败: {str(e)}")
    
    def get_project_summary(self, analysis_result: Dict[str, Any]) -> str:
        """生成项目分析摘要
        
        Args:
            analysis_result: 分析结果
            
        Returns:
            str: 项目摘要
        """
        if analysis_result.get("status") != "success":
            return f"项目分析失败: {analysis_result.get('error')}"
        
        project_path = analysis_result.get("project_path", "未知路径")
        total_files = analysis_result.get("total_files", 0)
        processed_files = analysis_result.get("processed_files", 0)
        sampled_files = analysis_result.get("sampled_files", processed_files)
        sample_rate = analysis_result.get("sample_rate", 1.0)
        supported_files = analysis_result.get("supported_files", 0)
        error_files = analysis_result.get("error_files", 0)
        processing_time = analysis_result.get("processing_time", 0)
        from_cache = analysis_result.get("from_cache", False)
        
        # 统计文件类型
        file_types = {}
        file_structure = analysis_result.get("file_structure", {}).get("items", {})
        
        def count_file_types(structure: Dict[str, Any]):
            """递归统计文件类型"""
            for item_name, item_info in structure.items():
                if item_info.get("type") == "file":
                    file_type = item_info.get("file_type", "unknown")
                    file_types[file_type] = file_types.get(file_type, 0) + 1
                elif item_info.get("type") == "directory" and "items" in item_info:
                    count_file_types(item_info["items"])
        
        count_file_types(file_structure)
        
        # 格式化文件类型统计
        file_types_summary = []
        for file_type, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
            file_types_summary.append(f"{file_type}: {count}个文件")
        
        # 生成摘要
        cache_info = "（使用缓存）" if from_cache else ""
        sample_info = f"采样率: {sample_rate:.2%}, " if sample_rate < 1.0 else ""
        
        summary = f"""
项目位置: {project_path} {cache_info}
总文件数: {total_files}
{sample_info}处理文件数: {processed_files}
成功解析: {supported_files}
解析失败: {error_files}
处理耗时: {processing_time:.2f}秒

文件类型统计:
{', '.join(file_types_summary[:10])}
{'' if len(file_types_summary) <= 10 else '...等更多文件类型'}
        """
        
        return summary.strip() 