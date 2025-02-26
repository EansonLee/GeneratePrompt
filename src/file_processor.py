import os
import zipfile
import shutil
import py7zr
import patoolib
from pathlib import Path
from typing import List, Dict, Any
import logging
from src.utils.vector_store import VectorStore
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime

logger = logging.getLogger(__name__)

def find_7zip_path() -> str:
    """查找7-Zip可执行文件的路径，支持Windows和Linux系统"""
    # 检查环境变量中的路径
    if 'SEVENZIP_PATH' in os.environ:
        path = os.environ['SEVENZIP_PATH']
        if os.path.exists(path):
            return path
    
    # 根据操作系统选择可能的路径
    if os.name == 'nt':  # Windows系统
        possible_paths = [
            r"C:\Program Files\7-Zip\7z.exe",
            r"C:\Program Files (x86)\7-Zip\7z.exe",
            r"C:\Users\Administrator\AppData\Local\Programs\7-Zip\7z.exe",
            r"C:\Users\Administrator\scoop\apps\7zip\current\7z.exe",
        ]
    else:  # Linux/Unix系统
        possible_paths = [
            "/usr/bin/7z",
            "/usr/local/bin/7z",
            "/usr/bin/7za",
            "/usr/local/bin/7za"
        ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    raise FileNotFoundError("找不到7-Zip可执行文件，请确保已正确安装7-Zip。\n" +
                          "Windows用户请使用: winget install 7zip.7zip\n" +
                          "Linux用户请使用: sudo apt-get install p7zip-full p7zip-rar")

class FileProcessor:
    """文件处理器类"""
    
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
        # 配置文件
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.env': 'env',
        # 样式文件
        '.css': 'css',
        '.scss': 'scss',
        '.less': 'less',
        # 压缩文件
        '.zip': 'archive',
        '.rar': 'archive',
        '.7z': 'archive'
    }

    SPECIAL_FILES = {
        'package.json': 'npm_config',
        'tsconfig.json': 'typescript_config',
        'README.md': 'documentation',
        '.env': 'environment',
        'vite.config.ts': 'vite_config',
        'vite.config.js': 'vite_config',
        'next.config.js': 'next_config',
        'next.config.ts': 'next_config'
    }

    def __init__(self, upload_dir: Path, vector_store: VectorStore):
        """初始化文件处理器
        
        Args:
            upload_dir: 上传文件目录
            vector_store: 向量存储实例
        """
        self.upload_dir = Path(upload_dir)
        self.temp_dir = self.upload_dir / "temp"
        self.vector_store = vector_store
        
        # 确保目录存在
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找7-Zip路径
        try:
            self.seven_zip_path = find_7zip_path()
            logger.info(f"找到7-Zip路径: {self.seven_zip_path}")
        except FileNotFoundError as e:
            logger.warning(f"7-Zip未找到: {str(e)}")
            self.seven_zip_path = None
            
        logger.info(f"文件处理器初始化完成，上传目录: {self.upload_dir}")

    def is_supported_file(self, filename: str) -> bool:
        """检查文件是否支持
        
        Args:
            filename: 文件名
            
        Returns:
            bool: 是否支持该文件类型
        """
        ext = Path(filename).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS

    def get_file_type(self, filename: str) -> str:
        """获取文件类型
        
        Args:
            filename: 文件名
            
        Returns:
            str: 文件类型
        """
        ext = Path(filename).suffix.lower()
        # 直接使用扩展名判断，不再依赖外部工具
        if ext in self.SUPPORTED_EXTENSIONS:
            return self.SUPPORTED_EXTENSIONS[ext]
        return 'unknown'

    async def process_file(self, file_path: Path, is_directory: bool = False) -> Dict[str, Any]:
        """处理上传的文件
        
        Args:
            file_path: 文件路径
            is_directory: 是否作为目录处理
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            logger.info(f"开始处理文件: {file_path}")
            logger.info(f"文件大小: {file_path.stat().st_size / 1024:.2f} KB")
            logger.info(f"处理模式: {'目录模式' if is_directory else '文件模式'}")
            
            # 获取文件类型
            file_type = self.get_file_type(file_path.name)
            logger.info(f"文件类型: {file_type}")
            
            if not self.is_supported_file(file_path.name):
                error_msg = f"不支持的文件类型: {Path(file_path).suffix}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 如果是压缩文件且需要作为目录处理
            if file_type == 'archive' and is_directory:
                logger.info("开始处理压缩文件...")
                return await self.process_archive(file_path)
            
            # 读取文件内容
            logger.info("读取文件内容...")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"文件内容长度: {len(content)} 字符")
            
            # 提取关键信息
            logger.info("提取文件关键信息...")
            extracted_info = self._extract_key_information(content, file_path.name)
            logger.info(f"提取的信息: {extracted_info}")
            
            # 准备元数据
            metadata = {
                "file_name": file_path.name,
                "file_type": file_type,
                "file_path": str(file_path),
                "timestamp": datetime.now().isoformat(),
                **extracted_info
            }
            logger.info(f"完整元数据: {metadata}")
            
            # 等待向量数据库就绪
            if not await self.vector_store.wait_until_ready():
                error = self.vector_store.get_initialization_error()
                raise ValueError(f"向量数据库未就绪: {error}")
            
            # 添加到向量数据库
            logger.info("添加到向量数据库...")
            chunks = self.vector_store.add_texts([content], [metadata])
            logger.info(f"添加完成，生成了 {len(chunks) if chunks else 0} 个文本块")
            
            # 验证插入
            if self.vector_store.verify_insertion(content):
                logger.info("文件内容成功添加到向量数据库")
                return {
                    "status": "success",
                    "file_type": file_type,
                    "chunks": len(chunks) if chunks else 0,
                    "extracted_info": extracted_info
                }
            else:
                error_msg = "文件内容可能未成功添加到向量数据库"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "file_type": file_type,
                    "message": error_msg
                }
            
        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def _extract_key_information(self, content: str, file_name: str = None) -> Dict[str, Any]:
        """从文件内容中提取关键信息
        
        Args:
            content: 文件内容
            file_name: 文件名
            
        Returns:
            Dict[str, Any]: 提取的关键信息
        """
        info = {}
        
        # 处理特殊文件
        if file_name:
            base_name = Path(file_name).name
            if base_name in self.SPECIAL_FILES:
                special_info = self._extract_from_special_file(content, base_name)
                if special_info:
                    info.update(special_info)
                    return info

        # 定义关键字模式
        patterns = {
            "frontend_tech": [
                r"前端技术[：:]\s*(.*?)(?:\n|$)",
                r"frontend[：:]\s*(.*?)(?:\n|$)",
                r"(?:react|vue|angular).*版本[：:]\s*(.*?)(?:\n|$)",
                r'"react":\s*"([^"]+)"',
                r'"@types/react":\s*"([^"]+)"',
                r'"next":\s*"([^"]+)"',
                r'"vue":\s*"([^"]+)"'
            ],
            "backend_tech": [
                r"后端技术[：:]\s*(.*?)(?:\n|$)",
                r"backend[：:]\s*(.*?)(?:\n|$)",
                r"(?:python|java|node).*版本[：:]\s*(.*?)(?:\n|$)",
                r'"express":\s*"([^"]+)"',
                r'"koa":\s*"([^"]+)"',
                r'"fastapi":\s*"([^"]+)"'
            ],
            "database_tech": [
                r"数据库[：:]\s*(.*?)(?:\n|$)",
                r"database[：:]\s*(.*?)(?:\n|$)",
                r"(?:mysql|postgresql|mongodb)[：:]\s*(.*?)(?:\n|$)",
                r'"mongoose":\s*"([^"]+)"',
                r'"sequelize":\s*"([^"]+)"',
                r'"prisma":\s*"([^"]+)"'
            ],
            "api_design": [
                r"API设计[：:]\s*(.*?)(?:\n|$)",
                r"接口设计[：:]\s*(.*?)(?:\n|$)",
                r"(?:rest|graphql)[：:]\s*(.*?)(?:\n|$)",
                r'"axios":\s*"([^"]+)"',
                r'"graphql":\s*"([^"]+)"',
                r'"apollo":\s*"([^"]+)"'
            ],
            "navigation": [
                r"导航[：:]\s*(.*?)(?:\n|$)",
                r"navigation[：:]\s*(.*?)(?:\n|$)",
                r"菜单设计[：:]\s*(.*?)(?:\n|$)",
                r'"react-router":\s*"([^"]+)"',
                r'"react-router-dom":\s*"([^"]+)"',
                r'"@reach/router":\s*"([^"]+)"'
            ],
            "responsive": [
                r"响应式[：:]\s*(.*?)(?:\n|$)",
                r"responsive[：:]\s*(.*?)(?:\n|$)",
                r"自适应[：:]\s*(.*?)(?:\n|$)",
                r'"tailwindcss":\s*"([^"]+)"',
                r'"@material-ui":\s*"([^"]+)"',
                r'"antd":\s*"([^"]+)"'
            ],
            "state_management": [
                r"状态管理[：:]\s*(.*?)(?:\n|$)",
                r"state.*management[：:]\s*(.*?)(?:\n|$)",
                r"(?:redux|mobx|vuex)[：:]\s*(.*?)(?:\n|$)",
                r'"redux":\s*"([^"]+)"',
                r'"@reduxjs/toolkit":\s*"([^"]+)"',
                r'"mobx":\s*"([^"]+)"',
                r'"recoil":\s*"([^"]+)"'
            ],
            "data_flow": [
                r"数据流[：:]\s*(.*?)(?:\n|$)",
                r"data.*flow[：:]\s*(.*?)(?:\n|$)",
                r"数据流转[：:]\s*(.*?)(?:\n|$)",
                r'"react-query":\s*"([^"]+)"',
                r'"swr":\s*"([^"]+)"',
                r'"apollo-client":\s*"([^"]+)"'
            ],
            "component_design": [
                r"组件设计[：:]\s*(.*?)(?:\n|$)",
                r"component[：:]\s*(.*?)(?:\n|$)",
                r"组件架构[：:]\s*(.*?)(?:\n|$)",
                r'"styled-components":\s*"([^"]+)"',
                r'"@emotion/react":\s*"([^"]+)"',
                r'"sass":\s*"([^"]+)"'
            ]
        }
        
        import re
        # 遍历所有模式并提取信息
        for key, patterns_list in patterns.items():
            for pattern in patterns_list:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    if key not in info:
                        info[key] = []
                    info[key].append(matches[0].strip())
                    
        logger.info(f"从文件中提取到的信息: {info}")
        return info

    def _extract_style_information(self, content: str, style_type: str) -> Dict[str, Any]:
        """从样式文件中提取信息
        
        Args:
            content: 文件内容
            style_type: 样式文件类型
            
        Returns:
            Dict[str, Any]: 提取的样式信息
        """
        info = {
            "style_type": style_type,
            "style_features": []
        }
        
        # 提取样式特征
        import re
        
        # CSS/SCSS特征
        features = {
            "responsive": [
                r"@media\s+[^{]+{",  # 媒体查询
                r"@include\s+respond-to\(",  # SCSS响应式混入
                r"@include\s+breakpoint\("
            ],
            "layout": [
                r"display:\s*flex",  # Flex布局
                r"display:\s*grid",  # Grid布局
                r"position:\s*absolute|relative|fixed",  # 定位
                r"float:\s*left|right"  # 浮动
            ],
            "animation": [
                r"@keyframes\s+\w+",  # 关键帧动画
                r"animation:",  # 动画属性
                r"transition:"  # 过渡效果
            ],
            "theme": [
                r"--[\w-]+:",  # CSS变量
                r"\$[\w-]+:",  # SCSS变量
                r"@mixin\s+[\w-]+",  # SCSS混入
                r"@extend\s+[\w-]+"  # SCSS继承
            ]
        }
        
        for category, patterns in features.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    info["style_features"].append(category)
                    break
        
        # 移除重复项
        info["style_features"] = list(set(info["style_features"]))
        
        # 提取组件样式信息
        if style_type in ['scss', 'less']:
            component_patterns = [
                r"\.[\w-]+\s*{",  # 类选择器
                r"#[\w-]+\s*{",  # ID选择器
                r"\[[\w-]+\]",  # 属性选择器
                r":[\w-]+\s*{",  # 伪类选择器
                r"::[\w-]+\s*{"  # 伪元素选择器
            ]
            
            component_styles = []
            for pattern in component_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    component_styles.extend(matches)
            
            if component_styles:
                info["component_styles"] = list(set(component_styles))
        
        logger.info(f"提取的样式信息: {info}")
        return info

    def _extract_from_special_file(self, content: str, file_name: str) -> Dict[str, Any]:
        """从特殊文件中提取信息
        
        Args:
            content: 文件内容
            file_name: 文件名
            
        Returns:
            Dict[str, Any]: 提取的信息
        """
        info = {}
        
        try:
            if file_name == 'package.json':
                import json
                package_data = json.loads(content)
                
                # 提取项目基本信息
                info['project_name'] = package_data.get('name', '')
                info['project_version'] = package_data.get('version', '')
                info['project_description'] = package_data.get('description', '')
                
                # 提取依赖信息
                all_deps = {}
                if 'dependencies' in package_data:
                    all_deps.update(package_data['dependencies'])
                if 'devDependencies' in package_data:
                    all_deps.update(package_data['devDependencies'])
                
                # 分类依赖
                tech_info = {
                    'frontend_tech': [],
                    'state_management': [],
                    'ui_framework': [],
                    'build_tools': [],
                    'testing': [],
                    'style_tools': []  # 新增样式工具分类
                }
                
                # 前端框架
                if 'react' in all_deps:
                    tech_info['frontend_tech'].append(f"React {all_deps['react']}")
                if 'next' in all_deps:
                    tech_info['frontend_tech'].append(f"Next.js {all_deps['next']}")
                
                # 状态管理
                for pkg in ['redux', '@reduxjs/toolkit', 'mobx', 'recoil', 'zustand']:
                    if pkg in all_deps:
                        tech_info['state_management'].append(f"{pkg} {all_deps[pkg]}")
                
                # UI框架
                for pkg in ['antd', '@material-ui/core', '@chakra-ui/react', '@mui/material']:
                    if pkg in all_deps:
                        tech_info['ui_framework'].append(f"{pkg} {all_deps[pkg]}")
                
                # 构建工具
                for pkg in ['webpack', 'vite', 'parcel', '@vitejs/plugin-react']:
                    if pkg in all_deps:
                        tech_info['build_tools'].append(f"{pkg} {all_deps[pkg]}")
                
                # 测试工具
                for pkg in ['jest', '@testing-library/react', 'cypress', 'vitest']:
                    if pkg in all_deps:
                        tech_info['testing'].append(f"{pkg} {all_deps[pkg]}")
                
                # 样式工具
                for pkg in ['sass', 'less', 'styled-components', '@emotion/react', 'tailwindcss']:
                    if pkg in all_deps:
                        tech_info['style_tools'].append(f"{pkg} {all_deps[pkg]}")
                
                info.update(tech_info)
                
            elif file_name == 'tsconfig.json':
                import json
                tsconfig = json.loads(content)
                info['typescript_config'] = tsconfig.get('compilerOptions', {})
                
            elif file_name == 'README.md':
                # 提取项目描述和技术栈信息
                import re
                tech_stack = re.findall(r'## Tech Stack\n(.*?)(?=\n#|$)', content, re.DOTALL)
                if tech_stack:
                    info['tech_stack'] = tech_stack[0].strip()
                    
                description = re.findall(r'## Description\n(.*?)(?=\n#|$)', content, re.DOTALL)
                if description:
                    info['project_description'] = description[0].strip()
                    
        except Exception as e:
            logger.error(f"从特殊文件提取信息失败 {file_name}: {str(e)}")
            
        return info

    async def process_archive(self, archive_path: Path) -> Dict[str, Any]:
        """处理压缩文件
        
        Args:
            archive_path: 压缩文件路径
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            # 创建临时解压目录
            extract_dir = self.temp_dir / archive_path.stem
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # 根据文件扩展名选择解压方法
                ext = archive_path.suffix.lower()
                if ext == '.zip':
                    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                elif ext == '.7z':
                    with py7zr.SevenZipFile(archive_path, 'r') as sz:
                        sz.extractall(extract_dir)
                elif ext == '.rar':
                    try:
                        if self.seven_zip_path:
                            # 尝试使用7z解压RAR文件
                            import subprocess
                            logger.info(f"使用7-Zip解压: {self.seven_zip_path}")
                            result = subprocess.run(
                                [self.seven_zip_path, 'x', str(archive_path), f'-o{str(extract_dir)}'],
                                capture_output=True,
                                text=True
                            )
                            if result.returncode != 0:
                                # 如果7z失败，尝试使用patool
                                logger.warning(f"使用7z解压失败: {result.stderr}")
                                logger.info("尝试使用patool解压...")
                                patoolib.extract_archive(str(archive_path), outdir=str(extract_dir), interactive=False)
                        else:
                            # 直接使用patool
                            logger.info("未找到7-Zip，使用patool解压...")
                            patoolib.extract_archive(str(archive_path), outdir=str(extract_dir), interactive=False)
                    except Exception as e:
                        raise Exception(f"解压失败: {str(e)}")
                else:
                    raise ValueError(f"暂不支持的压缩格式: {ext}")
                    
                # 处理解压后的文件
                total_chunks = 0
                processed_files = 0
                failed_files = 0
                
                for file_path in extract_dir.rglob("*"):
                    if file_path.is_file() and self.is_supported_file(file_path.name):
                        try:
                            result = await self.process_file(file_path)
                            if result.get("status") == "success":
                                total_chunks += result.get("chunks", 0)
                                processed_files += 1
                            else:
                                failed_files += 1
                        except Exception as e:
                            logger.warning(f"处理解压文件失败 {file_path}: {str(e)}")
                            failed_files += 1
                            
                return {
                    "status": "success",
                    "file_type": "archive",
                    "processed_files": processed_files,
                    "failed_files": failed_files,
                    "total_chunks": total_chunks
                }
                
            finally:
                # 清理临时目录
                if extract_dir.exists():
                    shutil.rmtree(extract_dir)
            
        except Exception as e:
            logger.error(f"处理压缩文件失败 {archive_path}: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def save_upload_file(self, file: Any) -> Path:
        """保存上传的文件
        
        Args:
            file: 上传的文件对象
            
        Returns:
            Path: 保存的文件路径
        """
        try:
            file_path = self.upload_dir / file.filename
            
            # 写入文件
            with file_path.open("wb") as buffer:
                content = await file.read()
                buffer.write(content)
                
            return file_path
            
        except Exception as e:
            logger.error(f"保存上传文件失败: {str(e)}")
            raise 