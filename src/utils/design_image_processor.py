import logging
import os
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import uuid
from datetime import datetime
import base64
from PIL import Image
import io
import numpy as np
import json
import hashlib
import time
import asyncio
import copy
import re
import tempfile
import traceback

from config.config import settings
from src.utils.vector_store import VectorStore
from openai import OpenAI
from langchain_openai.chat_models import ChatOpenAI

logger = logging.getLogger(__name__)

class DesignImageProcessor:
    """设计图处理器，用于处理上传的设计图并生成向量"""
    
    # 添加缓存相关的类变量
    _analysis_cache = {}  # 内存缓存：图像哈希 -> 分析结果
    _analysis_cache_file = None  # 缓存文件路径
    _cache_expiry = 30 * 24 * 60 * 60  # 缓存过期时间（30天）
    _max_cache_size = 100  # 最大缓存条目数
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        upload_dir: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        """初始化设计图处理器
        
        Args:
            vector_store: 向量存储对象
            upload_dir: 上传目录
            temperature: 温度
            max_tokens: 最大Token数
        """
        # 记录模型配置信息
        logger.info("初始化设计图处理器")
        logger.info(f"环境配置检查 - OPENAI_API_KEY: {'已设置' if settings.OPENAI_API_KEY else '未设置'}")
        logger.info(f"环境配置检查 - VISION_MODEL: {settings.VISION_MODEL or '未设置'}")
        logger.info(f"环境配置检查 - OPENAI_BASE_URL: {settings.OPENAI_BASE_URL or '未设置'}")
        
        # 初始化基本属性
        self.vector_store = vector_store
        
        # 确保上传目录是Path对象并且存在
        if upload_dir:
            self.upload_dir = Path(upload_dir)
        else:
            self.upload_dir = Path(settings.UPLOAD_DIR)
        
        # 确保上传目录存在
        try:
            self.upload_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"已确保上传目录存在: {self.upload_dir}")
        except Exception as e:
            logger.error(f"创建上传目录失败: {self.upload_dir}, 错误: {str(e)}")
            # 尝试使用临时目录
            import tempfile
            self.upload_dir = Path(tempfile.gettempdir()) / "design_images"
            self.upload_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"使用临时目录作为上传目录: {self.upload_dir}")
        
        self.temperature = temperature if temperature is not None else settings.VISION_MODEL_CONFIG["temperature"]
        self.max_tokens = max_tokens if max_tokens is not None else settings.VISION_MODEL_CONFIG["max_tokens"]
        
        # 初始化OpenAI客户端用于图像识别
        try:
            if not settings.OPENAI_API_KEY:
                raise ValueError("OpenAI API密钥未设置")
            
            self.vision_model = OpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL
            )
            logger.info(f"OpenAI客户端初始化成功，使用模型: {settings.VISION_MODEL}")
        except Exception as e:
            logger.error(f"OpenAI客户端初始化失败: {str(e)}")
            self.vision_model = None
        
        # 初始化缓存文件路径
        if DesignImageProcessor._analysis_cache_file is None:
            cache_dir = Path(settings.DATA_DIR)
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                DesignImageProcessor._analysis_cache_file = cache_dir / "design_analysis_cache.json"
                logger.info(f"设置分析缓存文件路径: {DesignImageProcessor._analysis_cache_file}")
            except Exception as e:
                logger.error(f"创建缓存目录失败: {str(e)}")
                # 使用临时目录
                import tempfile
                temp_dir = Path(tempfile.gettempdir())
                DesignImageProcessor._analysis_cache_file = temp_dir / "design_analysis_cache.json"
                logger.warning(f"使用临时目录作为缓存文件路径: {DesignImageProcessor._analysis_cache_file}")
        
        # 加载缓存
        self._load_analysis_cache()
        
        logger.info("设计图处理器初始化完成")
    
    def _load_analysis_cache(self):
        """加载设计图分析结果缓存"""
        try:
            if DesignImageProcessor._analysis_cache_file.exists():
                with open(DesignImageProcessor._analysis_cache_file, "r", encoding="utf-8") as f:
                    DesignImageProcessor._analysis_cache = json.load(f)
                logger.info(f"已加载设计图分析结果缓存: {len(DesignImageProcessor._analysis_cache)}条记录")
            else:
                logger.info("设计图分析结果缓存文件不存在，将创建新缓存")
                DesignImageProcessor._analysis_cache = {}
        except Exception as e:
            logger.error(f"加载设计图分析结果缓存失败: {str(e)}")
            DesignImageProcessor._analysis_cache = {}
    
    def _save_analysis_cache(self):
        """保存设计图分析结果缓存"""
        try:
            # 确保缓存文件的父目录存在
            if DesignImageProcessor._analysis_cache_file.parent:
                DesignImageProcessor._analysis_cache_file.parent.mkdir(parents=True, exist_ok=True)
                
            # 限制缓存大小
            if len(DesignImageProcessor._analysis_cache) > DesignImageProcessor._max_cache_size:
                # 按时间戳排序，保留最新的条目
                sorted_cache = sorted(
                    DesignImageProcessor._analysis_cache.items(),
                    key=lambda x: x[1].get('timestamp', 0)
                )
                # 保留最新的条目
                DesignImageProcessor._analysis_cache = dict(sorted_cache[-DesignImageProcessor._max_cache_size:])
                logger.info(f"裁剪后缓存条目数: {len(DesignImageProcessor._analysis_cache)}")
            
            # 保存缓存
            with open(DesignImageProcessor._analysis_cache_file, "w", encoding="utf-8") as f:
                json.dump(DesignImageProcessor._analysis_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"已保存分析缓存到: {DesignImageProcessor._analysis_cache_file}")
        except Exception as e:
            logger.error(f"保存设计图分析缓存失败: {str(e)}")
    
    def is_supported_image(self, file_name: str) -> bool:
        """检查文件是否为支持的图片格式
        
        Args:
            file_name: 文件名
            
        Returns:
            bool: 如果文件是支持的图片格式则返回True
        """
        supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
        ext = os.path.splitext(file_name)[1].lower()
        return ext in supported_formats
    
    def _compute_image_hash(self, image_data: bytes) -> str:
        """计算图片哈希值
        
        Args:
            image_data: 图片数据
            
        Returns:
            str: 图片哈希值
        """
        return hashlib.sha256(image_data).hexdigest()
    
    def _get_design_image_path(self, design_id: str) -> Optional[Path]:
        """获取设计图片路径
        
        Args:
            design_id: 设计ID
            
        Returns:
            设计图片的路径或None（如果找不到）
        """
        if not design_id:
            logger.error("设计ID不能为空")
            return None
        
        upload_dir = self.upload_dir
        
        # 确保上传目录存在
        if not upload_dir.exists():
            try:
                upload_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"已创建上传目录: {upload_dir}")
            except Exception as e:
                logger.error(f"创建上传目录失败: {upload_dir}, 错误: {str(e)}")
                return None
        
        # 尝试找到对应的图片文件
        logger.info(f"正在查找设计图: 设计ID={design_id}, 查找目录={upload_dir}")
        
        # 支持的图像扩展名
        supported_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.gif']
        
        # 尝试直接匹配设计ID
        for ext in supported_extensions:
            file_path = upload_dir / f"{design_id}{ext}"
            if file_path.exists():
                logger.info(f"找到设计图: {file_path}")
                return file_path
        
        # 尝试查找包含设计ID的文件
        for file_path in upload_dir.glob("*"):
            if file_path.is_file() and any(file_path.name.endswith(ext) for ext in supported_extensions):
                # 检查文件名是否包含设计ID
                if design_id in file_path.stem:
                    logger.info(f"找到匹配的设计图: {file_path}")
                    return file_path
        
        # 尝试找到最近上传的图片文件（倒序排列，最新的在前面）
        image_files = [f for f in upload_dir.glob("*") 
                      if f.is_file() and any(f.name.endswith(ext) for ext in supported_extensions)]
        
        if image_files:
            # 按修改时间排序，最新的在前面
            sorted_files = sorted(image_files, key=lambda x: x.stat().st_mtime, reverse=True)
            newest_file = sorted_files[0]
            logger.warning(f"未找到与设计ID匹配的图片，使用最新上传的图片: {newest_file}")
            return newest_file
        
        logger.error(f"未找到任何设计图，设计ID: {design_id}")
        return None
    
    def _prune_cache_if_needed(self, max_entries: int = 100):
        """如果缓存大小超过限制，清理最旧的缓存条目
        
        Args:
            max_entries: 缓存条目的最大数量
        """
        if len(DesignImageProcessor._analysis_cache) <= max_entries:
            return
            
        # 按时间戳排序，保留最新的条目
        sorted_entries = sorted(
            DesignImageProcessor._analysis_cache.items(),
            key=lambda x: x[1].get("timestamp", 0),
            reverse=True
        )
        
        # 保留最新的条目
        DesignImageProcessor._analysis_cache = {
            k: v for k, v in sorted_entries[:max_entries]
        }
        
        logger.info(f"缓存清理完成，保留了 {len(DesignImageProcessor._analysis_cache)} 个条目")
    
    async def process_image(
        self, 
        file_content: bytes,
        file_name: str,
        tech_stack: str = None
    ) -> Dict[str, Any]:
        """处理设计图
        
        Args:
            file_content: 文件内容
            file_name: 文件名
            tech_stack: 技术栈
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            # 1. 生成唯一ID
            image_id = str(uuid.uuid4())
            logger.info(f"开始处理设计图: {file_name}, ID: {image_id}")
            
            # 2. 保存图片文件
            # 使用原始文件扩展名
            ext = os.path.splitext(file_name)[1].lower()
            if not ext or ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
                ext = '.png'  # 默认使用png格式
                
            unique_filename = f"{image_id}{ext}"
            file_path = os.path.join(self.upload_dir, unique_filename)
            
            # 确保上传目录存在
            upload_dir = Path(self.upload_dir)
            upload_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"确保上传目录存在: {upload_dir}")
            
            # Windows路径兼容性处理
            file_path = str(Path(file_path))
            
            # 保存文件
            with open(file_path, "wb") as f:
                f.write(file_content)
            logger.info(f"设计图已保存到: {file_path}")
            
            # 验证图片文件是否可以正常打开
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(file_content))
                logger.info(f"图片尺寸: {img.size}, 格式: {img.format}")
                
                # 保存为PNG格式确保兼容性
                normalized_path = os.path.join(self.upload_dir, f"{image_id}_normalized.png")
                normalized_path = str(Path(normalized_path))
                img.save(normalized_path)
                logger.info(f"图片已标准化保存为PNG格式: {normalized_path}")
                
                # 使用标准化的路径
                file_path = normalized_path
            except Exception as e:
                logger.warning(f"图片验证或标准化失败，将使用原始文件: {str(e)}")
            
            # 3. 计算图像哈希，用于缓存查找
            image_hash = self._compute_image_hash(file_content)
            logger.info(f"图像哈希值: {image_hash}")
            
            # 4. 检查缓存中是否有分析结果
            analysis_text = None
            cache_hit = False
            
            if image_hash in DesignImageProcessor._analysis_cache:
                cache_entry = DesignImageProcessor._analysis_cache[image_hash]
                # 更新时间戳
                cache_entry["timestamp"] = time.time()
                analysis_text = cache_entry.get("analysis_text")
                if analysis_text:
                    logger.info(f"使用缓存的分析结果: {image_hash}")
                    cache_hit = True
            
            # 5. 如果缓存中没有，则调用视觉模型分析
            if not analysis_text:
                try:
                    # 准备图片数据
                    img_format = ext.lstrip('.') or 'png'
                    
                    # 准备API调用
                    logger.info(f"开始调用vision model进行图片分析: {file_name}")
                    
                    # 减少分析延迟的简化分析
                    response = self.vision_model.chat.completions.create(
                        model=settings.VISION_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "你是一个高效的UI/UX设计分析专家，需要快速提供设计图的要点分析。"
                                    "专注于最重要的UI元素、布局结构和设计模式，不需要详细描述每个细节。"
                                    "分析应当简洁、准确，并突出对开发人员最有用的信息。"
                                )
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": (
                                            f"请简洁分析这张设计图，重点关注：\n"
                                            f"1. 主要布局结构\n"
                                            f"2. 关键UI组件\n"
                                            f"3. 核心色彩主题\n"
                                            f"4. 主要交互元素\n\n"
                                            f"请提供精简的分析，不超过300字。"
                                            f"如果涉及技术栈 {tech_stack or '(未指定)'} 的特性，简要提及。"
                                        )
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/{img_format};base64,{base64.b64encode(file_content).decode('utf-8')}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=1000,  # 减少token数量加快响应
                        temperature=0.3  # 使用较低温度获得更确定的结果
                    )
                    logger.info(f"Vision model ({settings.VISION_MODEL}) 调用成功: {file_name}")
                    
                    # 提取分析文本
                    analysis_text = response.choices[0].message.content
                    
                    if not analysis_text:
                        logger.error(f"分析结果为空: {file_name}")
                        analysis_text = f"设计图文件名: {file_name}\n技术栈: {tech_stack}\n上传时间: {datetime.now().isoformat()}\n注意：自动分析返回结果为空，请检查图片是否有效。"
                    elif len(analysis_text.strip()) < 100:
                        logger.warning(f"分析结果过短 ({len(analysis_text.strip())}字符): {file_name}")
                        analysis_text += f"\n\n附加信息: 文件名: {file_name}, 技术栈: {tech_stack}, 上传时间: {datetime.now().isoformat()}"
                    
                    logger.info(f"成功获取分析结果 ({len(analysis_text.strip())}字符): {file_name}")
                        
                except Exception as e:
                    logger.error(f"图片分析失败: {str(e)}")
                    analysis_text = f"设计图文件名: {file_name}\n技术栈: {tech_stack}\n上传时间: {datetime.now().isoformat()}\n分析失败原因: {str(e)}"
            
            # 6. 保存分析结果到缓存
            if not cache_hit and analysis_text:
                DesignImageProcessor._analysis_cache[image_hash] = {
                    "analysis_text": analysis_text,
                    "tech_stack": tech_stack,
                    "timestamp": time.time(),
                    "file_name": file_name
                }
                self._save_analysis_cache()
                logger.info(f"已将分析结果添加到缓存: {image_hash}")
            
            # 7. 将分析结果添加到向量存储
            vector_store_success = True
            if self.vector_store:
                try:
                    metadata = {
                        "id": image_id,
                        "file_name": file_name,
                        "tech_stack": tech_stack,
                        "timestamp": datetime.now().isoformat(),
                        "type": "design_analysis"
                    }
                    
                    # 添加到向量存储
                    doc_ids = await self.vector_store.add_content(
                        content=analysis_text,
                        content_type="designs",
                        metadata=metadata
                    )
                    
                    if not doc_ids:
                        logger.warning(f"向量存储添加分析结果失败: {image_id}")
                        vector_store_success = False
                    else:
                        logger.info(f"已将分析结果添加到向量存储: {image_id}")
                except Exception as e:
                    logger.error(f"向量存储添加分析结果失败: {str(e)}")
                    vector_store_success = False
            
            # 8. 构建返回结果
            result = {
                "id": image_id,
                "file_name": file_name,
                "file_path": file_path,
                "file_size": len(file_content),
                "image_hash": image_hash,
                "analysis_text": analysis_text,
                "tech_stack": tech_stack,
                "timestamp": datetime.now().isoformat(),
                "vector_store_success": vector_store_success,
                "cache_hit": cache_hit
            }
            
            # 9. 自动清理缓存，避免内存占用过大
            self._prune_cache_if_needed(DesignImageProcessor._max_cache_size)
            
            logger.info(f"设计图处理完成: {image_id}")
            return result
            
        except Exception as e:
            logger.error(f"处理设计图失败: {str(e)}", exc_info=True)
            return {
                "id": str(uuid.uuid4()),
                "file_name": file_name,
                "file_path": "",
                "file_size": len(file_content) if file_content else 0,
                "error": str(e),
                "tech_stack": tech_stack,
                "timestamp": datetime.now().isoformat(),
                "vector_store_success": False,
                "cache_hit": False
            }
    
    async def analyze_design(self, image_path: str) -> Dict[str, Any]:
        """分析设计图
        
        Args:
            image_path: 图片路径
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            # 读取图片
            with open(image_path, "rb") as f:
                image_content = f.read()
            
            # 计算图片哈希值用于缓存
            image_hash = self._compute_image_hash(image_content)
            
            # 检查缓存
            if image_hash in self.analysis_cache:
                cache_data = self.analysis_cache[image_hash]
                logger.info(f"从缓存中获取设计图分析结果: {image_hash}")
                # 更新时间戳
                cache_data["timestamp"] = time.time()
                self.analysis_cache[image_hash] = cache_data
                self._save_analysis_cache()
                
                # 返回缓存的分析结果
                if "result" in cache_data and "analysis" in cache_data["result"]:
                    return cache_data["result"]["analysis"]
            
            # 编码图片为base64
            image_base64 = base64.b64encode(image_content).decode("utf-8")
            
            # 构建请求
            messages = [
                {
                    "role": "system",
                    "content": """你是一个专业的UI/UX设计分析师。你的任务是分析用户提供的UI设计图，并提供详细的分析报告。
请关注以下方面：
1. 设计风格和主题
2. 布局结构和组织
3. 颜色方案和视觉层次
4. 主要UI组件和元素
5. 交互设计元素
6. 可用性和用户体验考虑
7. 技术实现建议

请以JSON格式返回分析结果，包含以下字段：
- design_style: 设计风格描述
- layout: 布局结构描述
- color_scheme: 颜色方案描述
- ui_components: UI组件列表及描述
- interaction_elements: 交互元素描述
- usability: 可用性评估
- tech_implementation: 技术实现建议
"""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请分析这个UI设计图，并提供详细的分析报告。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }
            ]
            
            # 调用视觉模型
            start_time = time.time()
            logger.info(f"开始分析设计图: {image_path}")
            
            response = await self.vision_model.agenerate(messages=[messages])
            
            # 解析响应
            response_text = response.generations[0][0].text
            
            # 尝试提取JSON
            try:
                # 查找JSON部分
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response_text
                
                # 解析JSON
                analysis_result = json.loads(json_str)
            except Exception as e:
                logger.error(f"解析分析结果JSON失败: {str(e)}")
                # 如果解析失败，返回原始文本
                analysis_result = {"raw_analysis": response_text}
            
            end_time = time.time()
            logger.info(f"设计图分析完成，耗时: {end_time - start_time:.2f}秒")
            
            # 保存到缓存
            self.analysis_cache[image_hash] = {
                "result": {"analysis": analysis_result},
                "timestamp": time.time()
            }
            self._save_analysis_cache()
            
            return analysis_result
        except Exception as e:
            logger.error(f"分析设计图失败: {str(e)}")
            logger.exception(e)
            return {"error": str(e)}
    
    async def _add_to_vector_store(self, analysis_sections: Dict[str, str], full_metadata: Dict[str, Any], file_id: str):
        """异步添加到向量存储
        
        Args:
            analysis_sections: 分析结果各部分
            full_metadata: 完整元数据
            file_id: 文件ID
        """
        added_sections = 0
        failed_sections = 0
        
        try:
            # 为每个部分创建单独的向量
            for section_type, section_text in analysis_sections.items():
                try:
                    section_metadata = {
                        **full_metadata,
                        "section_type": section_type
                    }
                    
                    doc_ids = await self.vector_store.add_texts(
                        texts=[section_text],
                        metadatas=[section_metadata],
                        ids=[f"{file_id}_{section_type}"]
                    )
                    
                    if doc_ids and len(doc_ids) > 0:
                        logger.info(f"已添加{section_type}部分到向量存储")
                        added_sections += 1
                    else:
                        logger.warning(f"未能添加{section_type}部分到向量存储，未返回ID")
                        failed_sections += 1
                        
                except Exception as section_error:
                    logger.error(f"添加{section_type}部分到向量存储失败: {str(section_error)}")
                    failed_sections += 1
                    # 继续处理其他部分，不中断整个流程
                    
            # 记录添加结果统计
            if added_sections > 0:
                if failed_sections > 0:
                    logger.info(f"部分添加成功: {added_sections}个部分添加成功，{failed_sections}个部分失败")
                else:
                    logger.info(f"全部添加成功: {added_sections}个部分")
            else:
                logger.warning(f"所有{len(analysis_sections)}个部分添加失败")
                
        except Exception as e:
            logger.error(f"添加到向量存储过程中发生错误: {str(e)}")
            if settings.DEBUG:
                import traceback
                logger.error(f"详细错误信息: {traceback.format_exc()}")
            # 继续执行，不要因为向量存储失败而中断整个流程
    
    def _build_analysis_prompt(self, tech_stack: str) -> str:
        """构建分析提示
        
        Args:
            tech_stack: 技术栈
            
        Returns:
            str: 分析提示
        """
        return (
                    f"你是一个专业的UI设计分析专家。请详细分析这个{tech_stack}应用界面设计图，"
                    "并按以下分类进行分析：\n\n"
                    
                    "1. 布局结构分析：\n"
                    "- 主要布局类型（如ConstraintLayout、LinearLayout、RelativeLayout等）\n"
                    "- 具体布局层次和嵌套关系\n"
                    "- 各组件的位置和对齐方式（顶部对齐、居中等）\n"
                    "- 页面整体结构（顶部导航、内容区、底部栏等）\n"
                    "- 响应式布局特征\n\n"
                    
                    "2. UI组件详细清单：\n"
                    "- 导航组件（Toolbar、TabLayout、BottomNavigationView等）\n"
                    "- 列表/网格组件（RecyclerView、GridView等）\n"
                    "- 基础组件（Button、TextView、ImageView等）\n"
                    "- 自定义组件\n"
                    "- 组件的具体属性（大小、边距、内边距等）\n\n"
                    
                    "3. 视觉设计规范：\n"
                    "- 主色调和辅助色（提供具体的色值，如#FFFFFF）\n"
                    "- 字体规范（字体大小、行高、字重）\n"
                    "- 图标规范（尺寸、颜色、风格）\n"
                    "- 阴影效果（elevation值）\n"
                    "- 圆角值（cornerRadius）\n"
                    "- 内外边距值（具体的dp值）\n\n"
                    
                    "4. 交互设计细节：\n"
                    "- 可点击区域的范围和反馈效果\n"
                    "- 状态变化（normal、pressed、disabled等）\n"
                    "- 动画效果（转场、加载、反馈动画）\n"
                    "- 手势操作（滑动、拖拽、缩放等）\n"
                    "- 页面跳转和过渡方式\n\n"
                    
                    "5. 技术实现指导：\n"
                    "- 布局实现方案（xml布局结构示例）\n"
                    "- 自定义View的实现建议\n"
                    "- 性能优化点（布局层级优化、视图缓存等）\n"
                    "- 屏幕适配方案（不同尺寸、分辨率）\n"
                    "- 无障碍支持建议\n\n"
                    
                    "请提供非常具体的分析，包括：\n"
                    "1. 所有尺寸使用dp为单位\n"
                    "2. 所有颜色使用十六进制值\n"
                    "3. 布局层次使用缩进表示\n"
                    "4. 组件的具体位置关系\n"
                    "5. 完整的属性值列表\n"
                )
                
    def _generate_default_analysis(self, tech_stack: str, filename: str) -> str:
        """生成默认分析结果
        
        Args:
            tech_stack: 技术栈
            filename: 文件名
            
        Returns:
            str: 默认分析结果
        """
        return f"""基于{tech_stack}平台的界面分析：

1. 布局结构分析：
   - 请检查设计图文件是否正确上传
   - 当前无法获取具体的设计图分析结果
   - 文件名: {filename}

2. UI组件和样式：
   - 建议重新上传设计图
   - 确保图片格式正确（支持jpg、png、webp）
   - 图片大小不超过5MB

3. 颜色主题：
   - 暂时无法分析具体的颜色主题
   - 建议使用{tech_stack}标准设计规范

4. 交互设计：
   - 暂时无法分析具体的交互设计
   - 建议参考{tech_stack}官方交互指南

5. 技术实现指导：
   - 请确保网络连接正常
   - 检查API密钥配置
   - 如果问题持续，请联系技术支持"""
    
    def _split_analysis_sections(self, analysis_text: str) -> Dict[str, str]:
        """将分析文本拆分为不同的部分
        
        Args:
            analysis_text: 分析文本
            
        Returns:
            Dict[str, str]: 按类别分类的文本
        """
        sections = {
            "layout": "",      # 布局结构分析
            "components": "",   # UI组件详细清单
            "visual": "",      # 视觉设计规范
            "interaction": "", # 交互设计细节
            "technical": ""    # 技术实现指导
        }
        
        try:
            # 使用更精确的文本分割
            current_section = None
            lines = analysis_text.split("\n")
            section_markers = {
                "1. 布局结构分析": "layout",
                "2. UI组件详细清单": "components",
                "3. 视觉设计规范": "visual",
                "4. 交互设计细节": "interaction",
                "5. 技术实现指导": "technical"
            }
            
            for line in lines:
                # 检查是否是新的部分
                for marker, section in section_markers.items():
                    if marker in line:
                        current_section = section
                        break
                        
                # 如果是当前部分的内容，则添加到对应部分
                if current_section and line.strip():
                    # 跳过部分标题行
                    if not any(marker in line for marker in section_markers.keys()):
                        sections[current_section] += line + "\n"
            
            # 清理每个部分的文本
            for section in sections:
                # 移除多余的空行
                sections[section] = "\n".join(
                    line for line in sections[section].split("\n")
                    if line.strip()
                )
                
            # 验证所有部分都有内容
            empty_sections = [
                section for section, content in sections.items()
                if not content.strip()
            ]
            if empty_sections:
                logger.warning(f"以下部分没有内容: {', '.join(empty_sections)}")
                
            return sections
                    
        except Exception as e:
            logger.error(f"分割分析文本失败: {str(e)}")
            # 如果分割失败，将整个文本放在layout部分
            sections["layout"] = analysis_text
            return sections
    
    def get_image_base64(self, file_path: str) -> Optional[str]:
        """获取图片的base64编码
        
        Args:
            file_path: 图片路径
            
        Returns:
            Optional[str]: base64编码的图片
        """
        try:
            with open(file_path, "rb") as f:
                image_data = f.read()
                return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            logger.error(f"获取图片base64编码失败: {str(e)}")
            return None
    
    async def search_similar_designs(
        self, 
        tech_stack: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """搜索相似设计图
        
        Args:
            tech_stack: 技术栈
            limit: 返回结果数量限制
            
        Returns:
            List[Dict[str, Any]]: 相似设计图列表
        """
        try:
            # 构建查询
            query = f"技术栈: {tech_stack} 设计图"
            
            # 搜索向量数据库
            results = await self.vector_store.search_contexts(query=query, limit=limit)
            
            # 过滤结果，只返回设计图
            design_results = [
                result for result in results 
                if result.get("metadata", {}).get("type") == "design_image"
                and result.get("metadata", {}).get("tech_stack") == tech_stack
            ]
            
            return design_results
            
        except Exception as e:
            logger.error(f"搜索相似设计图失败: {str(e)}")
            return []

    async def analyze_design_image(self, image_path_or_base64: str, tech_stack: str) -> Dict[str, Any]:
        """分析设计图

        Args:
            image_path_or_base64: 设计图路径或base64数据
            tech_stack: 技术栈

        Returns:
            Dict[str, Any]: 分析结果
        """
        import tempfile
        import os
        import base64
        import hashlib
        from pathlib import Path
        
        start_time = time.time()
        logger.info(f"开始分析设计图: tech_stack={tech_stack}")
        
        # 准备分析图像
        temp_file = None
        local_image_path = None
        original_image_data = None
        image_hash = None
        
        try:
            # 检查缓存 - 先计算哈希
            is_base64_data = isinstance(image_path_or_base64, str) and image_path_or_base64.startswith('data:image')
            if is_base64_data:
                try:
                    # 保存原始base64数据
                    original_image_data = image_path_or_base64
                    
                    # 提取base64数据部分计算哈希值
                    header, b64_data = image_path_or_base64.split(',', 1)
                    image_hash = hashlib.sha256(b64_data.encode('utf-8')).hexdigest()
                except Exception as e:
                    logger.warning(f"计算base64哈希值出错: {str(e)}")
            else:
                # 如果是文件路径，计算文件哈希值
                if os.path.exists(image_path_or_base64):
                    try:
                        with open(image_path_or_base64, 'rb') as f:
                            file_data = f.read()
                        image_hash = hashlib.sha256(file_data).hexdigest()
                        
                        # 读取并转换为base64
                        encoded = base64.b64encode(file_data).decode('utf-8')
                        image_type = os.path.splitext(image_path_or_base64)[1][1:].lower()
                        if image_type not in ['png', 'jpg', 'jpeg', 'webp', 'gif']:
                            image_type = 'png'
                        original_image_data = f"data:image/{image_type};base64,{encoded}"
                    except Exception as e:
                        logger.warning(f"计算文件哈希值或转换base64时出错: {str(e)}")
            
            # 如果有哈希值和原始图像数据，尝试从缓存获取
            if image_hash and original_image_data:
                cache_key = f"{image_hash}_{tech_stack}"
                if cache_key in self._analysis_cache:
                    logger.info(f"从缓存获取设计图分析结果: {cache_key}")
                    cached_result = self._analysis_cache[cache_key]
                    
                    # 确保缓存结果中包含原始图像数据
                    if not cached_result.get("image_base64"):
                        cached_result["image_base64"] = original_image_data
                        logger.info("已将原始图像数据添加到缓存结果中")
                    
                    return cached_result
            
            # 没有缓存，处理图像
            if is_base64_data:
                try:
                    # 提取base64数据部分
                    header, b64_data = image_path_or_base64.split(',', 1)
                    
                    # 创建临时文件
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    temp_file.close()
                    
                    # 解码并写入临时文件
                    image_data = base64.b64decode(b64_data)
                    with open(temp_file.name, 'wb') as f:
                        f.write(image_data)
                    
                    local_image_path = temp_file.name
                    logger.info(f"已将base64图像数据保存为临时文件: {local_image_path}")
                except Exception as e:
                    logger.error(f"处理base64数据时出错: {str(e)}")
                    return {
                        "status": "error",
                        "error": "base64处理失败",
                        "error_message": str(e),
                        "image_base64": original_image_data  # 返回原始base64数据
                    }
            else:
                # 使用提供的文件路径
                if os.path.exists(image_path_or_base64):
                    local_image_path = image_path_or_base64
                    logger.info(f"使用文件路径: {local_image_path}")
                else:
                    logger.error(f"文件不存在: {image_path_or_base64}")
                    return {
                        "status": "error",
                        "error": "文件不存在",
                        "error_message": f"指定的文件路径不存在: {image_path_or_base64}"
                    }
            
            # 分析图像
            try:
                # 构建分析提示
                analysis_prompt = self._build_analysis_prompt(tech_stack)
                
                # 读取图像数据
                with open(local_image_path, 'rb') as f:
                    image_data = f.read()
                
                # 使用视觉模型分析图像
                vision_analysis = await self._analyze_with_vision_model(image_data, analysis_prompt)
                
                # 处理分析结果
                try:
                    # 格式化为结构化数据
                    sections = self._split_analysis_sections(vision_analysis)
                    
                    # 构建基本结果
                    result = {
                        "status": "success",
                        "raw_analysis": vision_analysis,
                        "summary": sections.get("overview", "未能提取设计概述"),
                        "elements": [],
                        "colors": [],
                        "layout": {},
                        "fonts": [],
                        "image_base64": original_image_data,  # 添加base64数据
                        "processing_time": time.time() - start_time
                    }
                    
                    # 处理UI组件部分
                    ui_components = self._parse_components_from_text(vision_analysis, tech_stack)
                    if ui_components:
                        result["elements"] = ui_components
                    
                    # 处理颜色方案
                    try:
                        color_scheme = await self._extract_color_scheme(local_image_path)
                        if color_scheme and "colors" in color_scheme:
                            result["colors"] = color_scheme["colors"]
                    except Exception as e:
                        logger.warning(f"提取颜色方案失败: {str(e)}")
                    
                    # 如果颜色为空，尝试从文本中提取
                    if not result["colors"]:
                        colors = []
                        color_section = sections.get("colors", "")
                        hex_pattern = r'#[0-9A-Fa-f]{6}'
                        hex_matches = re.findall(hex_pattern, color_section)
                        for i, hex_code in enumerate(hex_matches[:5]):
                            colors.append({
                                "hex": hex_code,
                                "name": f"颜色{i+1}"
                            })
                        result["colors"] = colors
                    
                    # 处理布局信息
                    layout = {"structure": "未知布局", "alignment": "未知对齐", "spacing": "未知间距"}
                    layout_section = sections.get("layout", "")
                    if layout_section:
                        layout_lines = layout_section.split('\n')
                        for line in layout_lines:
                            if "结构" in line:
                                layout["structure"] = line.split("结构", 1)[1].strip().strip(':：').strip()
                            elif "对齐" in line:
                                layout["alignment"] = line.split("对齐", 1)[1].strip().strip(':：').strip()
                            elif "间距" in line:
                                layout["spacing"] = line.split("间距", 1)[1].strip().strip(':：').strip()
                    result["layout"] = layout
                    
                    # 处理字体信息
                    fonts = []
                    fonts_section = sections.get("typography", "")
                    if fonts_section:
                        font_lines = fonts_section.split('\n')
                        for line in font_lines:
                            if line.startswith('-') or line.startswith('*'):
                                font_info = line[1:].strip()
                                font_parts = font_info.split(',')
                                if len(font_parts) >= 1:
                                    font_name = font_parts[0].strip()
                                    font_style = "Regular"
                                    font_size = "未知"
                                    
                                    if len(font_parts) >= 2:
                                        for part in font_parts[1:]:
                                            if "px" in part or "pt" in part or "sp" in part:
                                                font_size = part.strip()
                                            elif any(style in part.lower() for style in ["bold", "regular", "medium", "light"]):
                                                font_style = part.strip()
                                    
                                    fonts.append({
                                        "name": font_name,
                                        "style": font_style,
                                        "size": font_size
                                    })
                    result["fonts"] = fonts
                    
                    # 保存到缓存
                    if image_hash:
                        cache_key = f"{image_hash}_{tech_stack}"
                        self._analysis_cache[cache_key] = result
                        
                        # 异步保存缓存
                        try:
                            self._save_analysis_cache()
                            logger.info(f"已保存分析结果到缓存: {cache_key}")
                        except Exception as e:
                            logger.warning(f"保存分析结果到缓存时出错: {str(e)}")
                    
                    # 清理临时文件
                    if temp_file and os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                        logger.info(f"已删除临时文件: {temp_file.name}")
                    
                    logger.info(f"设计图分析完成，耗时: {result['processing_time']:.2f}秒")
                    return result
                    
                except Exception as e:
                    logger.error(f"解析分析结果时出错: {str(e)}")
                    logger.error(traceback.format_exc())
                    
                    # 返回带有base64数据的基本结果
                    return {
                        "status": "error",
                        "error": "解析失败",
                        "error_message": str(e),
                        "image_base64": original_image_data,
                        "raw_analysis": vision_analysis,
                        "elements": [],
                        "colors": [],
                        "layout": {},
                        "fonts": [],
                        "summary": "分析结果解析失败，无法提供详细信息。"
                    }
            except Exception as e:
                logger.error(f"分析图像时出错: {str(e)}")
                logger.error(traceback.format_exc())
                
                # 返回带有base64数据的错误结果
                return {
                    "status": "error",
                    "error": "分析失败",
                    "error_message": str(e),
                    "image_base64": original_image_data,
                    "elements": [],
                    "colors": [],
                    "layout": {},
                    "fonts": [],
                    "summary": "图像分析过程中出错，无法提供分析结果。"
                }
        except Exception as e:
            logger.error(f"处理设计图时出错: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 返回基本的错误结果
            return {
                "status": "error",
                "error": "未知错误",
                "error_message": str(e),
                "image_base64": original_image_data,
                "elements": [],
                "colors": [],
                "layout": {},
                "fonts": [],
                "summary": "处理设计图时出现未知错误，无法进行分析。"
            }
        finally:
            # 确保清理临时文件
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                    logger.info(f"已删除临时文件: {temp_file.name}")
                except Exception as e:
                    logger.warning(f"清理临时文件时出错: {str(e)}")
    
    def _extract_basic_image_info(self, image_path: str) -> Dict[str, Any]:
        """提取图像的基本信息
        
        Args:
            image_path: 图像路径
            
        Returns:
            Dict[str, Any]: 基本图像信息
        """
        try:
            from PIL import Image
            img = Image.open(image_path)
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "aspect_ratio": round(img.width / img.height, 2) if img.height > 0 else 0
            }
        except Exception as e:
            logger.warning(f"提取图像基本信息失败: {str(e)}")
            return {
                "width": 0,
                "height": 0,
                "format": "未知",
                "mode": "未知",
                "aspect_ratio": 0
            }
    
    async def _save_analysis_cache_async(self):
        """异步保存分析缓存"""
        try:
            with open(self._analysis_cache_file, "w", encoding="utf-8") as f:
                json.dump(self._analysis_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"已异步保存分析缓存: {len(self._analysis_cache)}条记录")
        except Exception as e:
            logger.error(f"异步保存分析缓存失败: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _analyze_image_for_tech_stack(
        self, 
        image_path: str, 
        tech_stack: str
    ) -> Dict[str, Any]:
        """根据技术栈分析设计图
        
        Args:
            image_path: 图片路径
            tech_stack: 技术栈
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            # 检查技术栈是否受支持
            supported_tech_stacks = settings.DESIGN_PROMPT_CONFIG.get("supported_tech_stacks", [])
            if tech_stack not in supported_tech_stacks:
                logger.warning(f"不支持的技术栈: {tech_stack}")
                return {
                    "status": "error",
                    "message": f"不支持的技术栈: {tech_stack}"
                }
            
            # 获取技术栈特定配置
            tech_config = settings.DESIGN_PROMPT_CONFIG.get("tech_stack_config", {}).get(tech_stack, {})
            
            # 拼接技术栈特定分析提示词
            prompt = self._build_tech_stack_analysis_prompt(tech_stack, tech_config)
            
            # 使用视觉模型分析
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                result = await self._analyze_with_vision_model(image_data, prompt)
            
            return {
                "status": "success",
                "tech_stack": tech_stack,
                "analysis": result,
                "tech_config": tech_config
            }
        except Exception as e:
            logger.error(f"技术栈特定分析失败: {str(e)}")
            return {
                "status": "error",
                "message": f"技术栈特定分析失败: {str(e)}"
            }
    
    def _build_tech_stack_analysis_prompt(self, tech_stack: str, tech_config: Dict[str, Any]) -> str:
        """构建技术栈特定分析提示词
        
        Args:
            tech_stack: 技术栈
            tech_config: 技术栈配置
            
        Returns:
            str: 分析提示词
        """
        # 基础提示词
        base_prompt = f"""
        请作为UI/UX专家和{tech_stack}开发者，详细分析这张设计图。
        
        请提供以下信息：
        1. 设计图总体描述
        2. 主要UI元素和布局结构
        3. 颜色方案和排版特点
        4. 交互元素和可能的用户流程
        5. 适合在{tech_stack}中实现的组件列表
        """
        
        # 添加技术栈特定提示
        if tech_stack == "Android":
            frameworks = tech_config.get("frameworks", [])
            prompt = base_prompt + f"""
            6. 如何使用{", ".join(frameworks)}实现此界面
            7. 推荐的Android特定组件和库：
               - 布局类型（ConstraintLayout/LinearLayout等或Compose布局）
               - 列表实现（RecyclerView或LazyColumn等）
               - 导航组件
               - 动画实现方式
            8. 可能的屏幕适配考虑点
            9. 深色模式适配建议
            
            请重点关注Material Design组件的应用，以及Android特有的UI模式和交互。
            """
        elif tech_stack == "iOS":
            frameworks = tech_config.get("frameworks", [])
            prompt = base_prompt + f"""
            6. 如何使用{", ".join(frameworks)}实现此界面
            7. 推荐的iOS特定组件和库：
               - 视图结构（UIKit视图层次或SwiftUI视图）
               - 列表实现（UITableView/UICollectionView或List/LazyVGrid等）
               - 导航控制器
               - 动画实现方式
            8. 可能的屏幕适配考虑点
            9. Dark Mode适配建议
            
            请重点关注Human Interface Guidelines，以及iOS特有的UI模式和交互。
            """
        elif tech_stack == "Flutter":
            prompt = base_prompt + """
            6. 如何使用Flutter Widget实现此界面
            7. 推荐的Flutter特定组件：
               - 布局Widget（Column/Row/Stack等）
               - 列表Widget（ListView/GridView等）
               - 导航方案
               - 动画实现方式
            8. 跨平台一致性考虑点
            9. 主题适配建议
            
            请重点关注Flutter Material/Cupertino组件的应用，以及Flutter特有的UI构建方式。
            """
        elif tech_stack == "React":
            prompt = base_prompt + """
            6. 如何使用React组件实现此界面
            7. 推荐的React UI库和组件：
               - 布局组件
               - 列表/表格组件
               - 导航/路由方案
               - 动画实现方式
            8. 响应式设计考虑点
            9. 主题切换实现建议
            
            请重点关注组件拆分策略，以及React生态中常用UI库的应用。
            """
        elif tech_stack == "Vue":
            prompt = base_prompt + """
            6. 如何使用Vue组件实现此界面
            7. 推荐的Vue UI库和组件：
               - 布局组件
               - 列表/表格组件
               - 导航/路由方案
               - 动画实现方式
            8. 响应式设计考虑点
            9. 主题切换实现建议
            
            请重点关注Vue组件设计，以及Vue生态中常用UI库的应用。
            """
        else:
            # 默认通用提示
            prompt = base_prompt
        
        return prompt
    
    async def _analyze_with_vision_model(self, image_data: bytes, prompt: str) -> str:
        """使用视觉模型分析图片
        
        Args:
            image_data: 图片数据
            prompt: 分析提示词
            
        Returns:
            str: 分析结果
        """
        try:
            # 使用OpenAI视觉模型
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            response = await asyncio.to_thread(
                self.vision_model.chat.completions.create,
                model=settings.VISION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的UI/UX分析师，擅长分析设计图并提供技术实现建议。"
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=settings.VISION_MODEL_CONFIG["temperature"],
                max_tokens=settings.VISION_MODEL_CONFIG["max_tokens"]
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"视觉模型分析失败: {str(e)}")
            raise
            
    async def _extract_color_scheme(self, image_path: str) -> Dict[str, Any]:
        """提取设计图的配色方案
        
        Args:
            image_path: 图片路径
            
        Returns:
            Dict[str, Any]: 配色方案
        """
        try:
            img = Image.open(image_path)
            img = img.convert('RGBA')
            img = img.resize((150, 150))  # 缩小图片以加快处理速度
            
            # 将图片转换为numpy数组
            pixels = np.array(img)
            
            # 排除透明背景
            pixels = pixels[pixels[:, :, 3] > 0]
            
            # 如果没有非透明像素，返回空结果
            if len(pixels) == 0:
                return {
                    "primary_colors": [],
                    "color_palette": []
                }
            
            # 使用K-means聚类提取主要颜色
            from sklearn.cluster import KMeans
            
            # 保留RGB通道，忽略Alpha通道
            pixels_rgb = pixels[:, :3]
            
            # 使用K-means提取6种主要颜色
            kmeans = KMeans(n_clusters=6, random_state=0)
            kmeans.fit(pixels_rgb)
            colors = kmeans.cluster_centers_
            
            # 计算每个颜色的重要性（所占像素比例）
            labels = kmeans.labels_
            counts = np.bincount(labels)
            percentage = counts / len(labels)
            
            # 按重要性排序颜色
            sorted_indices = percentage.argsort()[::-1]
            sorted_colors = colors[sorted_indices]
            sorted_percentages = percentage[sorted_indices]
            
            # 将颜色转换为十六进制格式
            hex_colors = []
            for color in sorted_colors:
                r, g, b = [int(c) for c in color]
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                hex_colors.append(hex_color)
            
            # 识别主要和次要颜色
            primary_colors = []
            for i, (hex_color, percent) in enumerate(zip(hex_colors[:3], sorted_percentages[:3])):
                primary_colors.append({
                    "color": hex_color,
                    "percentage": float(percent),
                    "role": "primary" if i == 0 else "secondary" if i == 1 else "accent"
                })
            
            return {
                "primary_colors": primary_colors,
                "color_palette": hex_colors
            }
        except Exception as e:
            logger.error(f"提取配色方案失败: {str(e)}")
            return {
                "primary_colors": [],
                "color_palette": []
            }
    
    async def _extract_typography(self, image_path: str) -> Dict[str, Any]:
        """提取设计图的排版信息
        
        Args:
            image_path: 图片路径
            
        Returns:
            Dict[str, Any]: 排版信息
        """
        # 使用视觉模型提取排版信息的提示词
        prompt = """
        请分析此设计图的排版系统，提供以下信息：
        1. 字体系列：识别主要使用的字体类型（sans-serif、serif、monospace等）
        2. 字体大小层级：识别不同级别文本的大小（如标题、副标题、正文等）
        3. 字重使用：粗体、常规、细体的使用模式
        4. 行高和段落间距
        5. 文本对齐方式
        
        请以结构化方式呈现结果，不要猜测具体字体名称，只需提供字体类别和特征。
        """
        
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                result = await self._analyze_with_vision_model(image_data, prompt)
            
            # 尝试提取结构化信息
            typography = {
                "font_family": self._extract_font_family(result),
                "text_styles": self._extract_text_styles(result),
                "raw_analysis": result
            }
            
            return typography
        except Exception as e:
            logger.error(f"提取排版信息失败: {str(e)}")
            return {
                "font_family": "未知",
                "text_styles": [],
                "raw_analysis": "分析失败"
            }
    
    def _extract_font_family(self, text: str) -> str:
        """从分析文本中提取字体家族
        
        Args:
            text: 分析文本
            
        Returns:
            str: 字体家族
        """
        # 使用正则表达式查找字体家族相关描述
        patterns = [
            r"字体系列[：:]\s*([^\n\.。]+)",
            r"主要(?:使用的)?字体(?:类型)?[：:]\s*([^\n\.。]+)",
            r"字体[：:]\s*([^\n\.。]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return "Sans-serif"  # 默认值
    
    def _extract_text_styles(self, text: str) -> List[Dict[str, str]]:
        """从分析文本中提取文本样式
        
        Args:
            text: 分析文本
            
        Returns:
            List[Dict[str, str]]: 文本样式列表
        """
        styles = []
        
        # 查找字体大小层级描述
        size_section = re.search(r"字体大小[^：:]*[：:]\s*([^\n]+(?:\n[^1-5\n][^\n]*)*)", text)
        if size_section:
            size_text = size_section.group(1)
            # 查找标题、副标题、正文等
            for style_type in ["标题", "副标题", "正文", "小标题", "说明文字", "按钮文字"]:
                pattern = rf"{style_type}[：:]?\s*([^\n\.。,，;；]+)"
                match = re.search(pattern, size_text)
                if match:
                    size_desc = match.group(1).strip()
                    styles.append({
                        "type": style_type,
                        "size": size_desc
                    })
        
        # 如果没有找到任何样式，添加默认样式
        if not styles:
            styles = [
                {"type": "标题", "size": "大"},
                {"type": "副标题", "size": "中"},
                {"type": "正文", "size": "小"}
            ]
        
        return styles
    
    async def _identify_ui_components(self, image_path: str, tech_stack: str) -> List[Dict[str, Any]]:
        """识别设计图中的UI组件
        
        Args:
            image_path: 图片路径
            tech_stack: 技术栈
            
        Returns:
            List[Dict[str, Any]]: UI组件列表
        """
        # 根据技术栈构建UI组件识别提示词
        tech_config = settings.DESIGN_PROMPT_CONFIG.get("tech_stack_config", {}).get(tech_stack, {})
        ui_components = tech_config.get("ui_components", [])
        
        prompt = f"""
        请识别此设计图中的UI组件，并匹配到{tech_stack}技术栈中的标准组件。
        
        特别关注以下{tech_stack}组件类型：
        {', '.join(ui_components)}
        
        对于每个识别到的组件，请提供：
        1. 组件名称
        2. 对应的{tech_stack}标准组件
        3. 组件在界面中的位置和作用
        4. 可能的属性和状态
        
        请以列表形式返回结果，确保信息清晰、准确。
        """
        
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                result = await self._analyze_with_vision_model(image_data, prompt)
            
            # 提取组件信息
            components = self._parse_components_from_text(result, tech_stack)
            
            return components
        except Exception as e:
            logger.error(f"识别UI组件失败: {str(e)}")
            return []
    
    def _parse_components_from_text(self, text: str, tech_stack: str) -> List[Dict[str, Any]]:
        """从分析文本中解析UI组件
        
        Args:
            text: 分析文本
            tech_stack: 技术栈
            
        Returns:
            List[Dict[str, Any]]: UI组件列表
        """
        components = []
        
        # 尝试提取组件列表
        # 匹配数字列表项或带点的列表项后跟组件名称
        component_patterns = [
            r'\d+\.\s*([^：:]+)[：:]\s*([^\n]+)',
            r'[-*]\s*([^：:]+)[：:]\s*([^\n]+)',
            r'([A-Za-z\u4e00-\u9fa5]+(?:\s+[A-Za-z\u4e00-\u9fa5]+)*)\s*组件[：:]\s*([^\n]+)'
        ]
        
        for pattern in component_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) >= 2:
                    name = match.group(1).strip()
                    description = match.group(2).strip()
                    
                    # 避免添加已存在的组件
                    if not any(c.get("name") == name for c in components):
                        components.append({
                            "name": name,
                            "tech_stack": tech_stack,
                            "description": description,
                            "source": "vision_analysis"
                        })
        
        # 如果没有找到任何组件，尝试从整个文本中提取信息
        if not components:
            # 尝试查找常见的UI组件名称
            ui_component_names = self._get_common_ui_components(tech_stack)
            for component_name in ui_component_names:
                if component_name.lower() in text.lower():
                    # 尝试提取关于这个组件的一句描述
                    pattern = rf'{component_name}[^。.。\n]*[。.。]'
                    match = re.search(pattern, text, re.IGNORECASE)
                    description = match.group(0) if match else f"{component_name}组件"
                    
                    components.append({
                        "name": component_name,
                        "tech_stack": tech_stack,
                        "description": description,
                        "source": "vision_analysis"
                    })
        
        return components
    
    def _get_common_ui_components(self, tech_stack: str) -> List[str]:
        """获取常见UI组件列表
        
        Args:
            tech_stack: 技术栈
            
        Returns:
            List[str]: UI组件名称列表
        """
        # 从配置中获取UI组件
        tech_config = settings.DESIGN_PROMPT_CONFIG.get("tech_stack_config", {}).get(tech_stack, {})
        return tech_config.get("ui_components", [])
    
    async def _extract_spacing(self, image_path: str) -> Dict[str, Any]:
        """提取设计图的间距信息
        
        Args:
            image_path: 图片路径
            
        Returns:
            Dict[str, Any]: 间距信息
        """
        prompt = """
        请分析此设计图的间距和对齐系统，提供以下信息：
        1. 水平间距模式
        2. 垂直间距模式
        3. 内边距(padding)使用模式
        4. 外边距(margin)使用模式
        5. 对齐方式
        
        请注意识别是否有一致的间距模式，例如8dp网格系统或其他间距系统。
        """
        
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                result = await self._analyze_with_vision_model(image_data, prompt)
            
            # 提取关键信息
            spacing = {
                "horizontal_spacing": self._extract_spacing_info(result, "水平间距"),
                "vertical_spacing": self._extract_spacing_info(result, "垂直间距"),
                "padding": self._extract_spacing_info(result, "内边距"),
                "margin": self._extract_spacing_info(result, "外边距"),
                "alignment": self._extract_spacing_info(result, "对齐方式"),
                "raw_analysis": result
            }
            
            return spacing
        except Exception as e:
            logger.error(f"提取间距信息失败: {str(e)}")
            return {
                "horizontal_spacing": "未知",
                "vertical_spacing": "未知",
                "padding": "未知",
                "margin": "未知",
                "alignment": "未知",
                "raw_analysis": "分析失败"
            }
    
    def _extract_spacing_info(self, text: str, type_name: str) -> str:
        """从分析文本中提取指定类型的间距信息
        
        Args:
            text: 分析文本
            type_name: 间距类型名称
            
        Returns:
            str: 间距信息
        """
        pattern = rf'{type_name}[^：:]*[：:]\s*([^\n\.。]+)'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return "一致性间距"  # 默认值
    
    async def enhance_design_analysis(
        self, 
        image_path: str, 
        tech_stack: str
    ) -> Dict[str, Any]:
        """增强设计图分析
        
        Args:
            image_path: 图片路径
            tech_stack: 技术栈
            
        Returns:
            Dict[str, Any]: 增强分析结果
        """
        # 获取基础分析结果
        base_analysis = await self.analyze_design(image_path)
        
        # 并行执行多个分析任务
        tech_stack_analysis, color_scheme, typography, spacing, ui_components = await asyncio.gather(
            self._analyze_image_for_tech_stack(image_path, tech_stack),
            self._extract_color_scheme(image_path),
            self._extract_typography(image_path),
            self._extract_spacing(image_path),
            self._identify_ui_components(image_path, tech_stack)
        )
        
        # 整合所有分析结果
        enhanced_analysis = {
            **base_analysis,
            "tech_stack_analysis": tech_stack_analysis,
            "color_scheme": color_scheme,
            "typography": typography,
            "spacing": spacing,
            "ui_components": ui_components
        }
        
        return enhanced_analysis 

    async def analyze_image(self, image_id: str, image_path: str) -> Dict[str, Any]:
        """分析设计图像（兼容性方法，是analyze_design_image的包装器）
        
        Args:
            image_id: 图像ID
            image_path: 图像路径
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        logger.info(f"调用analyze_image方法: image_id={image_id}, image_path={image_path}")
        try:
            # 确定技术栈
            tech_stack = None
            
            # 尝试从图像ID提取技术栈信息
            if "_" in image_id and not image_id.startswith("data:image/"):
                parts = image_id.split("_")
                if len(parts) > 1 and parts[0] in settings.DESIGN_PROMPT_CONFIG.get("supported_tech_stacks", []):
                    tech_stack = parts[0]
                    logger.info(f"从图像ID提取到技术栈: {tech_stack}")
            
            if not tech_stack:
                # 使用默认技术栈
                tech_stack = "Android"
                logger.info(f"使用默认技术栈: {tech_stack}")
            
            # 调用实际的分析方法
            result = await self.analyze_design_image(image_path, tech_stack)
            return result
        except Exception as e:
            logger.error(f"analyze_image失败: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 返回基本分析结果
            return {
                "status": "error",
                "error": str(e),
                "elements": [],
                "colors": [],
                "fonts": [],
                "layout": {},
                "summary": f"图像分析失败: {str(e)}"
            } 