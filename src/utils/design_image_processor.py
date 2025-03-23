import logging
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import uuid
from datetime import datetime
import base64
from PIL import Image
import io
import numpy as np
import json
<<<<<<< HEAD
import hashlib
import time
=======
import asyncio
import time
import copy
import re
>>>>>>> 82c1bcc0ead144b5abb7ab2621735f4f0e5a6b88

from config.config import settings
from src.utils.vector_store import VectorStore
from openai import OpenAI
from langchain_openai.chat_models import ChatOpenAI

logger = logging.getLogger(__name__)

class DesignImageProcessor:
    """设计图处理器，用于处理上传的设计图并生成向量"""
    
<<<<<<< HEAD
    # 添加缓存相关的类变量
    _analysis_cache = {}  # 内存缓存：图像哈希 -> 分析结果
    _analysis_cache_file = None  # 缓存文件路径
    _cache_expiry = 7 * 24 * 60 * 60  # 缓存过期时间（7天）
    _max_cache_size = 100  # 最大缓存条目数
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
=======
    def __init__(
        self,
        upload_dir: Optional[str] = None,
        vector_store: Optional[VectorStore] = None,
        temperature: float = None,
        max_tokens: int = None
    ):
>>>>>>> 82c1bcc0ead144b5abb7ab2621735f4f0e5a6b88
        """初始化设计图处理器
        
        Args:
            upload_dir: 上传目录
            vector_store: 向量存储实例
            temperature: 温度参数
            max_tokens: 最大token数
        """
        self.upload_dir = upload_dir or os.path.join(settings.UPLOAD_DIR, "design_images")
        os.makedirs(self.upload_dir, exist_ok=True)
        
        self.vector_store = vector_store or VectorStore()
        
        # 使用传入的参数或配置文件中的默认值
        self.temperature = temperature or settings.PROMPT_OPTIMIZATION_CONFIG["temperature"]
        self.max_tokens = max_tokens or settings.PROMPT_OPTIMIZATION_CONFIG["max_tokens"]
        
        # 初始化LLM
        self.llm = ChatOpenAI(
            model_name=settings.DESIGN_PROMPT_CONFIG.get("model_name", settings.DEFAULT_MODEL_NAME),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        )
        
        # 初始化视觉模型
        self.vision_model = ChatOpenAI(
            model_name=settings.VISION_MODEL_NAME,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        )
        
<<<<<<< HEAD
        # 初始化缓存文件路径
        if DesignImageProcessor._analysis_cache_file is None:
            DesignImageProcessor._analysis_cache_file = Path(settings.VECTOR_DB_PATH) / "design_analysis_cache.json"
            self._load_analysis_cache()
        
        # 记录使用的视觉模型
        vision_model = settings.VISION_MODEL
        vision_temp = settings.VISION_MODEL_CONFIG["temperature"]
        vision_max_tokens = settings.VISION_MODEL_CONFIG["max_tokens"]
        logger.info(f"设计图处理器初始化完成，使用视觉模型: {vision_model}, 温度: {vision_temp}, 最大tokens: {vision_max_tokens}")
        logger.info(f"上传目录: {self.upload_dir}")
    
    def _load_analysis_cache(self):
        """加载分析结果缓存"""
        try:
            if DesignImageProcessor._analysis_cache_file.exists():
                with open(DesignImageProcessor._analysis_cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    # 过滤掉过期的缓存项
                    current_time = time.time()
                    DesignImageProcessor._analysis_cache = {
                        k: v for k, v in cache_data.items()
                        if current_time - v.get('timestamp', 0) < self._cache_expiry
                    }
                logger.info(f"加载了 {len(DesignImageProcessor._analysis_cache)} 个设计图分析缓存")
            else:
                logger.info("没有找到设计图分析缓存文件，将创建新缓存")
                DesignImageProcessor._analysis_cache = {}
        except Exception as e:
            logger.error(f"加载设计图分析缓存失败: {str(e)}")
            DesignImageProcessor._analysis_cache = {}
    
    def _save_analysis_cache(self):
        """保存分析结果缓存"""
        try:
            # 确保目录存在
            DesignImageProcessor._analysis_cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(DesignImageProcessor._analysis_cache_file, 'w', encoding='utf-8') as f:
                json.dump(DesignImageProcessor._analysis_cache, f, ensure_ascii=False)
            logger.info(f"已保存 {len(DesignImageProcessor._analysis_cache)} 个设计图分析缓存")
        except Exception as e:
            logger.error(f"保存设计图分析缓存失败: {str(e)}")
    
    def _compute_image_hash(self, image_data: bytes) -> str:
        """计算图像内容哈希值
        
        Args:
            image_data: 图像数据
            
        Returns:
            str: 图像哈希值
        """
        return hashlib.md5(image_data).hexdigest()
    
    def _prune_cache_if_needed(self):
        """如果缓存大小超过限制，清理最旧的缓存项"""
        if len(DesignImageProcessor._analysis_cache) > self._max_cache_size:
            # 按时间戳排序
            sorted_items = sorted(
                DesignImageProcessor._analysis_cache.items(),
                key=lambda x: x[1].get('timestamp', 0)
            )
            # 移除最旧的项，直到缓存大小符合限制
            items_to_remove = len(sorted_items) - self._max_cache_size
            for i in range(items_to_remove):
                del DesignImageProcessor._analysis_cache[sorted_items[i][0]]
            logger.info(f"清理了 {items_to_remove} 个旧的缓存项")
    
    def is_supported_image(self, filename: str) -> bool:
        """检查文件是否为支持的图片格式
        
        Args:
            filename: 文件名
=======
        # 添加设计图分析结果缓存
        self.analysis_cache = {}
        self.cache_file = Path(settings.UPLOAD_DIR) / "design_analysis_cache.json"
        self._load_analysis_cache()
        
        logger.info("设计图处理器初始化完成")
    
    def _load_analysis_cache(self):
        """加载设计图分析结果缓存"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.analysis_cache = json.load(f)
                logger.info(f"已加载设计图分析结果缓存: {len(self.analysis_cache)}条记录")
            else:
                logger.info("设计图分析结果缓存文件不存在，将创建新缓存")
                self.analysis_cache = {}
        except Exception as e:
            logger.error(f"加载设计图分析结果缓存失败: {str(e)}")
            self.analysis_cache = {}
    
    def _save_analysis_cache(self):
        """保存设计图分析结果缓存"""
        try:
            # 限制缓存大小，最多保留100条记录
            if len(self.analysis_cache) > 100:
                # 按时间戳排序，保留最新的100条
                sorted_cache = sorted(
                    self.analysis_cache.items(), 
                    key=lambda x: x[1].get("timestamp", 0),
                    reverse=True
                )
                self.analysis_cache = dict(sorted_cache[:100])
            
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.analysis_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"已保存设计图分析结果缓存: {len(self.analysis_cache)}条记录")
        except Exception as e:
            logger.error(f"保存设计图分析结果缓存失败: {str(e)}")
    
    def _get_image_hash(self, image_content: bytes) -> str:
        """计算图片哈希值
        
        Args:
            image_content: 图片数据
            
        Returns:
            str: 图片哈希值
        """
        import hashlib
        return hashlib.md5(image_content).hexdigest()
    
    def is_supported_image(self, file_path: str) -> bool:
        """检查是否为支持的图片格式
        
        Args:
            file_path: 文件路径
>>>>>>> 82c1bcc0ead144b5abb7ab2621735f4f0e5a6b88
            
        Returns:
            bool: 是否支持
        """
<<<<<<< HEAD
        try:
            # 获取文件扩展名并转为小写
            ext = Path(filename).suffix.lower().lstrip('.')
            
            # 检查扩展名是否在支持列表中
            supported_formats = ['jpeg', 'jpg', 'png', 'webp']
            is_supported = ext in supported_formats
            
            if not is_supported:
                logger.warning(f"不支持的图片格式: {ext}")
            
            return is_supported
            
        except Exception as e:
            logger.error(f"检查图片格式失败: {str(e)}")
            return False
=======
        supported_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in supported_extensions
>>>>>>> 82c1bcc0ead144b5abb7ab2621735f4f0e5a6b88
    
    async def process_image(
        self,
        file_content: bytes,
        file_name: str,
        image_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """处理设计图
        
        Args:
            file_content: 文件内容
            file_name: 文件名
            image_id: 图片ID，如果为None则自动生成
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            # 检查文件格式
            if not self.is_supported_image(file_name):
                raise ValueError(f"不支持的文件格式: {file_name}")
            
            # 生成图片ID
            if image_id is None:
                image_id = str(uuid.uuid4())
            
            # 获取文件扩展名
            ext = os.path.splitext(file_name)[1].lower()
            
<<<<<<< HEAD
            # 5. 计算图像哈希，用于缓存查找
            image_hash = self._compute_image_hash(image_data)
            logger.info(f"图像哈希值: {image_hash}")
            
            # 6. 检查缓存中是否有分析结果
            analysis_text = None
            cache_hit = False
            if image_hash in DesignImageProcessor._analysis_cache:
                cache_entry = DesignImageProcessor._analysis_cache[image_hash]
                # 检查是否过期
                if time.time() - cache_entry.get('timestamp', 0) < self._cache_expiry:
                    analysis_text = cache_entry.get('analysis_text')
                    analysis_sections = cache_entry.get('analysis_sections', {})
                    if analysis_text:
                        logger.info(f"使用缓存的设计图分析结果: {image_hash}")
                        cache_hit = True
            
            # 7. 如果缓存中没有，使用多模态模型分析图片
            if not cache_hit:
                try:
                    # 构建详细的分析提示
                    analysis_prompt = (
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
                    
                    # 准备API调用
                    logger.info(f"开始调用vision model进行图片分析: {filename}")
                    try:
                        response = self.vision_model.chat.completions.create(
                            model=settings.VISION_MODEL,
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "你是一个专业的UI设计分析专家，擅长分析移动应用界面设计。"
                                        "请提供详细、准确、可实现的分析结果，包含所有必要的技术细节。"
                                        "确保所有尺寸、颜色、位置关系等信息都是具体和精确的。"
                                    )
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": analysis_prompt
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/{img_format};base64,{base64.b64encode(image_data).decode('utf-8')}"
                                            }
                                        }
                                    ]
                                }
                            ],
                            max_tokens=settings.VISION_MODEL_CONFIG["max_tokens"],
                            temperature=settings.VISION_MODEL_CONFIG["temperature"]
                        )
                        logger.info(f"Vision model ({settings.VISION_MODEL}) 调用成功: {filename}")
                    except Exception as e:
                        logger.error(f"Vision model ({settings.VISION_MODEL}) 调用失败: {str(e)}", exc_info=True)
                        raise ValueError(f"Vision model调用失败: {str(e)}")
                    
                    # 验证分析结果
                    analysis_text = response.choices[0].message.content
                    
                    if not analysis_text:
                        logger.error(f"分析结果为空: {filename}")
                        raise ValueError("分析结果为空")
                    elif len(analysis_text.strip()) < 100:
                        logger.warning(f"分析结果过短 ({len(analysis_text.strip())}字符): {filename}")
                        raise ValueError("分析结果过短")
                    
                    logger.info(f"成功获取分析结果 ({len(analysis_text.strip())}字符): {filename}")
                        
                except Exception as e:
                    logger.error(f"图片分析失败: {str(e)}")
                    analysis_text = f"设计图文件名: {filename}\n技术栈: {tech_stack}\n上传时间: {datetime.now().isoformat()}"
            
            # 8. 将分析结果按类别拆分
            analysis_sections = self._split_analysis_sections(analysis_text)
            
            # 9. 如果是新分析的结果，添加到缓存
            if not cache_hit:
                # 保存分析结果到缓存
                DesignImageProcessor._analysis_cache[image_hash] = {
                    'analysis_text': analysis_text,
                    'analysis_sections': analysis_sections,
                    'tech_stack': tech_stack,
                    'timestamp': time.time()
                }
                # 清理缓存（如果需要）
                self._prune_cache_if_needed()
                # 保存缓存到磁盘
                self._save_analysis_cache()
                logger.info(f"已将分析结果添加到缓存: {image_hash}")
                
            # 10. 添加到向量存储
            try:
                # 为每个部分创建单独的向量
                for section_type, section_text in analysis_sections.items():
                    section_metadata = {
                        **full_metadata,
                        "section_type": section_type,
                        "image_hash": image_hash  # 添加图像哈希到元数据，便于后续查找
                    }
                    
                    await self.vector_store.add_texts(
                        texts=[section_text],
                        metadatas=[section_metadata],
                        ids=[f"{file_id}_{section_type}"]
                    )
                    logger.info(f"已添加{section_type}部分到向量存储")
                    
            except Exception as e:
                logger.error(f"添加到向量存储失败: {str(e)}")
                raise
=======
            # 保存文件
            file_path = os.path.join(self.upload_dir, f"{image_id}{ext}")
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            # 计算图片哈希值用于缓存
            image_hash = self._get_image_hash(file_content)
            
            # 检查缓存
            if image_hash in self.analysis_cache:
                cache_data = self.analysis_cache[image_hash]
                logger.info(f"从缓存中获取设计图分析结果: {image_hash}")
                # 更新时间戳
                cache_data["timestamp"] = time.time()
                self.analysis_cache[image_hash] = cache_data
                self._save_analysis_cache()
                
                # 返回缓存结果，但更新ID和路径
                result = copy.deepcopy(cache_data["result"])
                result["id"] = image_id
                result["file_path"] = file_path
                result["file_name"] = file_name
                result["file_size"] = len(file_content)
                return result
>>>>>>> 82c1bcc0ead144b5abb7ab2621735f4f0e5a6b88
            
            # 分析设计图
            analysis_result = await self.analyze_design(file_path)
            
<<<<<<< HEAD
            return {
                "id": file_id,
                "filename": filename,
                "unique_filename": unique_filename,
                "file_path": str(file_path),
                "tech_stack": tech_stack,
                "analysis": analysis_text,
                "analysis_sections": analysis_sections,
                "image_hash": image_hash,
                "cache_hit": cache_hit,
                "image_dimensions": img.size,
                "success": True
=======
            # 构建结果
            result = {
                "id": image_id,
                "file_name": file_name,
                "file_path": file_path,
                "file_size": len(file_content),
                "analysis": analysis_result
>>>>>>> 82c1bcc0ead144b5abb7ab2621735f4f0e5a6b88
            }
            
            # 保存到缓存
            self.analysis_cache[image_hash] = {
                "result": result,
                "timestamp": time.time()
            }
            self._save_analysis_cache()
            
            return result
        except Exception as e:
            logger.error(f"处理设计图失败: {str(e)}")
            logger.exception(e)
            raise
    
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
            image_hash = self._get_image_hash(image_content)
            
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
        try:
            # 为每个部分创建单独的向量
            for section_type, section_text in analysis_sections.items():
                section_metadata = {
                    **full_metadata,
                    "section_type": section_type
                }
                
                await self.vector_store.add_texts(
                    texts=[section_text],
                    metadatas=[section_metadata],
                    ids=[f"{file_id}_{section_type}"]
                )
                logger.info(f"已添加{section_type}部分到向量存储")
        except Exception as e:
            logger.error(f"添加到向量存储失败: {str(e)}")
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

    async def analyze_design_image(self, image_path: str, tech_stack: str) -> Dict[str, Any]:
        """分析设计图
        
        Args:
            image_path: 设计图路径
            tech_stack: 技术栈
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            # 读取图片
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            # 获取文件名和格式
            filename = os.path.basename(image_path)
            img_format = os.path.splitext(filename)[1][1:].lower()
            
            # 构建分析提示
            analysis_prompt = f"""
            请分析这个UI设计图，并提供以下信息：
            
            1. 布局结构：描述整体布局、主要区域和组件排列
            2. UI组件：识别所有UI组件（按钮、输入框、列表等）
            3. 颜色方案：提取主要颜色及其用途
            4. 字体和排版：描述字体大小、样式和文本排版
            5. 交互元素：识别可交互的元素及其可能的行为
            6. 技术实现建议：基于{tech_stack}技术栈的实现建议
            
            请提供详细、准确、可实现的分析结果。
            """
            
            # 使用视觉模型分析图片
            try:
                # 记录使用的视觉模型信息
                vision_model = settings.VISION_MODEL
                vision_temp = settings.VISION_MODEL_CONFIG["temperature"]
                vision_max_tokens = settings.VISION_MODEL_CONFIG["max_tokens"]
                logger.info(f"使用视觉模型 {vision_model} 分析设计图: {filename}")
                
                response = self.vision_model.chat.completions.create(
                    model=vision_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "你是一个专业的UI设计分析专家，擅长分析移动应用界面设计。"
                                "请提供详细、准确、可实现的分析结果，包含所有必要的技术细节。"
                                "确保所有尺寸、颜色、位置关系等信息都是具体和精确的。"
                            )
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": analysis_prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/{img_format};base64,{base64.b64encode(image_data).decode('utf-8')}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=vision_max_tokens,
                    temperature=vision_temp
                )
                logger.info(f"视觉模型 {vision_model} 分析完成: {filename}, 使用温度: {vision_temp}, 最大tokens: {vision_max_tokens}")
                
                # 提取分析结果
                analysis_result = response.choices[0].message.content
                
                # 构建结果
                result = {
                    "filename": filename,
                    "tech_stack": tech_stack,
                    "analysis": analysis_result,
                    "timestamp": datetime.now().isoformat()
                }
                
                return result
                
            except Exception as e:
                logger.error(f"视觉模型 {settings.VISION_MODEL} 分析失败: {str(e)}", exc_info=True)
                raise ValueError(f"视觉模型分析失败: {str(e)}")
                
        except Exception as e:
            logger.error(f"分析设计图失败: {str(e)}", exc_info=True)
            raise 