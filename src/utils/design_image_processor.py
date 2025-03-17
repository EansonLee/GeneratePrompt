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

from config.config import settings
from src.utils.vector_store import VectorStore
from openai import OpenAI

logger = logging.getLogger(__name__)

class DesignImageProcessor:
    """设计图处理器，用于处理上传的设计图并生成向量"""
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """初始化设计图处理器
        
        Args:
            vector_store: 向量存储实例
        """
        self.vector_store = vector_store or VectorStore()
        self.upload_dir = Path(settings.UPLOAD_DIR) / "design_images"
        
        # 初始化多模态模型
        self.vision_model = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        )
        
        # 确保上传目录存在
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"设计图处理器初始化完成，上传目录: {self.upload_dir}")
    
    def is_supported_image(self, filename: str) -> bool:
        """检查文件是否为支持的图片格式
        
        Args:
            filename: 文件名
            
        Returns:
            bool: 是否支持
        """
        supported_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        ext = Path(filename).suffix.lower()
        return ext in supported_extensions
    
    async def process_image(
        self, 
        image_content: bytes, 
        filename: str, 
        tech_stack: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """处理设计图
        
        Args:
            image_content: 图片数据
            filename: 文件名
            tech_stack: 技术栈 (Android/iOS/Flutter)
            metadata: 元数据
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        file_id = str(uuid.uuid4())
        file_path = None
        
        try:
            logger.info(f"开始处理设计图: {filename}, 技术栈: {tech_stack}, 数据大小: {len(image_content)} 字节")
            
            # 1. 验证图片格式
            try:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(image_content))
                img_format = img.format.lower() if img.format else "unknown"
                img_size = img.size
                logger.info(f"图片尺寸: {img_size[0]}x{img_size[1]}, 格式: {img_format}")
                
                if img_format not in ['jpeg', 'jpg', 'png', 'webp']:
                    logger.error(f"不支持的图片格式: {img_format}")
                    return {
                        "success": False,
                        "error": f"不支持的图片格式: {img_format}",
                        "filename": filename,
                        "tech_stack": tech_stack,
                        "id": file_id
                    }
                logger.info(f"图片格式验证通过: {img_format}")
            except Exception as e:
                logger.error(f"图片格式验证失败: {str(e)}")
                import traceback
                logger.error(f"详细错误信息: {traceback.format_exc()}")
                return {
                    "success": False,
                    "error": f"无效的图片数据: {str(e)}",
                    "filename": filename,
                    "tech_stack": tech_stack,
                    "id": file_id
                }

            # 2. 生成唯一文件名
            ext = Path(filename).suffix
            unique_filename = f"{file_id}{ext}"
            file_path = self.upload_dir / unique_filename
            logger.info(f"生成唯一文件名: {unique_filename}, 完整路径: {file_path}")
            
            # 3. 保存图片
            try:
                with open(file_path, "wb") as f:
                    f.write(image_content)
                logger.info(f"图片已保存: {file_path}")
            except Exception as e:
                logger.error(f"保存图片失败: {str(e)}")
                import traceback
                logger.error(f"详细错误信息: {traceback.format_exc()}")
                return {
                    "success": False,
                    "error": f"保存图片失败: {str(e)}",
                    "filename": filename,
                    "tech_stack": tech_stack,
                    "id": file_id
                }
            
            # 4. 准备元数据
            if metadata is None:
                metadata = {}
                
            full_metadata = {
                **metadata,
                "id": file_id,
                "filename": filename,
                "unique_filename": unique_filename,
                "file_path": str(file_path),
                "tech_stack": tech_stack,
                "upload_time": datetime.now().isoformat(),
                "type": "design_image",
                "image_format": img_format,
                "image_size": len(image_content),
                "image_dimensions": img.size
            }
            logger.info(f"元数据准备完成: {json.dumps(full_metadata, ensure_ascii=False)}")
            
            # 5. 使用多模态模型分析图片
            analysis_text = ""
            try:
                # 检查 vision_model 是否已初始化
                if not hasattr(self, 'vision_model') or self.vision_model is None:
                    logger.error("Vision model未初始化")
                    analysis_text = self._generate_default_analysis(tech_stack, filename)
                else:
                    # 构建详细的分析提示
                    analysis_prompt = self._build_analysis_prompt(tech_stack)
                    
                    logger.info(f"使用模型: gpt-4o, 最大token数: 3000, 温度: 0.3")
                    logger.info(f"API基础URL: {settings.OPENAI_BASE_URL}")
                    
                    # 准备base64编码的图片
                    base64_image = base64.b64encode(image_content).decode('utf-8')
                    logger.info(f"图片base64编码长度: {len(base64_image)}")
                    
                    # 创建请求消息
                    messages = [
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
                                        "url": f"data:image/{img_format};base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ]
                    
                    logger.info("开始发送请求到OpenAI API")
                    try:
                        response = self.vision_model.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=3000,
                            temperature=0.3
                        )
                        logger.info(f"Vision model调用成功: {filename}")
                        logger.info(f"响应ID: {response.id}, 模型: {response.model}, 使用token: {response.usage.total_tokens}")
                        
                        # 验证分析结果
                        analysis_text = response.choices[0].message.content
                        logger.info(f"获取到分析结果，长度: {len(analysis_text)} 字符")
                        
                        if not analysis_text:
                            logger.error(f"分析结果为空: {filename}")
                            analysis_text = self._generate_default_analysis(tech_stack, filename)
                        elif len(analysis_text.strip()) < 100:
                            logger.warning(f"分析结果过短 ({len(analysis_text.strip())}字符): {filename}")
                            analysis_text = self._generate_default_analysis(tech_stack, filename)
                        
                        logger.info(f"成功获取分析结果 ({len(analysis_text.strip())}字符): {filename}")
                    except Exception as api_error:
                        logger.error(f"Vision model API调用失败: {str(api_error)}")
                        import traceback
                        logger.error(f"详细错误信息: {traceback.format_exc()}")
                        analysis_text = self._generate_default_analysis(tech_stack, filename)
            except Exception as e:
                logger.error(f"图片分析失败: {str(e)}")
                import traceback
                logger.error(f"详细错误信息: {traceback.format_exc()}")
                analysis_text = self._generate_default_analysis(tech_stack, filename)
            
            # 6. 添加到向量存储
            try:
                # 将分析结果按类别拆分
                analysis_sections = self._split_analysis_sections(analysis_text)
                logger.info(f"分析结果已按类别拆分: {list(analysis_sections.keys())}")
                
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
            
            logger.info(f"设计图处理完成: {filename}, ID: {file_id}")
            
            return {
                "id": file_id,
                "filename": filename,
                "unique_filename": unique_filename,
                "file_path": str(file_path),
                "tech_stack": tech_stack,
                "analysis": analysis_text,
                "analysis_sections": analysis_sections,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"处理设计图失败: {str(e)}", exc_info=True)
            # 如果文件已经保存，返回文件路径
            file_path_str = str(file_path) if file_path else "未保存"
            return {
                "success": False,
                "error": str(e),
                "filename": filename,
                "tech_stack": tech_stack,
                "id": file_id,
                "file_path": file_path_str
            }
            
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