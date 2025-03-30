import logging
import time
import json
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser

from src.models.schemas import (
    DesignPromptResponse, 
    DesignAnalysisResult,
    PromptMetadata
)
from src.models.enums import DesignType, TechStack
from src.utils.vector_store import VectorStore
from config.config import settings

logger = logging.getLogger(__name__)

class DesignPromptAgent:
    """设计提示生成Agent"""
    
    def __init__(self):
        """初始化设计提示生成Agent"""
        self.vector_store = None
        self.llm = ChatOpenAI(
            model_name=settings.GPT_MODEL,
            temperature=0.7,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # 加载系统提示模板
        self.system_template = self._load_system_template()
        logger.info("设计提示生成Agent初始化完成")
    
    def _load_system_template(self) -> str:
        """加载系统提示模板
        
        Returns:
            str: 系统提示模板
        """
        template_path = Path(settings.TEMPLATES_DIR) / "design_prompt_system.txt"
        
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"加载系统提示模板失败: {str(e)}")
            # 使用默认模板
            return """你是一个专业的设计提示生成专家，擅长将设计图转换为高质量的开发提示。
分析提供的设计图信息，生成详细且优化的开发提示。
确保提示考虑了技术栈要求、设计风格、布局、交互流程和代码风格。

## 设计分析
{design_analysis}

## 技术栈
{tech_stack}

## 设计类型
{design_type}

## 目标受众
{target_audience}

## 用户需求
{user_requirements}

## 相关上下文
{relevant_context}

## 当前日期
{current_date}

生成一个详细且优化的开发提示，包括：
1. 整体布局和结构
2. 颜色和样式规范
3. 组件层次结构
4. 交互和动画效果
5. 响应式设计考虑
6. 技术实现建议
7. 代码组织和最佳实践"""
    
    async def generate_design_prompt(
        self,
        design_analysis: Optional[Dict[str, Any]] = None,
        project_analysis: Optional[Dict[str, Any]] = None,
        tech_stack: Optional[TechStack] = None,
        design_type: Optional[DesignType] = DesignType.UI,
        target_audience: Optional[str] = None,
        user_requirements: Optional[str] = None,
        max_token_count: int = 4000,
        temperature: float = 0.7
    ) -> DesignPromptResponse:
        """生成设计提示
        
        Args:
            design_analysis: 设计分析结果
            project_analysis: 项目分析结果
            tech_stack: 技术栈
            design_type: 设计类型
            target_audience: 目标受众
            user_requirements: 用户需求
            max_token_count: 最大token数量
            temperature: 生成温度
            
        Returns:
            DesignPromptResponse: 生成的提示响应
        """
        try:
            start_time = time.time()
            logger.info(f"开始生成设计提示，技术栈: {tech_stack}, 设计类型: {design_type}")
            
            # 设置生成参数
            self.llm.temperature = temperature
            
            # 准备输入数据
            input_data = {
                "design_analysis": self._format_design_analysis(design_analysis) if design_analysis else "未提供设计分析",
                "project_analysis": self._format_project_analysis(project_analysis) if project_analysis else "未提供项目分析",
                "tech_stack": tech_stack.value if tech_stack else "未指定",
                "design_type": design_type.value if design_type else DesignType.UI.value,
                "target_audience": target_audience if target_audience else "未指定",
                "user_requirements": user_requirements if user_requirements else "未指定",
                "current_date": time.strftime("%Y-%m-%d")
            }
            
            # 获取相关上下文（如果有向量存储）
            if self.vector_store:
                # 构建查询
                query = f"设计类型: {input_data['design_type']}, 技术栈: {input_data['tech_stack']}"
                if design_analysis and isinstance(design_analysis, dict):
                    if "elements" in design_analysis:
                        elements = design_analysis.get("elements", [])
                        element_text = ", ".join([e.get("name", "") for e in elements[:5]])
                        query += f", 元素: {element_text}"
                    if "colors" in design_analysis:
                        colors = design_analysis.get("colors", [])
                        color_text = ", ".join([c.get("hex", "") for c in colors[:3]])
                        query += f", 颜色: {color_text}"
                        
                # 获取相关上下文
                relevant_context = await self.vector_store.get_relevant_context(
                    query=query,
                    max_tokens=max_token_count // 2
                )
                
                if relevant_context:
                    input_data["relevant_context"] = relevant_context
                else:
                    input_data["relevant_context"] = "未找到相关上下文"
            else:
                input_data["relevant_context"] = "向量存储未初始化"
            
            # 构建系统提示
            system_prompt = self.system_template.format(**input_data)
            
            # 构建用户提示
            user_prompt = self._build_user_prompt(input_data)
            
            # 创建LLM链
            chain = LLMChain(
                llm=self.llm,
                prompt=PromptTemplate.from_template(system_prompt),
                output_parser=StrOutputParser()
            )
            
            # 生成提示
            result = await chain.ainvoke({"user_prompt": user_prompt})
            prompt = self._post_process_prompt(result)
            
            # 创建响应
            processing_time = time.time() - start_time
            meta = PromptMetadata(
                model=settings.GPT_MODEL,
                tech_stack=tech_stack.value if tech_stack else None,
                design_type=design_type.value if design_type else DesignType.UI.value,
                processing_time=processing_time,
                token_count=None  # 可以添加token计数逻辑
            )
            
            return DesignPromptResponse(
                prompt=prompt,
                meta=meta
            )
            
        except Exception as e:
            logger.error(f"生成设计提示失败: {str(e)}", exc_info=True)
            # 返回错误响应
            meta = PromptMetadata(
                model=settings.GPT_MODEL,
                tech_stack=tech_stack.value if tech_stack else None,
                design_type=design_type.value if design_type else DesignType.UI.value,
                processing_time=0.0,
                token_count=None,
                error=str(e)
            )
            return DesignPromptResponse(
                prompt="生成提示时出错: " + str(e),
                meta=meta
            )
    
    def _format_design_analysis(self, analysis: Optional[Dict[str, Any]]) -> str:
        """格式化设计分析结果
        
        Args:
            analysis: 设计分析结果
            
        Returns:
            str: 格式化后的设计分析结果
        """
        if not analysis:
            return "未提供设计分析"
            
        try:
            # 提取关键信息
            elements = analysis.get("elements", [])
            colors = analysis.get("colors", [])
            fonts = analysis.get("fonts", [])
            layout = analysis.get("layout", {})
            
            # 格式化元素
            elements_text = ""
            for i, element in enumerate(elements[:10]):  # 限制显示前10个元素
                element_type = element.get("type", "未知类型")
                element_name = element.get("name", f"元素{i+1}")
                element_desc = element.get("description", "无描述")
                elements_text += f"- {element_name} ({element_type}): {element_desc}\n"
            
            # 格式化颜色
            colors_text = ""
            for i, color in enumerate(colors[:5]):  # 限制显示前5个颜色
                hex_code = color.get("hex", "#000000")
                color_name = color.get("name", f"颜色{i+1}")
                colors_text += f"- {color_name}: {hex_code}\n"
            
            # 格式化字体
            fonts_text = ""
            for i, font in enumerate(fonts[:3]):  # 限制显示前3个字体
                font_name = font.get("name", f"字体{i+1}")
                font_style = font.get("style", "常规")
                fonts_text += f"- {font_name} ({font_style})\n"
            
            # 组合结果
            result = f"""## 设计分析结果

### 元素 (前10个)
{elements_text if elements_text else "未检测到元素"}

### 颜色 (前5个)
{colors_text if colors_text else "未检测到颜色"}

### 字体 (前3个)
{fonts_text if fonts_text else "未检测到字体"}

### 布局信息
- 结构: {layout.get("structure", "未检测到布局结构")}
- 对齐方式: {layout.get("alignment", "未检测到对齐方式")}
- 间距: {layout.get("spacing", "未检测到间距信息")}
"""
            return result
            
        except Exception as e:
            logger.error(f"格式化设计分析结果失败: {str(e)}")
            return f"设计分析结果格式化失败: {str(e)}"
    
    def _format_project_analysis(self, analysis: Optional[Dict[str, Any]]) -> str:
        """格式化项目分析结果
        
        Args:
            analysis: 项目分析结果
            
        Returns:
            str: 格式化后的项目分析结果
        """
        if not analysis:
            return "未提供项目分析"
            
        try:
            # 提取关键信息
            tech_stack = analysis.get("tech_stack", {})
            components = analysis.get("components", {})
            key_files = analysis.get("key_files", [])
            summary = analysis.get("summary", "未提供摘要")
            
            # 格式化技术栈
            tech_stack_text = ""
            for category, techs in tech_stack.items():
                tech_list = ", ".join(techs[:5])  # 限制每个类别前5个技术
                tech_stack_text += f"- {category}: {tech_list}\n"
            
            # 格式化组件
            components_text = ""
            for category, comps in components.items():
                comp_list = ", ".join(comps[:5])  # 限制每个类别前5个组件
                components_text += f"- {category}: {comp_list}\n"
            
            # 格式化关键文件
            key_files_text = ""
            for i, file in enumerate(key_files[:10]):  # 限制显示前10个关键文件
                file_path = file.get("path", f"文件{i+1}")
                file_type = file.get("type", "未知类型")
                file_desc = file.get("description", "无描述")
                key_files_text += f"- {file_path} ({file_type}): {file_desc}\n"
            
            # 组合结果
            result = f"""## 项目分析结果

### 摘要
{summary}

### 技术栈
{tech_stack_text if tech_stack_text else "未检测到技术栈"}

### 组件
{components_text if components_text else "未检测到组件"}

### 关键文件 (前10个)
{key_files_text if key_files_text else "未检测到关键文件"}
"""
            return result
            
        except Exception as e:
            logger.error(f"格式化项目分析结果失败: {str(e)}")
            return f"项目分析结果格式化失败: {str(e)}"
    
    def _build_user_prompt(self, input_data: Dict[str, Any]) -> str:
        """构建用户提示
        
        Args:
            input_data: 输入数据
            
        Returns:
            str: 用户提示
        """
        user_prompt = f"""请根据以下信息生成高质量的开发提示:

技术栈: {input_data.get('tech_stack', '未指定')}
设计类型: {input_data.get('design_type', '未指定')}
目标受众: {input_data.get('target_audience', '未指定')}
用户需求: {input_data.get('user_requirements', '未指定')}

请根据提供的设计分析和项目分析生成详细、结构化的开发提示。确保提示考虑到设计细节和代码风格。
"""
        return user_prompt
    
    def _post_process_prompt(self, prompt: str) -> str:
        """后处理生成的提示
        
        Args:
            prompt: 生成的提示
            
        Returns:
            str: 处理后的提示
        """
        # 简单清理
        prompt = prompt.strip()
        
        # 这里可以添加更多的后处理逻辑
        
        return prompt 