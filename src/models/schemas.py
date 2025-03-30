from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import time
from datetime import datetime

from src.models.enums import DesignType, TechStack

# 错误响应
class ErrorResponse(BaseModel):
    """错误响应"""
    detail: str = Field(..., description="错误详情")

# 设计提示请求
class DesignPromptRequest(BaseModel):
    """设计提示请求"""
    design_type: Optional[DesignType] = Field(DesignType.UI, description="设计类型")
    tech_stack: Optional[TechStack] = Field(None, description="技术栈")
    target_audience: Optional[str] = Field(None, description="目标受众")
    user_requirements: Optional[str] = Field(None, description="用户需求")
    max_tokens: Optional[int] = Field(4000, description="最大token数量")
    temperature: Optional[float] = Field(0.7, description="生成温度")

# 设计分析结果
class DesignElement(BaseModel):
    """设计元素"""
    name: str = Field(..., description="元素名称")
    type: str = Field(..., description="元素类型")
    description: Optional[str] = Field(None, description="元素描述")
    position: Optional[Dict[str, Any]] = Field(None, description="元素位置")
    size: Optional[Dict[str, Any]] = Field(None, description="元素大小")
    attributes: Optional[Dict[str, Any]] = Field(None, description="元素属性")

class DesignColor(BaseModel):
    """设计颜色"""
    name: Optional[str] = Field(None, description="颜色名称")
    hex: str = Field(..., description="颜色十六进制值")
    rgb: Optional[Dict[str, int]] = Field(None, description="RGB值")
    usage: Optional[str] = Field(None, description="颜色用途")

class DesignFont(BaseModel):
    """设计字体"""
    name: str = Field(..., description="字体名称")
    style: Optional[str] = Field(None, description="字体样式")
    size: Optional[str] = Field(None, description="字体大小")
    usage: Optional[str] = Field(None, description="字体用途")

class DesignLayout(BaseModel):
    """设计布局"""
    structure: Optional[str] = Field(None, description="布局结构")
    alignment: Optional[str] = Field(None, description="对齐方式")
    spacing: Optional[str] = Field(None, description="间距")
    grid: Optional[Dict[str, Any]] = Field(None, description="网格信息")

class DesignAnalysisResult(BaseModel):
    """设计分析结果"""
    id: str = Field(..., description="分析ID")
    design_type: str = Field(..., description="设计类型")
    summary: str = Field(..., description="摘要")
    elements: List[DesignElement] = Field([], description="元素列表")
    colors: List[DesignColor] = Field([], description="颜色列表")
    fonts: List[DesignFont] = Field([], description="字体列表")
    layout: DesignLayout = Field(DesignLayout(), description="布局信息")
    interactions: Optional[List[Dict[str, Any]]] = Field(None, description="交互信息")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")

# 本地项目分析结果
class LocalProjectAnalysisResult(BaseModel):
    """本地项目分析结果"""
    status: str = Field(..., description="状态")
    project_path: str = Field(..., description="项目路径")
    tech_stack: Dict[str, List[str]] = Field({}, description="技术栈")
    components: Dict[str, List[str]] = Field({}, description="组件")
    key_files: List[Dict[str, Any]] = Field([], description="关键文件")
    summary: str = Field(..., description="摘要")
    timestamp: float = Field(..., description="时间戳")
    processing_time: float = Field(..., description="处理时间")
    error: Optional[str] = Field(None, description="错误信息")

# 提示元数据
class PromptMetadata(BaseModel):
    """提示元数据"""
    model: Optional[str] = Field(None, description="模型")
    tech_stack: Optional[str] = Field(None, description="技术栈")
    design_type: Optional[str] = Field(None, description="设计类型")
    processing_time: float = Field(..., description="处理时间")
    token_count: Optional[int] = Field(None, description="Token数量")
    error: Optional[str] = Field(None, description="错误信息")

# 设计提示响应
class DesignPromptResponse(BaseModel):
    """设计提示响应"""
    prompt: str = Field(..., description="生成的提示")
    meta: PromptMetadata = Field(..., description="元数据")
    processing_time: Optional[float] = Field(None, description="处理时间")
    design_analysis: Optional[Dict[str, Any]] = Field(None, description="设计分析结果")
    tasks_status: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="任务状态信息")
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "基于上传的UI设计图，创建一个React应用...",
                "meta": {
                    "model": "gpt-4",
                    "tech_stack": "React",
                    "design_type": "UI",
                    "processing_time": 2.5,
                    "token_count": 1500
                },
                "processing_time": 3.2,
                "design_analysis": {
                    "id": "design-123",
                    "design_type": "UI",
                    "summary": "现代化的电子商务界面..."
                },
                "tasks_status": {
                    "design_analysis": {"status": "completed", "result": {}}
                }
            }
        } 