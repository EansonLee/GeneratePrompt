import logging
import time
import asyncio
import os
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query, BackgroundTasks

from src.models.schemas import (
    DesignPromptRequest, 
    DesignPromptResponse, 
    DesignAnalysisResult,
    ErrorResponse,
    LocalProjectAnalysisResult
)
from src.agents.design_prompt_agent import DesignPromptAgent
from src.utils.design_image_processor import DesignImageProcessor
from src.file_processor import FileProcessor
from src.utils.local_project_processor import LocalProjectProcessor
from src.utils.vector_store import VectorStore
from src.models.enums import DesignType, TechStack
from config.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/generate",
    response_model=DesignPromptResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="基于设计图生成优化提示",
    description="上传设计图，生成针对该设计的优化提示"
)
async def generate_prompt(
    background_tasks: BackgroundTasks,
    design_file: Optional[UploadFile] = File(None, description="设计图文件（支持PNG, JPG, PDF, Figma URL）"),
    tech_stack: Optional[TechStack] = Form(None, description="技术栈"),
    design_type: Optional[DesignType] = Form(DesignType.UI, description="设计类型"),
    target_audience: Optional[str] = Form(None, description="目标受众"),
    user_requirements: Optional[str] = Form(None, description="用户需求"),
    max_token_count: Optional[int] = Form(4000, description="最大token数量"),
    temperature: Optional[float] = Form(0.7, description="生成温度")
) -> DesignPromptResponse:
    """基于设计图生成优化提示
    
    Args:
        design_file: 设计图文件
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
        
        # 检查设计文件是否提供
        if not design_file:
            raise HTTPException(
                status_code=400, 
                detail="设计文件需要提供"
            )
            
        # 初始化分析结果
        design_analysis_result = None
        
        # 初始化任务，用于记录任务状态
        tasks = {
            "design_analysis": {"status": "pending", "result": None}
        }
        
        # 处理设计文件（如果提供）
        if design_file:
            # 处理设计文件
            file_processor = FileProcessor()
            file_content = await file_processor.process(design_file)
            
            # 分析设计
            logger.info(f"开始分析设计: {design_file.filename}")
            tasks["design_analysis"]["status"] = "running"
            
            analyzer = DesignImageProcessor()
            # 使用process_image方法处理图片内容
            design_analysis_result = await analyzer.process_image(
                file_content=file_content, 
                file_name=design_file.filename,
                tech_stack=str(tech_stack) if tech_stack else None
            )
            
            tasks["design_analysis"]["status"] = "completed"
            tasks["design_analysis"]["result"] = design_analysis_result
            logger.info(f"设计分析完成: {design_file.filename}")
        
        vector_store = VectorStore()
        
        # 生成提示
        prompt_agent = DesignPromptAgent()
        
        prompt_response = await prompt_agent.generate_design_prompt(
            design_analysis=design_analysis_result,
            project_analysis=None,
            tech_stack=tech_stack,
            design_type=design_type,
            target_audience=target_audience,
            user_requirements=user_requirements,
            max_token_count=max_token_count,
            temperature=temperature
        )
        
        processing_time = time.time() - start_time
        
        # 构建响应
        response = DesignPromptResponse(
            prompt=prompt_response.prompt,
            meta=prompt_response.meta,
            processing_time=processing_time,
            design_analysis=design_analysis_result,
            tasks_status=tasks  # 添加任务状态到响应中
        )
        
        # 添加后台任务保存向量存储
        if vector_store:
            background_tasks.add_task(vector_store.save)
            
        return response
        
    except Exception as e:
        logger.error(f"生成设计提示失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"生成设计提示失败: {str(e)}"
        )

@router.post(
    "/analyze-local-project",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="分析本地项目",
    description="分析本地项目文件夹路径，提取代码上下文信息，用于后续提示生成"
)
async def analyze_local_project(
    background_tasks: BackgroundTasks,
    local_project_path: str = Form(..., description="本地项目文件夹路径"),
    git_repo_branch: str = Form("main", description="Git仓库分支（如果是Git项目）")
) -> Dict[str, Any]:
    """分析本地项目文件夹，提取代码上下文信息
    
    Args:
        background_tasks: 后台任务
        local_project_path: 本地项目文件夹路径
        git_repo_branch: Git仓库分支（如果是Git项目）
        
    Returns:
        Dict[str, Any]: 分析结果，包含项目摘要和处理时间
    """
    try:
        start_time = time.time()
        
        logger.info(f"开始分析本地项目: {local_project_path}")
        
        # 校验路径
        if not local_project_path or not os.path.exists(local_project_path):
            raise HTTPException(
                status_code=400,
                detail=f"项目路径不存在: {local_project_path}"
            )
        
        if not os.path.isdir(local_project_path):
            raise HTTPException(
                status_code=400,
                detail=f"指定的路径不是文件夹: {local_project_path}"
            )
        
        # 初始化向量存储
        vector_store = VectorStore()
        
        # 处理本地项目
        project_processor = LocalProjectProcessor(vector_store=vector_store)
        
        # 分析本地项目
        analysis_result = await project_processor.analyze_project(
            project_path=local_project_path,
            branch=git_repo_branch  # 如果是Git项目可能会用到
        )
        
        processing_time = time.time() - start_time
        
        # 添加后台任务保存向量存储
        if vector_store:
            background_tasks.add_task(vector_store.save)
        
        # 返回结果
        if analysis_result.get("status") != "success":
            logger.warning(f"本地项目分析失败: {analysis_result.get('error')}")
            return {
                "status": "error",
                "error": analysis_result.get("error"),
                "processing_time": processing_time
            }
        else:
            logger.info(f"本地项目分析完成: {local_project_path}, 耗时: {processing_time:.2f}秒")
            return {
                "status": "success",
                "project_analysis": analysis_result,
                "summary": project_processor.get_project_summary(analysis_result),
                "processing_time": processing_time
            }
            
    except HTTPException as e:
        # 直接重新抛出HTTP异常
        raise e
    except Exception as e:
        logger.error(f"分析本地项目失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"分析本地项目失败: {str(e)}"
        ) 