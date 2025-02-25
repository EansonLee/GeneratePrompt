from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
from pathlib import Path
from src.prompt_optimizer import PromptOptimizer
from src.template_generator import TemplateGenerator
from src.utils.vector_store import VectorStore
from src.file_processor import FileProcessor
from config.config import (
    DEBUG,
    LOG_LEVEL,
    BASE_DIR,
    AGENT_CONFIG,
    TEMPLATE_GENERATION_CONFIG,
    PROMPT_OPTIMIZATION_TEMPERATURE,
    PROMPT_OPTIMIZATION_MAX_TOKENS,
    OPENAI_MODEL
)

# 配置日志级别
import logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# 创建上传目录
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 初始化全局向量存储
try:
    logger.info("初始化全局向量存储...")
    vector_store = VectorStore(use_mock=False)
    logger.info("全局向量存储初始化成功")
except Exception as e:
    logger.error(f"全局向量存储初始化失败: {str(e)}")
    if DEBUG:
        logger.debug("详细错误信息:", exc_info=True)
    logger.info("使用mock数据初始化向量存储...")
    vector_store = VectorStore(use_mock=True)

app = FastAPI(
    title="Prompt生成优化API",
    description="提供Prompt模板生成和优化的API服务",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化文件处理器
file_processor = FileProcessor(UPLOAD_DIR, vector_store)

class PromptRequest(BaseModel):
    prompt: str
    context_files: Optional[List[str]] = None

class TemplateRequest(BaseModel):
    context_files: Optional[List[str]] = None

class ConfirmTemplateRequest(BaseModel):
    template: str

class ConfirmPromptRequest(BaseModel):
    optimized_prompt: str

@app.post("/api/generate-template")
async def generate_template(request: TemplateRequest) -> Dict[str, Any]:
    """生成模板的API
    
    Args:
        request: 请求体
        
    Returns:
        Dict[str, Any]: 生成结果
    """
    try:
        logger.info("开始生成模板...")
        
        # 使用全局向量存储实例
        template_generator = TemplateGenerator()
        template_generator.vector_store = vector_store
        
        template = await template_generator.generate()
        logger.info("模板生成成功")
        
        return {
            "status": "success",
            "template": template
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"生成模板失败: {error_msg}")
        if DEBUG:
            logger.debug("详细错误信息:", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/optimize-prompt")
async def optimize_prompt(request: PromptRequest) -> Dict[str, Any]:
    """优化prompt的API
    
    Args:
        request: 请求体，包含原始prompt和上下文文件列表
        
    Returns:
        Dict[str, Any]: 优化结果
    """
    try:
        logger.info("开始优化prompt...")
        
        if not request.prompt:
            raise ValueError("提示词不能为空")
            
        optimizer = PromptOptimizer(vector_store=vector_store)
        optimized_prompt = await optimizer.optimize(request.prompt)
        logger.info("Prompt优化成功")
        
        return {
            "status": "success",
            "optimized_prompt": optimized_prompt
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"优化prompt失败: {error_msg}")
        if DEBUG:
            logger.debug("详细错误信息:", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/upload-context")
async def upload_context(
    file: UploadFile = File(...),
    is_directory: bool = Form(False)
) -> Dict[str, Any]:
    """上传上下文文件的API
    
    Args:
        file: 上传的文件
        is_directory: 是否作为目录处理
        
    Returns:
        Dict[str, Any]: 处理结果
    """
    try:
        logger.info(f"开始处理上传文件: {file.filename}")
        
        # 检查文件类型是否支持
        if not file_processor.is_supported_file(file.filename):
            error_msg = f"不支持的文件类型: {Path(file.filename).suffix}"
            logger.warning(error_msg)
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )
            
        # 保存上传的文件
        file_path = await file_processor.save_upload_file(file)
        logger.info(f"文件已保存到: {file_path}")
        
        try:
            # 处理文件
            result = await file_processor.process_file(file_path, is_directory)
            logger.info("文件处理完成")
            return {
                "status": "success",
                "file_name": file.filename,
                "processing_result": result
            }
        finally:
            # 清理上传的文件
            if file_path.exists():
                os.remove(file_path)
                logger.info(f"临时文件已删除: {file_path}")
                
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"处理上传文件失败: {error_msg}")
        if DEBUG:
            logger.debug("详细错误信息:", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/confirm-template")
async def confirm_template(request: ConfirmTemplateRequest) -> Dict[str, Any]:
    """确认并保存模板到向量数据库
    
    Args:
        request: 请求体，包含确认的模板内容
        
    Returns:
        Dict[str, Any]: 保存结果
    """
    try:
        logger.info("开始保存模板到向量数据库...")
        
        # 验证模板格式
        template_generator = TemplateGenerator()
        is_valid, missing_fields = template_generator.validate_template(request.template)
        
        if not is_valid:
            raise ValueError(f"模板格式无效，缺少以下字段：{', '.join(missing_fields)}")
        
        # 保存到向量数据库
        vector_store.add_template(request.template)
        
        # 验证保存是否成功
        if not vector_store.verify_insertion(request.template, "templates"):
            raise ValueError("模板保存验证失败")
            
        logger.info("模板保存成功")
        return {
            "status": "success",
            "message": "模板已成功保存到向量数据库"
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"保存模板失败: {error_msg}")
        if DEBUG:
            logger.debug("详细错误信息:", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/confirm-prompt")
async def confirm_prompt(request: ConfirmPromptRequest) -> Dict[str, Any]:
    """确认并保存优化后的prompt到向量数据库
    
    Args:
        request: 请求体，包含确认的优化后prompt内容
        
    Returns:
        Dict[str, Any]: 保存结果
    """
    try:
        logger.info("开始保存优化后的prompt到向量数据库...")
        
        if not request.optimized_prompt:
            raise ValueError("优化后的prompt内容不能为空")
        
        # 保存到向量数据库
        vector_store.add_template(request.optimized_prompt)
        
        # 验证保存是否成功
        if not vector_store.verify_insertion(request.optimized_prompt, "templates"):
            raise ValueError("优化后的prompt保存验证失败")
            
        logger.info("优化后的prompt保存成功")
        return {
            "status": "success",
            "message": "优化后的prompt已成功保存到向量数据库"
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"保存优化后的prompt失败: {error_msg}")
        if DEBUG:
            logger.debug("详细错误信息:", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/health")
async def health_check():
    """健康检查API"""
    try:
        return {
            "status": "healthy",
            "debug_mode": DEBUG,
            "log_level": LOG_LEVEL
        }
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        if DEBUG:
            logger.debug("详细错误信息:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vector-db-status")
async def get_vector_db_status() -> Dict[str, Any]:
    """获取向量数据库状态的API
    
    Returns:
        Dict[str, Any]: 状态信息
    """
    try:
        is_ready = vector_store.is_initialized()
        error = vector_store.get_initialization_error() if not is_ready else None
        
        return {
            "status": "ready" if is_ready else "initializing",
            "error": error
        }
    except Exception as e:
        logger.error(f"获取向量数据库状态失败: {str(e)}")
        if DEBUG:
            logger.debug("详细错误信息:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=DEBUG) 