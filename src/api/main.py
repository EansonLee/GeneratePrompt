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

# 初始化向量存储和文件处理器
try:
    logger.info("初始化向量存储...")
    vector_store = VectorStore(use_mock=False)
    logger.info("向量存储初始化成功")
except Exception as e:
    logger.error(f"向量存储初始化失败: {str(e)}")
    if DEBUG:
        logger.debug("详细错误信息:", exc_info=True)
    logger.info("使用mock数据初始化向量存储...")
    vector_store = VectorStore(use_mock=True)

file_processor = FileProcessor(UPLOAD_DIR, vector_store)

class PromptRequest(BaseModel):
    prompt: str
    context_files: Optional[List[str]] = None

class TemplateRequest(BaseModel):
    context_files: Optional[List[str]] = None

@app.post("/api/generate-template")
async def generate_template(request: TemplateRequest):
    """生成模板prompt的API"""
    try:
        logger.info("开始生成模板...")
        generator = TemplateGenerator(
            model_name=OPENAI_MODEL,
            temperature=TEMPLATE_GENERATION_CONFIG["temperature"],
            max_tokens=TEMPLATE_GENERATION_CONFIG["max_tokens"],
            max_contexts=TEMPLATE_GENERATION_CONFIG["max_contexts"],
            max_templates=TEMPLATE_GENERATION_CONFIG["max_templates"]
        )
        template = generator.generate()
        logger.info("模板生成成功")
        return {"status": "success", "template": template}
    except Exception as e:
        logger.error(f"生成模板失败: {str(e)}")
        if DEBUG:
            logger.debug("详细错误信息:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/optimize-prompt")
async def optimize_prompt(request: PromptRequest):
    """优化prompt的API"""
    try:
        logger.info("开始优化prompt...")
        optimizer = PromptOptimizer(
            temperature=PROMPT_OPTIMIZATION_TEMPERATURE,
            max_tokens=PROMPT_OPTIMIZATION_MAX_TOKENS
        )
        optimized = optimizer.optimize(request.prompt)
        logger.info("Prompt优化成功")
        return {"status": "success", "optimized_prompt": optimized}
    except Exception as e:
        logger.error(f"优化prompt失败: {str(e)}")
        if DEBUG:
            logger.debug("详细错误信息:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=DEBUG) 