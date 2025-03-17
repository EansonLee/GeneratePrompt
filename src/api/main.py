"""统一API入口模块"""

import logging
import asyncio
import uvicorn
import os
import sys
from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime

# 添加项目根目录到Python路径
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(project_root)

# 加载.env文件中的环境变量
def load_env_file():
    """加载.env文件中的环境变量"""
    env_path = Path(project_root) / '.env'
    if env_path.exists():
        print(f"从.env文件加载配置: {env_path}")
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
                    except ValueError:
                        print(f"警告: 无法解析环境变量行: {line}")
    else:
        print("未找到.env文件，将使用默认配置")

# 加载环境变量
load_env_file()

from config.config import settings
from src.template_generator import TemplateGenerator
from src.utils.vector_store import VectorStore
from src.utils.rag_manager import RAGManager
from src.file_processor import FileProcessor
from src.prompt_optimizer import PromptOptimizer
from src.utils.design_image_processor import DesignImageProcessor
from src.agents.design_prompt_agent import DesignPromptAgent

# 配置日志
logging.basicConfig(**settings.get_logging_config())
logger = logging.getLogger(__name__)

# 创建上传目录
UPLOAD_DIR = settings.UPLOAD_DIR
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 初始化全局向量存储
try:
    logger.info("初始化全局向量存储...")
    vector_store = VectorStore(use_mock=False)
    logger.info("全局向量存储初始化成功")
except Exception as e:
    logger.error(f"全局向量存储初始化失败: {str(e)}")
    if settings.DEBUG:
        logger.debug("详细错误信息:", exc_info=True)
    logger.info("使用mock数据初始化向量存储...")
    vector_store = VectorStore(use_mock=True)

# 创建FastAPI应用
app = FastAPI(
    title=settings.API_CONFIG["title"],
    description=settings.API_CONFIG["description"],
    version=settings.API_CONFIG["version"]
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.API_CONFIG["cors_origins"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # 预检请求的缓存时间
)

# 初始化组件
optimizer = PromptOptimizer(vector_store=vector_store)
rag_manager = RAGManager(vector_store=vector_store)
file_processor = FileProcessor(UPLOAD_DIR, vector_store)
design_image_processor = DesignImageProcessor(vector_store=vector_store)
design_prompt_agent = DesignPromptAgent(vector_store=vector_store, design_processor=design_image_processor)

# 请求模型定义
class OptimizePromptRequest(BaseModel):
    """优化提示词请求"""
    prompt: str
    context: Optional[Dict[str, Any]] = None
    file_ids: Optional[List[str]] = None

class AddContextRequest(BaseModel):
    """添加上下文请求"""
    content: str
    content_type: str
    metadata: Optional[Dict[str, Any]] = None

class ProcessFileRequest(BaseModel):
    """处理文件请求"""
    file_path: str
    file_type: Optional[str] = None

class GenerateDesignPromptRequest(BaseModel):
    """生成设计图Prompt请求"""
    tech_stack: str = Field(..., description="技术栈 (Android/iOS/Flutter)")
    design_image_id: str = Field(..., description="设计图ID")
    design_image_path: str = Field(..., description="设计图路径")
    rag_method: str = Field(settings.DESIGN_PROMPT_CONFIG["default_rag_method"], description="RAG方法")
    retriever_top_k: int = Field(settings.DESIGN_PROMPT_CONFIG["default_retriever_top_k"], description="检索结果数量")
    agent_type: str = Field(settings.DESIGN_PROMPT_CONFIG["default_agent_type"], description="Agent类型")
    temperature: float = Field(settings.DESIGN_PROMPT_CONFIG["temperature"], description="温度")
    context_window_size: int = Field(settings.DESIGN_PROMPT_CONFIG["default_context_window_size"], description="上下文窗口大小")

class SaveUserModifiedPromptRequest(BaseModel):
    """保存用户修改后的Prompt请求"""
    prompt: str = Field(..., description="修改后的Prompt")
    tech_stack: str = Field(..., description="技术栈")
    design_image_id: str = Field(..., description="设计图ID")

class PromptRequest(BaseModel):
    """优化prompt请求模型"""
    prompt: str
    context_files: Optional[List[str]] = None

class TemplateRequest(BaseModel):
    """模板生成请求模型"""
    project_type: Optional[str] = None
    project_description: Optional[str] = None
    context_files: Optional[List[str]] = None

class ConfirmTemplateRequest(BaseModel):
    """确认模板请求模型"""
    template: str

class ConfirmPromptRequest(BaseModel):
    """确认优化后的prompt请求模型"""
    optimized_prompt: str

# API路由
@app.post("/api/optimize")
async def optimize_prompt_with_rag(request: OptimizePromptRequest):
    """优化提示词（使用RAG增强）
    
    Args:
        request: 优化请求
        
    Returns:
        Dict[str, Any]: 优化结果
    """
    try:
        # 1. 使用RAG增强提示词
        enhanced = await rag_manager.enhance_prompt(request.prompt)
        
        # 2. 优化增强后的提示词
        result = await optimizer.optimize(
            prompt=enhanced["enhanced_prompt"],
            context=request.context
        )
        
        # 3. 合并结果
        return {
            **result,
            "rag_info": enhanced
        }
        
    except Exception as e:
        logger.error(f"优化提示词失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/optimize-prompt")
async def optimize_prompt(request: PromptRequest) -> Dict[str, Any]:
    """优化prompt的API（兼容旧版）
    
    Args:
        request: 请求体，包含原始prompt和上下文文件列表
        
    Returns:
        Dict[str, Any]: 优化结果
    """
    try:
        logger.info("开始优化prompt...")
        logger.info(f"请求参数: {request.dict()}")
        
        if not request.prompt:
            raise HTTPException(
                status_code=422,
                detail={
                    "status": "error",
                    "message": "提示词不能为空",
                    "code": "EMPTY_PROMPT"
                }
            )
            
        # 等待向量存储就绪
        if not await vector_store.wait_until_ready():
            error = vector_store.get_initialization_error()
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "error",
                    "message": f"向量存储未就绪: {error}",
                    "code": "VECTOR_STORE_NOT_READY"
                }
            )
            
        result = await optimizer.optimize(request.prompt)
        
        # 检查优化结果
        if not result or not result.get("optimized_prompt"):
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "error",
                    "message": "优化失败，未能生成优化后的提示词",
                    "code": "OPTIMIZATION_FAILED"
                }
            )
            
        return {
            "status": "success",
            "original_prompt": request.prompt,
            "optimized_prompt": result["optimized_prompt"],
            "optimization_steps": result.get("optimization_steps", []),
            "optimization_time": result.get("optimization_time", 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"优化prompt失败: {str(e)}")
        if settings.DEBUG:
            logger.debug("详细错误信息:", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"优化失败: {str(e)}",
                "code": "INTERNAL_ERROR"
            }
        )

@app.post("/api/context")
async def add_context(request: AddContextRequest):
    """添加上下文内容
    
    Args:
        request: 添加上下文请求
        
    Returns:
        Dict[str, str]: 操作结果
    """
    try:
        await rag_manager.add_context(
            content=request.content,
            content_type=request.content_type,
            metadata=request.metadata
        )
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"添加上下文失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/files")
async def process_files(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """处理上传的文件
    
    Args:
        files: 上传的文件列表
        background_tasks: 后台任务
        
    Returns:
        Dict[str, Any]: 处理结果
    """
    try:
        results = []
        
        for file in files:
            # 验证文件类型
            if not file_processor.is_supported_file(file.filename):
                continue
                
            # 保存文件
            file_content = await file.read()
            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            # 处理文件
            result = await file_processor.process_file(str(file_path))
            results.append(result)
            
        return {
            "status": "success",
            "processed_files": results
        }
        
    except Exception as e:
        logger.error(f"处理文件失败: {str(e)}")
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
            
        # 检查文件大小
        file_content = await file.read()
        file_size = len(file_content)
        if file_size > settings.FILE_PROCESSING_CONFIG["max_file_size"]:
            error_msg = f"文件大小超过限制: {file_size} > {settings.FILE_PROCESSING_CONFIG['max_file_size']}"
            logger.warning(error_msg)
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )
            
        # 保存上传的文件
        try:
            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as f:
                f.write(file_content)
            logger.info(f"文件已保存到: {file_path}")
        except Exception as e:
            error_msg = f"保存文件失败: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )
            
        # 处理文件
        try:
            result = await file_processor.process_file(str(file_path))
            logger.info(f"文件处理结果: {result}")
            return {
                "status": "success",
                "message": "文件上传并处理成功",
                "file_id": result["file_id"],
                "file_name": file.filename,
                "file_path": str(file_path),
                "file_size": file_size,
                "chunks": result.get("chunks", 0),
                "processing_time": result.get("processing_time", 0)
            }
        except Exception as e:
            error_msg = f"处理文件失败: {str(e)}"
            logger.error(error_msg)
            if settings.DEBUG:
                logger.debug("详细错误信息:", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"上传文件失败: {str(e)}"
        logger.error(error_msg)
        if settings.DEBUG:
            logger.debug("详细错误信息:", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

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
        logger.info(f"请求参数: {request.dict()}")
        
        # 使用全局向量存储实例
        template_generator = TemplateGenerator()
        template_generator.vector_store = vector_store
        
        # 等待向量存储就绪
        if not await vector_store.wait_until_ready():
            error = vector_store.get_initialization_error()
            raise ValueError(f"向量存储未就绪: {error}")
        
        # 调用异步方法生成模板
        result = await template_generator.generate_template(
            project_type=request.project_type,
            project_description=request.project_description,
            context_files=request.context_files
        )
        
        return {
            "status": "success",
            "template": result["template"],
            "generation_time": result.get("generation_time", 0),
            "token_usage": result.get("token_usage", {})
        }
        
    except ValueError as e:
        logger.error(f"生成模板失败 (ValueError): {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": str(e),
                "code": "VALIDATION_ERROR"
            }
        )
    except Exception as e:
        logger.error(f"生成模板失败: {str(e)}")
        if settings.DEBUG:
            logger.debug("详细错误信息:", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"生成模板失败: {str(e)}",
                "code": "INTERNAL_ERROR"
            }
        )

@app.post("/api/design/upload")
async def upload_design_image(
    file: UploadFile = File(...),
    tech_stack: str = Form(...)
):
    """上传设计图
    
    Args:
        file: 上传的设计图文件
        tech_stack: 技术栈
        
    Returns:
        Dict[str, Any]: 处理结果
    """
    try:
        # 验证文件类型
        if not design_image_processor.is_supported_image(file.filename):
            raise ValueError(f"不支持的图片格式: {file.filename}")
            
        # 读取文件内容
        content = await file.read()
        
        # 保存并处理设计图
        result = await design_image_processor.process_image(
            image_content=content,
            filename=file.filename,
            tech_stack=tech_stack
        )
        
        return {
            "status": "success",
            "image_id": result["image_id"],
            "image_path": result["image_path"],
            "tech_stack": tech_stack,
            "width": result.get("width", 0),
            "height": result.get("height", 0),
            "processing_time": result.get("processing_time", 0)
        }
        
    except ValueError as e:
        logger.error(f"上传设计图失败 (ValueError): {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"上传设计图失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/design/generate")
async def generate_design_prompt(request: GenerateDesignPromptRequest):
    """生成设计图Prompt
    
    Args:
        request: 生成设计图Prompt请求
        
    Returns:
        Dict[str, Any]: 生成结果
    """
    try:
        # 生成设计图Prompt
        result = await design_prompt_agent.generate_prompt(
            tech_stack=request.tech_stack,
            design_image_id=request.design_image_id,
            design_image_path=request.design_image_path,
            rag_method=request.rag_method,
            retriever_top_k=request.retriever_top_k,
            agent_type=request.agent_type,
            temperature=request.temperature,
            context_window_size=request.context_window_size
        )
        
        return {
            "status": "success",
            "prompt": result["prompt"],
            "generation_time": result.get("generation_time", 0),
            "token_usage": result.get("token_usage", {}),
            "rag_info": result.get("rag_info", {})
        }
        
    except ValueError as e:
        logger.error(f"生成设计图Prompt失败 (ValueError): {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"生成设计图Prompt失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/design/save")
async def save_user_modified_prompt(request: SaveUserModifiedPromptRequest):
    """保存用户修改后的Prompt
    
    Args:
        request: 保存用户修改后的Prompt请求
        
    Returns:
        Dict[str, str]: 操作结果
    """
    try:
        # 保存用户修改后的Prompt
        await design_prompt_agent.save_user_modified_prompt(
            prompt=request.prompt,
            tech_stack=request.tech_stack,
            design_image_id=request.design_image_id
        )
        
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"保存用户修改后的Prompt失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/confirm-template")
async def confirm_template(request: ConfirmTemplateRequest) -> Dict[str, Any]:
    """确认模板的API
    
    Args:
        request: 请求体
        
    Returns:
        Dict[str, Any]: 确认结果
    """
    try:
        logger.info("确认模板...")
        
        # 这里可以添加模板确认的逻辑，例如保存到数据库或向量存储
        # 目前只是简单返回成功
        
        return {
            "status": "success",
            "message": "模板已确认",
            "template": request.template
        }
        
    except Exception as e:
        logger.error(f"确认模板失败: {str(e)}")
        if settings.DEBUG:
            logger.debug("详细错误信息:", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"确认模板失败: {str(e)}",
                "code": "INTERNAL_ERROR"
            }
        )

@app.post("/api/confirm-prompt")
async def confirm_prompt(request: ConfirmPromptRequest) -> Dict[str, Any]:
    """确认优化后的prompt的API
    
    Args:
        request: 请求体
        
    Returns:
        Dict[str, Any]: 确认结果
    """
    try:
        logger.info("确认优化后的prompt...")
        
        # 这里可以添加prompt确认的逻辑，例如保存到数据库或向量存储
        # 目前只是简单返回成功
        
        return {
            "status": "success",
            "message": "优化后的prompt已确认",
            "optimized_prompt": request.optimized_prompt
        }
        
    except Exception as e:
        logger.error(f"确认优化后的prompt失败: {str(e)}")
        if settings.DEBUG:
            logger.debug("详细错误信息:", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"确认优化后的prompt失败: {str(e)}",
                "code": "INTERNAL_ERROR"
            }
        )

if __name__ == "__main__":
    # 确保加载最新的环境变量
    load_env_file()
    print(f"DEBUG模式: {settings.DEBUG}")
    print(f"日志级别: {settings.LOG_LEVEL}")
    print(f"OpenAI模型: {settings.OPENAI_MODEL}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 