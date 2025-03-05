"""主程序入口模块"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

from config.config import settings
from src.prompt_optimizer import PromptOptimizer
from src.utils.vector_store import VectorStore
from src.utils.rag_manager import RAGManager

# 配置日志
logging.basicConfig(**settings.get_logging_config())
logger = logging.getLogger(__name__)

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
    allow_methods=settings.API_CONFIG["cors_methods"],
    allow_headers=settings.API_CONFIG["cors_headers"]
)

# 初始化组件
vector_store = VectorStore()
optimizer = PromptOptimizer(vector_store=vector_store)
rag_manager = RAGManager(vector_store=vector_store)

# 请求模型
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

# API路由
@app.post("/api/optimize")
async def optimize_prompt(request: OptimizePromptRequest):
    """优化提示词
    
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
            if not _is_valid_file(file):
                continue
                
            # 保存文件
            file_path = await _save_upload_file(file)
            
            # 处理文件
            result = await _process_file(file_path)
            results.append(result)
            
            # 添加后台清理任务
            if background_tasks:
                background_tasks.add_task(_cleanup_file, file_path)
                
        return {
            "status": "success",
            "processed": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"处理文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """获取统计信息
    
    Returns:
        Dict[str, Any]: 统计信息
    """
    try:
        vector_stats = vector_store.get_stats()
        context_stats = await rag_manager.get_context_stats()
        
        return {
            "vector_store": vector_stats,
            "context": context_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 工具函数
def _is_valid_file(file: UploadFile) -> bool:
    """验证文件是否有效
    
    Args:
        file: 上传的文件
        
    Returns:
        bool: 是否有效
    """
    # 检查文件扩展名
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.FILE_PROCESSING_CONFIG["supported_extensions"]:
        logger.warning(f"不支持的文件类型: {ext}")
        return False
        
    # 检查MIME类型
    if file.content_type not in settings.FILE_PROCESSING_CONFIG["allowed_mime_types"]:
        logger.warning(f"不支持的MIME类型: {file.content_type}")
        return False
        
    return True

async def _save_upload_file(file: UploadFile) -> Path:
    """保存上传的文件
    
    Args:
        file: 上传的文件
        
    Returns:
        Path: 保存的文件路径
    """
    # 生成保存路径
    save_dir = Path(settings.UPLOAD_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"{timestamp}_{file.filename}"
    
    # 保存文件
    content = await file.read()
    save_path.write_bytes(content)
    
    return save_path

async def _process_file(file_path: Path) -> Dict[str, Any]:
    """处理文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        Dict[str, Any]: 处理结果
    """
    try:
        # 读取文件内容
        content = file_path.read_text(encoding='utf-8')
        
        # 添加到RAG上下文
        await rag_manager.add_context(
            content=content,
            content_type="file",
            metadata={
                "file_name": file_path.name,
                "file_type": file_path.suffix
            }
        )
        
        return {
            "file_name": file_path.name,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"处理文件失败 {file_path}: {str(e)}")
        return {
            "file_name": file_path.name,
            "status": "failed",
            "error": str(e)
        }

async def _cleanup_file(file_path: Path):
    """清理临时文件
    
    Args:
        file_path: 文件路径
    """
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.error(f"清理文件失败 {file_path}: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    try:
        # 等待向量存储就绪
        if not await vector_store.wait_until_ready():
            raise Exception("向量存储初始化失败")
            
        logger.info("应用启动成功")
        
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理"""
    try:
        # 保存向量存储
        await vector_store.save()
        logger.info("应用关闭，数据已保存")
        
    except Exception as e:
        logger.error(f"应用关闭失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    ) 