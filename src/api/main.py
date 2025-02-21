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

# 创建上传目录
UPLOAD_DIR = Path("uploads")
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
vector_store = VectorStore()
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
        generator = TemplateGenerator()
        template = generator.generate()
        return {"status": "success", "template": template}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/optimize-prompt")
async def optimize_prompt(request: PromptRequest):
    """优化prompt的API"""
    try:
        optimizer = PromptOptimizer()
        optimized = optimizer.optimize(request.prompt)
        return {"status": "success", "optimized_prompt": optimized}
    except Exception as e:
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
        # 检查文件类型是否支持
        if not file_processor.is_supported_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {Path(file.filename).suffix}"
            )
            
        # 保存上传的文件
        file_path = await file_processor.save_upload_file(file)
        
        try:
            # 处理文件
            result = await file_processor.process_file(file_path, is_directory)
            return {
                "status": "success",
                "file_name": file.filename,
                "processing_result": result
            }
        finally:
            # 清理上传的文件
            if file_path.exists():
                os.remove(file_path)
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """健康检查API"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 