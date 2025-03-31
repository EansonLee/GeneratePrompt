"""统一API入口模块"""

import logging
import asyncio
import uvicorn
import os
import sys
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel, Field
from datetime import datetime
import json
import traceback

# 导入路由模块
from src.api.endpoints import design_prompt

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
from src.utils.cache_manager import CacheManager
from src.file_processor import FileProcessor
from src.prompt_optimizer import PromptOptimizer
from src.utils.design_image_processor import DesignImageProcessor
from src.utils.local_project_processor import LocalProjectProcessor
from src.utils.rag_manager import RAGManager
from src.agents.design_prompt_agent import DesignPromptAgent, DesignPromptRequest, ResponseStatus

# 配置日志
logging.basicConfig(**settings.get_logging_config())
logger = logging.getLogger(__name__)

# 创建上传目录
UPLOAD_DIR = settings.UPLOAD_DIR
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 创建应用程序范围的服务实例
vector_store = VectorStore()
cache_manager = CacheManager()
file_processor = FileProcessor(upload_dir=settings.UPLOAD_DIR, vector_store=vector_store)
local_project_processor = LocalProjectProcessor(vector_store=vector_store)
design_image_processor = DesignImageProcessor(vector_store=vector_store)
template_generator = TemplateGenerator()
prompt_optimizer = PromptOptimizer(vector_store=vector_store)

# 创建FastAPI应用
app = FastAPI(
    title=settings.API_CONFIG["title"],
    description=settings.API_CONFIG["description"],
    version=settings.API_CONFIG["version"]
)

# 注册路由
app.include_router(design_prompt.router, prefix="/api", tags=["design_prompt"])

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.API_CONFIG["cors_origins"],
    allow_credentials=True,
    allow_methods=settings.API_CONFIG["cors_methods"],
    allow_headers=settings.API_CONFIG["cors_headers"],
)

# 初始化组件
optimizer = PromptOptimizer(vector_store=vector_store)
rag_manager = RAGManager(vector_store=vector_store)
design_prompt_agent = DesignPromptAgent(
    vector_store=vector_store, 
    design_processor=design_image_processor
)

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
    design_image_path: Optional[str] = Field(None, description="设计图路径")
    user_feedback: Optional[str] = Field(None, description="用户反馈")
    rag_method: str = Field(settings.DESIGN_PROMPT_CONFIG["default_rag_method"], description="RAG方法")
    retriever_top_k: int = Field(settings.DESIGN_PROMPT_CONFIG["default_retriever_top_k"], description="检索结果数量")
    agent_type: str = Field(settings.DESIGN_PROMPT_CONFIG["default_agent_type"], description="Agent类型")
    temperature: float = Field(settings.DESIGN_PROMPT_CONFIG["temperature"], description="温度")
    context_window_size: int = Field(settings.DESIGN_PROMPT_CONFIG["default_context_window_size"], description="上下文窗口大小")
    skip_cache: bool = Field(False, description="是否跳过缓存（强制重新生成）")
    prompt: Optional[str] = Field(None, description="提示词")

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

def check_env_config():
    """检查环境配置状态
    
    Returns:
        Dict: 包含环境变量检查结果的字典
    """
    # 检查关键环境变量
    env_check = {
        "OPENAI_API_KEY": "已设置" if settings.OPENAI_API_KEY else "未设置",
        "DESIGN_PROMPT_MODEL": settings.DESIGN_PROMPT_MODEL or "未设置",
        "OPENAI_BASE_URL": settings.OPENAI_BASE_URL or "未设置"
    }
    
    # 检查配置文件
    config_check = {
        "设计提示词模型配置": (
            settings.DESIGN_PROMPT_CONFIG.get("model") is not None or 
            settings.DESIGN_PROMPT_MODEL is not None
        )
    }
    
    # 合并检查结果
    env_check.update(config_check)
    
    # 检查发现的问题
    problems = []
    if not settings.OPENAI_API_KEY:
        problems.append("OPENAI_API_KEY未设置")
    
    if not settings.DESIGN_PROMPT_MODEL and not settings.DESIGN_PROMPT_CONFIG.get("model"):
        problems.append("设计提示词模型未在配置中设置")
        
    # 记录检查结果
    logger.info(f"环境变量和配置检查结果: {env_check}")
    if problems:
        logger.warning(f"环境变量和配置检查发现问题: {problems}")
    
    return {
        "status": env_check,
        "problems": problems,
        "all_valid": len(problems) == 0
    }

class DesignImageUploadResponse(BaseModel):
    """设计图上传响应模型"""
    success: bool
    message: str
    image_id: str
    file_name: str
    file_path: str
    file_size: int
    tech_stack: str
    vector_store_success: bool = True

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

@app.post("/api/design/upload", response_model=DesignImageUploadResponse)
async def upload_design_image(
    file: UploadFile = File(...),
    tech_stack: str = Form(...),
):
    """上传设计图
    
    Args:
        file: 设计图文件
        tech_stack: 技术栈
        
    Returns:
        DesignImageUploadResponse: 上传结果
    """
    try:
        # 读取文件内容
        file_content = await file.read()
        
        # 处理设计图
        result = await design_image_processor.process_image(
            file_content=file_content,
            file_name=file.filename,
            tech_stack=tech_stack,
        )
        
        # 检查向量存储状态
        vector_store_success = result.get("vector_store_success", True)
        message = "上传成功"
        if not vector_store_success:
            message = "设计图已保存，但未成功添加到向量数据库，某些功能可能受限"
            logger.warning(f"设计图 {result['id']} 未成功添加到向量数据库")
        
        # 构建响应
        return DesignImageUploadResponse(
            success=True,
            message=message,
            image_id=result["id"],
            file_name=result["file_name"],
            file_path=result["file_path"],
            file_size=result["file_size"],
            tech_stack=tech_stack,
            vector_store_success=vector_store_success,
        )
    except Exception as e:
        logger.error(f"上传设计图失败: {str(e)}")
        if settings.DEBUG:
            logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"上传设计图失败: {str(e)}"
        )

@app.post("/api/design/generate")
async def generate_design_prompt(request: GenerateDesignPromptRequest):
    """生成设计图Prompt
    
    Args:
        request: 设计图Prompt请求，包含技术栈、设计图ID等信息
        
    Returns:
        Dict[str, Any]: 生成的Prompt响应，包含状态、消息、提示词、分析结果等
    """
    start_time = time.time()  # 确保在所有执行路径之前定义start_time
    try:
        # 记录请求基本信息，但不记录完整的base64数据
        design_id_log = request.design_image_id
        if request.design_image_id and request.design_image_id.startswith('data:image/'):
            design_id_log = f"[BASE64_IMAGE]({len(request.design_image_id)} 字符)"
        
        logger.info(f"接收到设计图Prompt生成请求: tech_stack={request.tech_stack}, design_image_id={design_id_log}")
        
        # 验证参数
        if not request.design_image_id:
            raise HTTPException(
                status_code=400,
                detail="设计图ID不能为空"
            )
            
        # 检查design_image_id是否是base64数据
        base64_data = None
        if request.design_image_id.startswith("data:image/"):
            logger.info("检测到design_image_id是base64数据")
            base64_data = request.design_image_id
            
            # 检查base64数据长度
            try:
                # 计算数据长度
                data_length = len(base64_data)
                # 设置最大长度限制为10MB的base64字符串
                max_length = 10 * 1024 * 1024
                if data_length > max_length:
                    logger.warning(f"base64数据过长: {data_length} 字符，超过了 {max_length} 的限制")
                    raise HTTPException(
                        status_code=400,
                        detail=f"图像数据过大，请使用小于10MB的图像"
                    )
                
                # 验证base64格式
                if ',' not in base64_data:
                    logger.warning("base64数据格式不正确，缺少逗号分隔符")
                    raise HTTPException(
                        status_code=400,
                        detail="base64图像数据格式不正确"
                    )
                
                # 提取数据部分，检查是否为有效的base64
                try:
                    import base64
                    header, encoded_data = base64_data.split(',', 1)
                    # 尝试解码一小部分数据，验证是否为有效的base64
                    base64.b64decode(encoded_data[:100].encode('utf-8'))
                    logger.info("base64数据格式有效")
                except Exception as e:
                    logger.warning(f"base64数据解码测试失败: {str(e)}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"无效的base64图像数据"
                    )
            except Exception as e:
                if isinstance(e, HTTPException):
                    raise e
                logger.error(f"处理base64数据时出错: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"处理图像数据失败: {str(e)}"
                )
        
        # 检查技术栈
        if not request.tech_stack or request.tech_stack not in settings.DESIGN_PROMPT_CONFIG.get("supported_tech_stacks", []):
            raise HTTPException(
                status_code=400,
                detail=f"不支持的技术栈: {request.tech_stack}"
            )
        
        # 确保向量存储初始化完成
        from src.utils.vector_store import VectorStore
        vector_store = VectorStore()
        
        # 确保存储属性都可以直接访问
        if hasattr(vector_store, 'fix_stores'):
            vector_store.fix_stores()
        if hasattr(vector_store, 'ensure_store_attributes'):
            success = vector_store.ensure_store_attributes()
            if not success:
                logger.warning("向量存储属性确保失败，可能导致检索问题")
        
        # 检查向量存储是否就绪
        if not vector_store.is_ready():
            logger.warning("向量存储未就绪，可能会影响相似设计检索质量")
        
        # 初始化设计图处理器
        from src.utils.design_image_processor import DesignImageProcessor
        design_image_processor = DesignImageProcessor(vector_store=vector_store)
        
        # 初始化LocalProjectProcessor
        from src.utils.local_project_processor import LocalProjectProcessor
        local_project_processor = LocalProjectProcessor(vector_store=vector_store)
        
        # 初始化RAG管理器
        from src.utils.rag_manager import RAGManager
        rag_manager = RAGManager(vector_store=vector_store)
        
        # 初始化设计图Prompt生成Agent
        from src.agents.design_prompt_agent import DesignPromptAgent, DesignPromptRequest
        agent = DesignPromptAgent(
            vector_store=vector_store,
            design_processor=design_image_processor,
            temperature=request.temperature,
            context_window_size=request.context_window_size,
            skip_cache=request.skip_cache
        )
        
        # 初始化Agent状态
        agent._init_state(request.design_image_id)
        
        # 设置技术栈
        agent.state["tech_stack"] = request.tech_stack
        
        # 若有base64数据，临时保存到状态中
        if base64_data:
            agent.state["design_image_base64"] = base64_data
            logger.info("已将base64数据保存到Agent状态中")
        
        # 创建请求对象
        agent_request = DesignPromptRequest(
            tech_stack=request.tech_stack,
            tech_stack_components=None,  # 如果需要，可从request中获取
            agent_type=request.agent_type,
            rag_method=request.rag_method,
            retriever_top_k=request.retriever_top_k,
            temperature=request.temperature,
            context_window_size=request.context_window_size,
            skip_cache=request.skip_cache
        )
        
        # 生成Prompt
        response = await agent.generate_design_prompt(
            design_image_id=request.design_image_id,
            design_prompt_request=agent_request,
            skip_cache=request.skip_cache
        )
        
        logger.info(f"生成设计图Prompt完成，状态: {response.status}, 耗时: {time.time() - start_time:.2f}秒")
        
        # 返回统一结构的响应
        return {
            "status": response.status,
            "message": response.message,
            "prompt": response.prompt,
            "design_analysis": response.design_analysis,
            "project_analysis": response.project_analysis,
            "similar_designs": response.similar_designs,
            "tech_stack": request.tech_stack,
            "processing_time": time.time() - start_time
        }
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"生成设计图Prompt时出错: {str(e)}")
        logger.error(traceback.format_exc())
        
        if isinstance(e, HTTPException):
            raise e
        
        return {
            "status": "failed",
            "message": f"生成设计图Prompt失败: {str(e)}",
            "prompt": "",
            "design_analysis": "",
            "project_analysis": "",
            "similar_designs": "",
            "tech_stack": getattr(request, "tech_stack", ""),
            "processing_time": process_time
        }

@app.post("/api/design/save")
async def save_user_modified_prompt(request: SaveUserModifiedPromptRequest):
    """保存用户修改后的Prompt
    
    Args:
        request: 保存用户修改后的Prompt请求
        
    Returns:
        Dict[str, Any]: 操作结果
    """
    try:
        logger.info(f"保存用户修改后的Prompt，字符数：{len(request.prompt)}")
        
        if not request.prompt:
            return {
                "status": "error",
                "message": "Prompt不能为空",
                "timestamp": datetime.now().isoformat()
            }
        
        # 尝试保存用户修改后的Prompt
        result = await design_prompt_agent.save_user_modified_prompt(
            prompt=request.prompt,
            tech_stack=request.tech_stack,
            design_image_id=request.design_image_id
        )
        
        if result.get("success"):
            return {
                "status": "success",
                "message": "Prompt保存成功",
                "prompt_id": result.get("id"),
                "timestamp": datetime.now().isoformat()
            }
        else:
            # 返回错误信息
            return {
                "status": "error",
                "message": result.get("error", "保存失败，未知错误"),
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"保存用户修改后的Prompt失败: {str(e)}")
        if settings.DEBUG:
            logger.debug("详细错误信息:", exc_info=True)
            
        # 尝试备选方法保存
        try:
            # 如果agent方法失败，尝试直接使用向量存储
            if hasattr(vector_store, 'add_prompt') and callable(vector_store.add_prompt):
                logger.info("尝试使用备选方法保存：vector_store.add_prompt")
                
                # 检查向量存储状态
                if not hasattr(vector_store, 'prompts') or vector_store.prompts is None:
                    logger.warning("prompts向量存储未初始化，先尝试初始化stores")
                    vector_store._init_stores()
                
                doc_id = await vector_store.add_prompt(request.prompt)
                
                if doc_id:
                    return {
                        "status": "partial_success",
                        "message": "通过备选方法保存了Prompt，但可能丢失一些元数据",
                        "prompt_id": doc_id,
                        "timestamp": datetime.now().isoformat()
                    }
            
            # 如果直接向量存储方法也失败，返回错误信息
            return {
                "status": "error",
                "message": f"保存失败: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e2:
            logger.error(f"备选保存方法也失败: {str(e2)}")
            
            return {
                "status": "error",
                "message": f"保存失败: {str(e)}, 备选方法也失败: {str(e2)}",
                "timestamp": datetime.now().isoformat()
            }

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
    """确认优化后的prompt
    
    Args:
        request: 请求体
        
    Returns:
        Dict[str, Any]: 确认结果
    """
    try:
        logger.info("确认优化后的prompt...")
        logger.info(f"请求参数长度: {len(request.optimized_prompt)} 字符")
        
        if not request.optimized_prompt:
            return {
                "status": "error",
                "message": "优化后的prompt不能为空",
                "timestamp": datetime.now().isoformat()
            }
        
        # 检查向量存储状态
        if not hasattr(vector_store, 'prompts') or vector_store.prompts is None:
            logger.warning("prompts向量存储未初始化，先尝试初始化stores")
            vector_store._init_stores()
            
            # 再次检查
            if not hasattr(vector_store, 'prompts') or vector_store.prompts is None:
                return {
                    "status": "error",
                    "message": "prompts向量存储未初始化，无法保存",
                    "timestamp": datetime.now().isoformat()
                }
        
        # 保存到向量数据库
        doc_id = await vector_store.add_prompt(request.optimized_prompt)
        
        # 检查是否成功保存
        if not doc_id:
            logger.warning("未能成功保存prompt（无文档ID返回）")
            return {
                "status": "warning",
                "message": "保存可能未完全成功，但应用可以继续使用",
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "status": "success",
            "message": "已确认并保存优化后的prompt",
            "doc_id": doc_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"确认优化后的prompt失败: {str(e)}")
        if settings.DEBUG:
            logger.debug("详细错误信息:", exc_info=True)
        
        # 尝试降级处理
        try:
            logger.info("尝试降级处理：使用add_prompt_history方法")
            # 尝试使用add_prompt_history作为备选方法
            vector_store.add_prompt_history("用户提交的prompt", request.optimized_prompt)
            
            return {
                "status": "partial_success",
                "message": "已通过备选方法保存prompt",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e2:
            logger.error(f"备选保存方法也失败: {str(e2)}")
            
            # 返回友好的错误信息
            return {
                "status": "error",
                "message": f"确认失败: {str(e)}",
                "detail": str(e),
                "code": "INTERNAL_ERROR",
                "timestamp": datetime.now().isoformat()
            }

@app.get("/api/vector-db-status", response_model=Dict[str, Any])
async def get_vector_db_status():
    """获取向量存储状态
    
    Returns:
        Dict[str, Any]: 向量存储状态信息
    """
    try:
        # 检查向量存储是否已初始化和就绪
        is_initialized = vector_store.is_initialized()
        is_ready = vector_store.is_ready()
        
        # 获取向量存储的基本信息
        status_info = {
            "is_initialized": is_initialized,
            "is_ready": is_ready,
            "status": "ready" if is_ready else "initializing",
            "timestamp": datetime.now().isoformat(),
            "stores": {},
            "diagnostic": {
                "has_embeddings": hasattr(vector_store, '_embeddings') and vector_store._embeddings is not None,
                "has_contexts": hasattr(vector_store, 'contexts') and vector_store.contexts is not None,
                "has_templates": hasattr(vector_store, 'templates') and vector_store.templates is not None,
                "has_prompts": hasattr(vector_store, 'prompts') and vector_store.prompts is not None,
                "has_designs": hasattr(vector_store, 'designs') and vector_store.designs is not None,
                "embedding_model": settings.EMBEDDING_MODEL,
                "vision_model": settings.VISION_MODEL,
                "storage_dir": str(getattr(vector_store, 'storage_dir', 'unknown'))
            }
        }
        
        # 如果有初始化错误，添加到状态信息中
        error = vector_store.get_initialization_error()
        if error:
            status_info["error"] = error
            status_info["status"] = "error"
        
        # 如果已初始化，获取更详细的信息
        if is_initialized:
            try:
                # 获取存储统计信息
                store_stats = await vector_store.get_store_stats()
                status_info["stores"] = store_stats
                
                # 获取嵌入模型信息
                if hasattr(vector_store, "_embeddings") and vector_store._embeddings is not None:
                    status_info["embedding_model"] = {
                        "available": True,
                        "model": settings.EMBEDDING_MODEL
                    }
                else:
                    status_info["embedding_model"] = {
                        "available": False,
                        "reason": "嵌入模型未初始化"
                    }
                    
                # 获取视觉模型信息
                if hasattr(vector_store, "_vision_model") and vector_store._vision_model is not None:
                    status_info["vision_model"] = {
                        "available": True,
                        "model": settings.VISION_MODEL
                    }
                else:
                    status_info["vision_model"] = {
                        "available": False,
                        "reason": "视觉模型未初始化"
                    }
                
                # 检查是否所有存储都可用，并提供详细状态
                all_stores_available = True
                for store_type in ["contexts", "templates", "prompts", "designs"]:
                    if not (hasattr(vector_store, store_type) and getattr(vector_store, store_type) is not None):
                        all_stores_available = False
                        break
                
                if not all_stores_available:
                    status_info["status"] = "partial"
                    status_info["message"] = "部分向量存储未初始化，但系统可以运行"
                
            except Exception as e:
                logger.error(f"获取向量存储详细信息失败: {str(e)}")
                status_info["detail_error"] = str(e)
                if settings.DEBUG:
                    status_info["detail_traceback"] = traceback.format_exc()
        
        return status_info
        
    except Exception as e:
        logger.error(f"获取向量存储状态失败: {str(e)}")
        if settings.DEBUG:
            logger.debug("详细错误信息:", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"获取向量存储状态失败: {str(e)}"
        )

def ensure_directories():
    """确保所有必要的目录存在"""
    directories = [
        settings.DATA_DIR,
        settings.LOG_DIR,
        settings.UPLOAD_DIR,
        Path(settings.VECTOR_DB_PATH)
    ]
    
    for directory in directories:
        directory_path = Path(directory)
        try:
            directory_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"确保目录存在: {directory_path}")
        except Exception as e:
            logger.error(f"创建目录失败 {directory_path}: {str(e)}")

# 添加启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时执行的操作"""
    # 确保所有必要的目录存在
    ensure_directories()
    
    # 记录配置信息
    logger.info(f"加载配置信息，服务启动中...")
    logger.info(f"当前项目根目录: {settings.BASE_DIR}")
    logger.info(f"调试模式: {settings.DEBUG}")
    logger.info(f"日志级别: {settings.LOG_LEVEL}")
    logger.info(f"OpenAI模型: {settings.OPENAI_MODEL}")
    logger.info(f"嵌入模型: {settings.EMBEDDING_MODEL}")
    logger.info(f"向量存储路径: {settings.VECTOR_DB_PATH}")
    
    # 检查向量存储状态
    try:
        # 确保向量存储已初始化
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # 检查vector_store是否已正确初始化
                if hasattr(vector_store, 'stores') and vector_store.stores:
                    logger.info("向量存储初始化成功")
                    
                    # 修复存储属性
                    if hasattr(vector_store, 'fix_stores'):
                        logger.info("检查并修复向量存储属性...")
                        vector_store.fix_stores()
                        
                    # 确保所有属性直接可访问
                    if hasattr(vector_store, 'ensure_store_attributes'):
                        logger.info("确保所有存储属性可直接访问...")
                        vector_store.ensure_store_attributes()
                        
                    # 验证存储是否就绪
                    if hasattr(vector_store, 'is_ready') and vector_store.is_ready():
                        logger.info("向量存储已就绪")
                    else:
                        logger.warning("向量存储未就绪，尝试修复")
                        
                    # 查看存储状态
                    if hasattr(vector_store, 'get_store_stats'):
                        try:
                            stats = await vector_store.get_store_stats()
                            logger.info(f"向量存储统计: {stats}")
                        except Exception as stats_err:
                            logger.error(f"获取向量存储统计失败: {str(stats_err)}")
                    
                    break
                else:
                    logger.warning(f"向量存储初始化尝试 {attempt + 1}/{max_retries} 失败")
                    if attempt < max_retries - 1:
                        logger.info(f"等待 {retry_delay} 秒后重试...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避
            except Exception as e:
                logger.error(f"向量存储初始化尝试 {attempt + 1}/{max_retries} 出错: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"等待 {retry_delay} 秒后重试...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    logger.error("向量存储初始化失败，应用将以有限功能运行")
    except Exception as e:
        logger.error(f"向量存储初始化过程中发生错误: {str(e)}")
        logger.warning("应用将以有限功能运行，一些依赖向量存储的功能可能不可用")

if __name__ == "__main__":
    # 确保加载最新的环境变量
    load_env_file()
    print(f"DEBUG模式: {settings.DEBUG}")
    print(f"日志级别: {settings.LOG_LEVEL}")
    print(f"OpenAI模型: {settings.OPENAI_MODEL}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 