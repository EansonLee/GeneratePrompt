"""
警告：此文件已被弃用，请使用 src/api/main.py 作为API入口点。
所有功能已被合并到 src/api/main.py 中。
"""

"""应用入口文件"""
import logging
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# 首先加载环境变量
from src.utils.env_loader import load_env_file
load_env_file()

# 然后导入其他模块
from config.config import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import register_routes
from src.utils.vector_store import VectorStore
from src.utils.design_image_processor import DesignImageProcessor

# 配置日志
logging.basicConfig(**settings.get_logging_config())
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """创建FastAPI应用"""
    logger.info("正在创建FastAPI应用...")
    logger.info(f"调试模式: {settings.DEBUG}")
    logger.info(f"日志级别: {settings.LOG_LEVEL}")
    logger.info(f"使用的视觉模型: {settings.VISION_MODEL}")
    
    # 创建FastAPI应用
    app = FastAPI(
        title="设计图分析API",
        description="用于分析设计图并生成代码的API",
        version="1.0.0",
        debug=settings.DEBUG
    )
    
    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由
    register_routes(app)
    
    # 初始化向量存储
    try:
        vector_store = VectorStore()
        logger.info("向量存储初始化成功")
    except Exception as e:
        logger.error(f"向量存储初始化失败: {str(e)}")
    
    # 添加启动事件
    @app.on_event("startup")
    async def startup_event():
        logger.info("应用启动...")
        # 验证配置
        logger.info(f"视觉模型: {settings.VISION_MODEL}")
        logger.info(f"视觉模型温度: {settings.VISION_MODEL_CONFIG['temperature']}")
        logger.info(f"视觉模型最大tokens: {settings.VISION_MODEL_CONFIG['max_tokens']}")
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=settings.DEBUG
    ) 