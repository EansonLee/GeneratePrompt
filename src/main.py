"""
警告：此文件已被弃用，请使用 src/api/main.py 作为API入口点。
所有功能已被合并到 src/api/main.py 中。
"""

"""应用入口文件 - 重定向到 api/main.py"""
import logging
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# 首先加载环境变量
from src.utils.env_loader import load_env_file
load_env_file()

# 直接导入api.main中的app对象
from src.api.main import app

if __name__ == "__main__":
    import uvicorn
    from config.config import settings
    uvicorn.run(
        "src.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=settings.DEBUG
    ) 