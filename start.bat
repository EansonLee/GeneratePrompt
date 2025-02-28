@echo off
echo 启动服务...

REM 检查Python是否安装
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python未安装，请先安装Python
    exit /b 1
)

REM 检查Node.js是否安装
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo Node.js未安装，请先安装Node.js
    exit /b 1
)

REM 检查虚拟环境
if not exist "venv" (
    echo 创建虚拟环境...
    python -m venv venv
)

REM 激活虚拟环境
call venv\Scripts\activate

REM 安装依赖
echo 安装Python依赖...
pip install -r requirements.txt

echo 安装前端依赖...
cd frontend
call npm install
cd ..

REM 运行启动脚本
python scripts/start.py

pause 