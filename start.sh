#!/bin/bash

echo "启动服务..."

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "Python未安装，请先安装Python"
    exit 1
fi

# 检查Node.js是否安装
if ! command -v node &> /dev/null; then
    echo "Node.js未安装，请先安装Node.js"
    exit 1
fi

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
echo "安装Python依赖..."
pip install -r requirements.txt

echo "安装前端依赖..."
cd frontend
npm install
cd ..

# 运行启动脚本
python3 scripts/start.py 