# Prompt生成优化系统

## 项目简介
这是一个基于FastAPI和LangChain的Prompt生成优化系统，提供模板生成和提示词优化功能。

## 环境要求

- Python 3.9+ ([Python官网](https://www.python.org/downloads/))
- Node.js 16+ ([Node.js官网](https://nodejs.org/))
- Git (可选，[Git官网](https://git-scm.com/downloads))

## 快速开始

### Windows用户

1. 配置OpenAI API密钥：
   在项目根目录创建 `.env` 文件，添加：
   ```
   OPENAI_API_KEY=你的OpenAI API密钥
   ```

2. 启动方式：

   方式一：使用批处理文件（推荐）
   ```bash
   # 直接双击 start.bat 文件
   # 或在命令提示符中运行：
   start.bat
   ```

   方式二：手动运行
   ```bash
   # 1. 打开命令提示符(cmd)或PowerShell
   # 2. 进入项目目录
   cd 项目目录路径

   # 3. 创建并激活虚拟环境
   python -m venv venv
   .\venv\Scripts\activate

   # 4. 安装Python依赖
   pip install -r requirements.txt

   # 5. 安装前端依赖
   cd frontend
   npm install
   cd ..

   # 6. 运行启动脚本
   python scripts/start.py
   ```

### Linux/Mac用户

```bash
# 1. 添加执行权限
chmod +x start.sh

# 2. 运行启动脚本
./start.sh
```

## 配置说明

### 配置文件
- 默认配置已内置在系统中
- 创建 `.env` 文件可覆盖默认配置
- 必须配置 `OPENAI_API_KEY`

### 配置检查
系统包含两个主要脚本：

1. `check_config.py`: 配置检查脚本
   - 检查环境变量设置
   - 验证必要目录
   - 测试OpenAI API连接

2. `start.py`: 启动脚本
   - 设置环境变量
   - 调用配置检查
   - 启动后端和前端服务

### 执行时机

```bash
# 场景1：首次运行或日常使用
start.bat  # Windows
./start.sh  # Linux/Mac

# 场景2：配置排查
python scripts/check_config.py  # 仅检查配置

# 场景3：开发调试
python scripts/start.py  # 直接启动服务
```

## 常见问题解决

### Windows相关

1. 权限问题
```bash
# 以管理员身份运行PowerShell并执行：
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

2. 端口占用
```bash
# 查找占用端口的进程
netstat -ano | findstr :8000  # 后端端口
netstat -ano | findstr :3000  # 前端端口

# 结束进程
taskkill /F /PID 进程ID
```

### Linux/Mac相关

```bash
# 端口占用处理
lsof -i :8000  # 检查后端端口
lsof -i :3000  # 检查前端端口
kill $(lsof -t -i:8000)  # 结束后端进程
kill $(lsof -t -i:3000)  # 结束前端进程
```

## 项目结构

```
项目根目录
├── frontend/          # 前端代码
├── src/              # 后端源码
│   ├── agents/       # 代理实现
│   ├── api/          # API实现
│   └── utils/        # 工具函数
├── scripts/          # 脚本目录
│   ├── check_config.py  # 配置检查脚本
│   └── start.py        # 启动脚本
├── data/             # 数据目录（自动创建）
├── logs/             # 日志目录（自动创建）
├── uploads/          # 上传文件目录（自动创建）
├── start.bat         # Windows启动脚本
├── start.sh          # Linux/Mac启动脚本
└── .env              # 环境配置文件（需要创建）
```

## 开发指南

### 1. 配置管理
- 修改配置后建议先运行 `check_config.py` 验证
- 开发新功能时先确保配置正确
- 可以在 `.env` 中覆盖任何默认配置

### 2. 启动服务
```bash
# 完整启动（推荐）
start.bat  # Windows
./start.sh  # Linux/Mac

# 分步启动（调试用）
python scripts/check_config.py  # 检查配置
python scripts/start.py        # 启动服务
```

### 3. API文档
启动服务后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- 前端页面: http://localhost:3000

## 注意事项

1. 确保Python和Node.js已添加到系统环境变量
2. 如果使用代理，在 `.env` 中设置 `OPENAI_BASE_URL`
3. 首次运行可能需要等待依赖安装
4. 不要提交 `.env` 文件到版本控制系统
5. 定期备份数据目录

## 调试提示

1. 检查配置：
```bash
python scripts/check_config.py
```

2. 查看日志：
- 检查 `logs` 目录下的日志文件
- 查看命令行输出的错误信息

3. 单独启动服务：
```bash
# 前端
cd frontend
npm start

# 后端
cd src/api
uvicorn main:app --reload
```

如果遇到问题，系统会提供详细的错误信息和解决建议。 