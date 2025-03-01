# Prompt生成优化系统

## 项目简介
这是一个基于FastAPI和LangChain的Prompt生成优化系统，提供模板生成和提示词优化功能。

## 环境要求

- Python 3.9+ ([Python官网](https://www.python.org/downloads/))
- Node.js 16+ ([Node.js官网](https://nodejs.org/))
- pip (Python包管理器)
- npm (Node.js包管理器)
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
   # 或在PowerShell中运行：
   .\start.bat
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

1. 添加执行权限并运行：
   ```bash
   # 方法1：使用bash直接执行
   bash start.sh
   
   # 方法2：添加执行权限后运行
   chmod +x start.sh
   ./start.sh
   ```

2. 如果遇到行尾符问题（从Windows复制到Linux时）：
   ```bash
   # 安装dos2unix
   sudo apt-get install dos2unix  # Ubuntu/Debian
   sudo yum install dos2unix      # CentOS/RHEL
   
   # 修复行尾符
   dos2unix start.sh
   ```

## 启动脚本说明

### 脚本功能

1. 环境检查：
   - 检查Python和Node.js是否安装
   - 验证必要的命令是否可用

2. 虚拟环境管理：
   - 自动创建Python虚拟环境（如果不存在）
   - 自动激活虚拟环境
   - Windows下在 `venv\Scripts\`
   - Linux下在 `venv/bin/`

3. 依赖安装：
   - 自动安装Python依赖（requirements.txt）
   - 自动安装前端依赖（package.json）

4. 服务启动：
   - 启动后端服务（FastAPI）
   - 启动前端服务（React）

### 日志和调试

1. 查看运行日志：
   ```bash
   # Windows
   start.bat > log.txt 2>&1
   
   # Linux/Mac
   ./start.sh > log.txt 2>&1
   ```

2. 调试模式运行：
   ```bash
   # Windows PowerShell
   $env:DEBUG=1; .\start.bat
   
   # Linux/Mac
   DEBUG=1 ./start.sh
   ```

### 常见问题解决

1. Python相关：
   ```bash
   # Linux安装Python虚拟环境包
   sudo apt-get install python3-venv  # Ubuntu/Debian
   sudo yum install python3-venv      # CentOS/RHEL
   
   # 检查Python位置
   which python3  # Linux/Mac
   where python   # Windows
   ```

2. Node.js相关：
   ```bash
   # 检查Node.js和npm版本
   node --version
   npm --version
   
   # 清除npm缓存
   npm cache clean --force
   ```

3. 权限问题：
   ```bash
   # Windows PowerShell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   
   # Linux/Mac
   sudo chmod +x start.sh
   ```

4. 端口占用：
   ```bash
   # Windows
   netstat -ano | findstr :8000  # 检查后端端口
   netstat -ano | findstr :3000  # 检查前端端口
   taskkill /F /PID 进程ID
   
   # Linux/Mac
   lsof -i :8000  # 检查后端端口
   lsof -i :3000  # 检查前端端口
   kill $(lsof -t -i:8000)  # 结束后端进程
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
├── requirements.txt   # Python依赖
└── .env              # 环境配置文件（需要创建）
```

## 开发指南

### 1. 开发环境准备

1. IDE推荐：
   - VS Code
   - PyCharm
   - WebStorm（前端开发）

2. VS Code插件推荐：
   - Python
   - Pylance
   - ESLint
   - Prettier
   - GitLens

3. 代码风格：
   - Python: PEP 8
   - JavaScript: ESLint + Prettier
   - 使用4空格缩进

### 2. 配置管理

1. 环境变量：
   ```bash
   # .env 文件示例
   OPENAI_API_KEY=你的API密钥
   OPENAI_BASE_URL=https://api.openai.com/v1  # 可选
   DEBUG=True  # 开发模式
   LOG_LEVEL=DEBUG  # 日志级别
   ```

2. 配置优先级：
   - 环境变量 > .env文件 > 默认配置
   - 修改配置后运行检查：
     ```bash
     python scripts/check_config.py
     ```

### 3. 开发工作流

1. 代码更新：
   ```bash
   # 拉取最新代码
   git pull
   
   # 安装新依赖
   pip install -r requirements.txt
   cd frontend && npm install && cd ..
   ```

2. 分步调试：
   ```bash
   # 后端调试（支持热重载）
   cd src/api
   uvicorn main:app --reload --port 8000
   
   # 前端调试（支持热重载）
   cd frontend
   npm start
   ```

3. 日志查看：
   - 运行时日志：`logs/app.log`
   - 错误日志：`logs/error.log`
   - API访问日志：`logs/access.log`

### 4. API文档

1. 在线文档（服务启动后）：
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

2. API测试：
   - 使用Swagger UI进行在线测试
   - 使用Postman进行本地测试
   - curl命令示例在文档中提供

### 5. 故障排除

1. 服务无法启动：
   - 检查端口占用
   - 验证环境变量
   - 查看错误日志

2. API调用失败：
   - 确认服务状态
   - 检查API密钥
   - 验证请求格式

3. 前端页面问题：
   - 清除浏览器缓存
   - 检查Console错误
   - 验证API连接

## 注意事项

1. 安全性：
   - 不要提交 `.env` 文件
   - 定期更新依赖
   - 注意API密钥保护

2. 性能优化：
   - 避免大量并发请求
   - 合理设置超时时间
   - 注意内存使用

3. 开发建议：
   - 遵循代码规范
   - 编写单元测试
   - 及时提交代码

4. 环境维护：
   - 定期更新依赖
   - 备份重要数据
   - 监控服务状态

## 贡献指南

1. 提交PR流程：
   - Fork项目
   - 创建特性分支
   - 提交变更
   - 发起Pull Request

2. 代码要求：
   - 遵循项目代码风格
   - 添加必要的注释
   - 更新相关文档

3. 文档维护：
   - 更新README
   - 添加注释
   - 编写使用示例

## 支持和帮助

- 提交Issue报告问题
- 查看Wiki获取详细指南
- 通过Discussions讨论功能特性

## 许可证

MIT License - 详见 LICENSE 文件 