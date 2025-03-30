# 设计图Prompt生成器

一个基于LLM的设计图解析与Prompt生成工具。该工具可以分析设计图并生成高质量的开发提示词，支持多种技术栈。

## 主要功能

- **设计图解析**：自动分析上传的UI设计图，识别UI元素、布局和交互
- **Prompt生成**：基于设计图和技术栈生成详细的开发提示词
- **多技术栈支持**：支持Android、iOS和Flutter等多种技术栈
- **提示词优化**：利用RAG技术提高生成的Prompt质量
- **历史提示词利用**：自动检索和利用类似设计的历史提示词

## 项目架构

```
├── src/                # 源代码
│   ├── api/            # API接口
│   ├── agents/         # 代理模块
│   ├── utils/          # 工具类
│   │   ├── design_image_processor.py  # 设计图处理器
│   │   ├── local_project_processor.py # 本地项目处理器
│   │   └── vector_store.py            # 向量存储
│   ├── file_processor.py              # 文件处理
│   └── prompt_optimizer.py            # 提示词优化器
├── frontend/           # 前端代码
│   └── src/
│       ├── components/ # React组件
│       │   └── DesignPromptGenerator.tsx # 设计图Prompt生成器组件
│       └── api/        # API客户端
├── config/             # 配置文件
└── data/               # 数据存储
```

## 技术栈

### 后端

- **FastAPI**: Web框架
- **LangChain**: LLM应用开发框架
- **LangGraph**: 多步骤Agent工作流引擎
- **OpenAI API**: LLM服务
- **Chroma**: 向量数据库

### 前端

- **React**: UI框架
- **Ant Design**: 组件库
- **TypeScript**: 类型安全的JavaScript

## 环境要求

- Python 3.9+
- Node.js 16+
- Git

## 安装与运行指南

### 克隆项目

```bash
git clone <repository-url>
cd design-prompt-generator
```

### 环境配置

1. **创建并配置.env文件**

   在项目根目录创建`.env`文件，参考以下内容：

   ```
   # OpenAI配置
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_BASE_URL=https://api.openai.com/v1  # 可选，如果使用代理

   # 模型配置
   OPENAI_MODEL=gpt-4
   EMBEDDING_MODEL=text-embedding-3-small

   # 视觉模型配置
   VISION_MODEL=gpt-4-vision-preview
   VISION_MODEL_TEMPERATURE=0.3
   VISION_MODEL_MAX_TOKENS=3000

   # 设计提示词生成配置
   DESIGN_PROMPT_MODEL=gpt-4
   DESIGN_PROMPT_TEMPERATURE=0.7
   DESIGN_PROMPT_MAX_TOKENS=3500
   OPENAI_TIMEOUT=60.0

   # 应用配置
   DEBUG=true
   LOG_LEVEL=DEBUG
   ```

2. **创建必要的目录**

   ```bash
   mkdir -p data/vector_store uploads logs
   ```

### 运行方式1：使用启动脚本（推荐）

项目提供了便捷的启动脚本，可以自动完成虚拟环境创建、依赖安装和服务启动。

**Windows系统**：
```
start.bat
```

**macOS/Linux系统**：
```
chmod +x start.sh
./start.sh
```

启动脚本会自动：
1. 检查Python和Node.js环境
2. 创建并激活Python虚拟环境
3. 安装后端和前端依赖
4. 启动后端和前端服务

### 运行方式2：手动步骤

如果启动脚本无法正常工作，您可以按照以下步骤手动启动：

1. **设置Python虚拟环境**

   ```bash
   # 创建虚拟环境
   python -m venv venv
   
   # 激活虚拟环境（Windows）
   venv\Scripts\activate
   
   # 激活虚拟环境（macOS/Linux）
   source venv/bin/activate
   ```

2. **安装后端依赖**

   ```bash
   pip install -r requirements.txt
   ```

3. **启动后端服务**

   ```bash
   python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **安装前端依赖（新开一个终端）**

   ```bash
   cd frontend
   npm install
   ```

5. **启动前端服务**

   ```bash
   npm start
   ```

### 运行方式3：使用Python启动脚本

项目提供了Python启动脚本，可以在一个命令中完成配置检查和服务启动：

```bash
python scripts/start.py
```

## 服务访问

启动成功后，可通过以下地址访问：

- **前端界面**：http://localhost:3000
- **API文档**：http://localhost:8000/docs
- **API接口**：http://localhost:8000/api/...

## 使用流程

1. 通过前端界面上传设计图
2. 选择技术栈（Android/iOS/Flutter/React/Vue）
3. 设置生成参数（可选）
4. 点击"生成Prompt"
5. 查看生成的Prompt，可以直接使用或编辑后保存
6. 复制生成的Prompt用于后续开发

## 常见问题排查

1. **端口占用问题**
   
   如果8000或3000端口被占用，您可以修改端口：
   - 后端：修改启动命令中的`--port`参数
   - 前端：修改`frontend/package.json`中的`start`脚本，添加`PORT=3001`环境变量

2. **API密钥问题**
   
   确保在`.env`文件中设置了有效的`OPENAI_API_KEY`，否则大部分功能将无法使用。

3. **依赖安装问题**
   
   如果某些依赖安装失败，可以尝试：
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --no-cache-dir
   ```

## 高级功能

### 本地项目分析

本地项目分析功能可以分析本地项目代码，提取代码的上下文信息，用于生成更精准的提示词：

1. **智能抽样**: 
   - 分析目录结构，确定关键文件
   - 检测技术栈相关文件（如package.json, pubspec.yaml等）
   - 识别UI组件和模型文件
2. **并行处理**: 并行处理文件内容
3. **增量分析**: 使用文件哈希缓存分析结果

## 贡献

欢迎贡献代码！请确保代码符合项目的代码风格和测试标准。

## 许可证

MIT License 