# Prompt生成优化系统

一个基于AI的提示词生成和优化系统，支持多种文件格式的上下文处理和向量化存储。

## 功能特点

- 🤖 基于OpenAI的提示词优化
- 📝 提示词模板生成
- 📁 多种文件格式支持
- 💾 向量化存储上下文
- 🌐 React + FastAPI全栈应用
- 🔄 实时优化反馈

### 支持的文件类型

- 文本文件：`.txt`, `.md`, `.markdown`
- 代码文件：`.py`, `.js`, `.jsx`, `.ts`, `.tsx`
- 配置文件：`.json`, `.yaml`, `.yml`
- 压缩文件：`.zip`, `.rar`, `.7z`

## 快速开始

### 环境要求

- Python 3.9+
- Node.js 16+
- OpenAI API密钥

### 安装步骤

1. 克隆项目
```bash
git clone <repository-url>
cd prompt-generator
```

2. 安装Python依赖
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

3. 安装前端依赖
```bash
cd frontend
npm install
```

4. 配置环境变量
```bash
cp .env.example .env
# 编辑.env文件，设置您的OpenAI API密钥和其他配置
```

### 启动服务

1. 启动后端服务
```bash
cd src/api
$env:PYTHONPATH = "."
python -m uvicorn src.api.main:app --reload
```

2. 启动前端服务
```bash
cd frontend
npm start
```

访问 http://localhost:3000 即可使用系统。

## 项目结构

```
prompt-generator/
├── src/
│   ├── api/              # FastAPI后端
│   │   ├── agents/
│   │   │   ├── template_generation_agent.py
│   │   │   └── prompt_optimization_agent.py
│   │   ├── utils/
│   │   │   └── vector_store.py
│   │   ├── template_generator.py
│   │   └── prompt_optimizer.py
│   ├── frontend/            # React前端
│   ├── tests/              # 测试文件
│   ├── data/               # 数据文件
│   └── docs/              # 文档
├── config/
│   └── config.py
└── README.md
```

## API文档

启动后端服务后，访问 http://localhost:8000/docs 查看API文档。

### 主要接口

- `POST /api/generate-template`: 生成提示词模板
- `POST /api/optimize-prompt`: 优化提示词
- `POST /api/upload-context`: 上传上下文文件

## 开发指南

### 代码规范

项目使用以下工具确保代码质量：
- Black: 代码格式化
- isort: 导入语句排序
- mypy: 类型检查
- flake8: 代码风格检查

运行代码检查：
```bash
# 格式化代码
black src tests
isort src tests

# 类型检查
mypy src

# 代码风格检查
flake8 src tests
```

### 运行测试

```bash
pytest
```

## 配置说明

主要配置项（在.env文件中设置）：

- `OPENAI_API_KEY`: OpenAI API密钥
- `OPENAI_MODEL`: 使用的模型（默认：gpt-3.5-turbo）
- `EMBEDDING_MODEL`: 嵌入模型
- `MAX_FILE_SIZE`: 最大文件大小限制
- `CHUNK_SIZE`: 文本分块大小
- `CHUNK_OVERLAP`: 分块重叠大小

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

[MIT License](LICENSE)

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。 