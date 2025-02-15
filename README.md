# React Prompt 优化器

这是一个基于LangChain的工具，用于优化生成React前端代码的提示。该工具使用向量数据库存储和检索相关上下文，通过LLM优化用户输入的提示，以生成更高质量的React代码。

## 功能特点

- 使用LangChain框架进行提示优化
- 集成向量数据库用于存储和检索相关上下文
- 支持React最佳实践和模式的提示优化
- 自动保存优化历史，便于后续参考

## 安装要求

- Python 3.8+
- 虚拟环境（推荐）

## 安装步骤

1. 克隆项目并创建虚拟环境：
```bash
python -m venv langchain_env
.\langchain_env\Scripts\activate  # Windows
source langchain_env/bin/activate  # Linux/Mac
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
```bash
cp .env.example .env
# 编辑.env文件，填入必要的配置信息
```

## 使用方法

1. 启动程序：
```bash
python src/main.py
```

2. 输入React代码生成提示，程序会自动优化并输出优化后的提示。

3. 输入'quit'退出程序。

## 项目结构

```
prompt_generator/
├── config/
│   └── config.py
├── data/
│   └── vector_store/
├── src/
│   ├── main.py
│   └── prompt_optimizer.py
├── utils/
│   └── vector_store.py
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

## 注意事项

- 请确保在使用前正确配置OpenAI API密钥
- 建议在虚拟环境中运行项目
- 首次运行时会自动初始化向量数据库

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 许可证

MIT License 