# Prompt优化器

这是一个基于LangChain和OpenAI的提示词优化工具，可以帮助用户生成和优化提示词模板。

## 功能特点

- 提示词模板生成
- 提示词优化
- React代码示例管理
- 最佳实践收集
- 向量存储支持

## 安装

1. 克隆项目
```bash
git clone https://github.com/yourusername/prompt-optimizer.git
cd prompt-optimizer
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境变量
创建`.env`文件并添加以下内容：
```
OPENAI_API_KEY=your_api_key_here
```

## 使用方法

1. 生成模板
```python
from src.template_generator import TemplateGenerator

generator = TemplateGenerator()
template = generator.generate()
print(template)
```

2. 优化提示词
```python
from src.prompt_optimizer import PromptOptimizer

optimizer = PromptOptimizer()
result = optimizer.optimize("创建一个用户登录页面")
print(result["optimized_prompt"])
```

3. 添加React代码示例
```python
optimizer.add_react_code(
    code="const App = () => <div>Hello</div>",
    metadata={"description": "简单的React组件"}
)
```

## 测试

运行测试：
```bash
python -m pytest test/ -v
```

## 项目结构

```
prompt-optimizer/
├── src/
│   ├── agents/
│   │   ├── template_generation_agent.py
│   │   └── prompt_optimization_agent.py
│   ├── utils/
│   │   └── vector_store.py
│   ├── template_generator.py
│   └── prompt_optimizer.py
├── test/
│   ├── test_template_generator.py
│   └── test_prompt_optimizer.py
├── config/
│   └── config.py
├── requirements.txt
└── README.md
```

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT 