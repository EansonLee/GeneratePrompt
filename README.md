# Prompt优化系统

这是一个基于LangChain和OpenAI的Prompt优化系统，提供两个主要功能：
1. 生成模板prompt
2. 优化用户输入的prompt

## 功能特点

- 支持上传上下文文件（代码、文档等）
- 使用向量数据库存储和检索上下文
- 智能生成和优化prompt
- 支持用户修改和确认
- 自动保存历史记录

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/prompt-optimization-system.git
cd prompt-optimization-system
```

2. 创建并激活虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 环境配置：
```bash
cp .env.example .env
```
编辑 `.env` 文件，设置必要的环境变量：
- `OPENAI_API_KEY`: 你的OpenAI API密钥
- `OPENAI_MODEL`: 使用的模型（默认为gpt-4-turbo-preview）
- 其他配置项请参考 `.env.example` 文件

## 安全注意事项

1. 敏感信息保护：
   - 永远不要提交 `.env` 文件到版本控制系统
   - 确保 `.gitignore` 正确配置以排除敏感文件
   - 不要在代码中硬编码任何密钥或敏感信息

2. 数据安全：
   - 向量数据库文件夹 (`data/vector_store`) 不会被提交
   - 所有临时文件和日志都被 `.gitignore` 排除
   - 建议定期备份重要数据

## 使用方法

1. 运行程序：
```bash
python src/main.py
```

2. 选择功能：
   - 输入 `1` 生成模板prompt
   - 输入 `2` 优化用户输入的prompt

3. 根据提示操作：
   - 选择是否上传上下文文件
   - 输入文件路径（如果需要）
   - 输入或查看prompt
   - 确认或修改生成的结果

## 项目结构

```
prompt-optimization-system/
├── config/
│   └── config.py         # 配置文件
├── src/
│   ├── main.py          # 主程序
│   ├── prompt_optimizer.py    # Prompt优化器
│   └── template_generator.py  # 模板生成器
├── agents/
│   └── template_generation_agent.py  # 模板生成Agent
├── utils/
│   └── vector_store.py  # 向量数据库工具
├── requirements.txt      # 依赖列表
├── .env.example         # 环境变量示例
└── README.md           # 说明文档
```

## 开发指南

1. 代码风格：
   - 遵循PEP 8规范
   - 使用类型注解
   - 添加适当的注释和文档字符串

2. 测试：
   - 运行测试：`python -m pytest`
   - 添加新功能时编写相应的测试用例
   - 确保测试覆盖率达到标准

3. 环境变量：
   - 开发新功能时在 `.env.example` 中添加新的配置项
   - 更新文档以反映新的配置要求

## 贡献

1. Fork 项目
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -am 'Add some feature'`
4. 推送到分支：`git push origin feature/your-feature`
5. 提交 Pull Request

## 许可证

MIT License 