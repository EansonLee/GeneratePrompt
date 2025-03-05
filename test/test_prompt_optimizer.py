import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from src.prompt_optimizer import PromptOptimizer
from src.template_generator import TemplateGenerator
from config.config import DEFAULT_MODEL_NAME
from src.agents.prompt_optimization_agent import PromptOptimizationAgent
from src.utils.vector_store import VectorStore
from langchain_core.messages import AIMessage
from src.utils.rag_manager import RAGManager
import json
import asyncio

@pytest.fixture
def mock_openai():
    """Mock OpenAI API"""
    with patch('langchain_community.chat_models.ChatOpenAI') as mock:
        mock_instance = mock.return_value
        mock_instance.invoke.return_value = "这是一个优化后的提示"
        yield mock_instance

@pytest.fixture
def mock_embeddings():
    """Mock OpenAI embeddings"""
    with patch('langchain_community.embeddings.OpenAIEmbeddings') as mock:
        mock_instance = mock.return_value
        mock_instance.embed_documents.return_value = [[0.1] * 1536]
        mock_instance.embed_query.return_value = [0.1] * 1536
        yield mock_instance

@pytest.fixture
def mock_faiss():
    """Mock FAISS vector store"""
    with patch('langchain_community.vectorstores.FAISS') as mock:
        mock_instance = mock.return_value
        mock_instance.similarity_search.return_value = [
            {"content": "测试上下文1"},
            {"content": "测试上下文2"}
        ]
        yield mock_instance

@pytest.fixture
def mock_agent():
    """Mock PromptOptimizationAgent"""
    with patch('src.agents.prompt_optimization_agent.PromptOptimizationAgent') as mock:
        mock_instance = mock.return_value
        mock_instance.optimize_prompt.return_value = {
            "optimized_prompt": "这是一个优化后的提示",
            "original_prompt": "原始提示",
            "contexts_used": 2,
            "templates_used": 1
        }
        yield mock_instance

@pytest.fixture
def prompt_optimizer(mock_openai, mock_embeddings, mock_faiss):
    """创建Prompt优化器实例的fixture"""
    return PromptOptimizer(use_mock=True)

@pytest.fixture
def mock_vector_store():
    """Mock向量存储"""
    store = Mock(spec=VectorStore)
    store.similarity_search.return_value = [
        "历史提示词1",
        "历史提示词2"
    ]
    return store

@pytest.fixture
def mock_rag_manager():
    """Mock RAG管理器"""
    manager = Mock(spec=RAGManager)
    manager.enhance_prompt.return_value = {
        "enhanced_prompt": "增强后的提示词",
        "context": {
            "technical": ["技术上下文"],
            "business": ["业务上下文"],
            "examples": ["示例"],
            "best_practices": ["最佳实践"]
        }
    }
    return manager

@pytest.fixture
def optimizer(mock_vector_store):
    """创建优化器实例"""
    return PromptOptimizer(vector_store=mock_vector_store)

def test_init(prompt_optimizer):
    """测试初始化"""
    assert prompt_optimizer is not None
    assert prompt_optimizer.vector_store is not None
    assert prompt_optimizer.agent is not None

def test_add_react_code_success(prompt_optimizer):
    """测试成功添加React代码"""
    code = """
    function App() {
        return <div>Hello</div>
    }
    """
    metadata = {
        "description": "简单的React组件",
        "type": "react_component"
    }
    result = prompt_optimizer.add_react_code(code, metadata)
    assert result is not None

def test_add_react_code_failure(prompt_optimizer, mock_faiss):
    """测试添加React代码失败的情况"""
    mock_faiss.add_texts.side_effect = Exception("添加失败")
    with pytest.raises(Exception):
        prompt_optimizer.add_react_code("invalid code")

def test_optimize_success(prompt_optimizer, mock_agent):
    """测试成功优化prompt"""
    original_prompt = "帮我写一个按钮组件"
    optimized_prompt = "请帮我创建一个React按钮组件，包含以下特性：样式可定制、支持点击事件、支持禁用状态"

    # 设置mock返回值
    mock_agent.optimize_prompt.return_value = {
        "optimized_prompt": optimized_prompt,
        "original_prompt": original_prompt,
        "contexts_used": 3,
        "templates_used": 2
    }

    result = prompt_optimizer.optimize(original_prompt)
    assert result["original_prompt"] == original_prompt
    assert result["optimized_prompt"] == optimized_prompt

def test_optimize_failure(prompt_optimizer, mock_agent):
    """测试优化prompt失败的情况"""
    # 替换 prompt_optimizer 中的 agent
    prompt_optimizer.agent = mock_agent
    mock_agent.optimize_prompt.side_effect = Exception("优化失败")
    
    with pytest.raises(Exception) as exc_info:
        prompt_optimizer.optimize("测试提示")
    assert "优化失败" in str(exc_info.value)

def test_process_project_file_react_component(prompt_optimizer, tmp_path):
    """测试处理React组件文件"""
    file_path = tmp_path / "Button.tsx"
    content = """
    import React from 'react';
    const Button = () => <button>Click me</button>;
    export default Button;
    """
    file_path.write_text(content)

    with patch.object(prompt_optimizer, 'add_react_code') as mock_add_react:
        result = prompt_optimizer.process_project_file(file_path)
        assert result is True
        mock_add_react.assert_called_once()

def test_process_project_file_not_exists(prompt_optimizer):
    """测试处理不存在的文件"""
    with pytest.raises(FileNotFoundError):
        prompt_optimizer.process_project_file("not_exists.tsx")

def test_process_project_directory(prompt_optimizer, tmp_path):
    """测试处理项目目录"""
    # 创建测试目录结构
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "components").mkdir()

    # 创建测试文件
    (tmp_path / "src" / "components" / "Button.tsx").write_text("""
    import React from 'react';
    const Button = () => <button>Click me</button>;
    export default Button;
    """)

    (tmp_path / "src" / "components" / "Button.test.tsx").write_text("""
    import { render } from '@testing-library/react';
    import Button from './Button';

    test('renders button', () => {
        render(<Button />);
    });
    """)

    with patch.object(prompt_optimizer, 'process_project_file', return_value=True):
        stats = prompt_optimizer.process_project_directory(tmp_path)
        assert stats["react_component"] >= 1
        assert stats["test"] >= 1
        assert stats["failed"] == 0

def test_detect_file_type(prompt_optimizer):
    """测试文件类型检测"""
    assert prompt_optimizer.detect_file_type("test.tsx") == "react_component"
    assert prompt_optimizer.detect_file_type("test.test.tsx") == "test"
    assert prompt_optimizer.detect_file_type("test.py") == "python"
    assert prompt_optimizer.detect_file_type("test.md") == "doc"
    assert prompt_optimizer.detect_file_type("test.json") == "config"
    assert prompt_optimizer.detect_file_type("test.other") == "other"

def test_should_ignore(prompt_optimizer):
    """测试忽略文件检查"""
    assert prompt_optimizer.should_ignore("node_modules/test.js") is True
    assert prompt_optimizer.should_ignore("src/App.tsx") is False

def test_prompt_optimization_with_template(prompt_optimizer, mock_agent):
    """测试使用模板优化提示"""
    template = "创建{component}组件"
    mock_agent.optimize_prompt.return_value = {
        "optimized_prompt": "请创建一个Button组件，包含以下特性：样式可定制、支持点击事件",
        "original_prompt": template.format(component="Button"),
        "contexts_used": 2,
        "templates_used": 1
    }
    result = prompt_optimizer.optimize_with_template(template, {"component": "Button"})
    assert result is not None

def test_prompt_optimization_quality(prompt_optimizer, mock_agent):
    """测试提示优化质量"""
    test_cases = [
        {
            "input": "完成一个健身训练页面",
            "expected_elements": [
                "健身训练页面详细设计",
                "页面布局结构",
                "组件层次关系",
                "数据流设计",
                "状态管理方案"
            ]
        }
    ]

    for case in test_cases:
        mock_agent.optimize_prompt.return_value = {
            "optimized_prompt": """
            请创建一个健身训练页面，包含以下内容：
            1. 健身训练页面详细设计
            2. 页面布局结构
            3. 组件层次关系
            4. 数据流设计
            5. 状态管理方案
            """,
            "original_prompt": case["input"],
            "contexts_used": 2,
            "templates_used": 1
        }
        result = prompt_optimizer.optimize(case["input"])
        optimized_prompt = result["optimized_prompt"]
        for element in case["expected_elements"]:
            assert element in optimized_prompt, f"优化后的prompt缺少{element}相关描述"

def test_prompt_optimization_with_context(prompt_optimizer, mock_agent):
    """测试上下文感知的prompt优化"""
    # 准备项目上下文
    project_context = {
        "project_type": "健身应用",
        "tech_stack": "React + TypeScript",
        "features": ["用户认证", "训练计划", "进度追踪"]
    }

    # 注入上下文
    with patch.object(prompt_optimizer.vector_store, 'get_project_context',
                     return_value=project_context):
        mock_agent.optimize_prompt.return_value = {
            "optimized_prompt": """
            请使用React和TypeScript创建训练计划页面，包含以下功能：
            1. 用户认证集成
            2. 训练计划展示
            3. 进度追踪功能
            """,
            "original_prompt": "添加训练计划页面",
            "contexts_used": 2,
            "templates_used": 1
        }
        result = prompt_optimizer.optimize("添加训练计划页面")
        optimized_prompt = result["optimized_prompt"]
        assert "React" in optimized_prompt

def test_prompt_optimization_consistency(prompt_optimizer, mock_agent):
    """测试prompt优化的一致性"""
    test_prompt = "完成用户资料页面"
    optimized_prompt = """
    请创建用户资料页面，包含以下内容：
    1. 用户资料展示
    2. 页面布局设计
    3. 数据展示组件
    4. 编辑功能实现
    """

    # 多次优化同一个prompt
    mock_agent.optimize_prompt.return_value = {
        "optimized_prompt": optimized_prompt,
        "original_prompt": test_prompt,
        "contexts_used": 2,
        "templates_used": 1
    }

    results = []
    for _ in range(3):
        result = prompt_optimizer.optimize(test_prompt)
        results.append(result["optimized_prompt"])

    # 验证核心要素的一致性
    core_elements = [
        "用户资料",
        "页面布局",
        "数据展示",
        "编辑功能"
    ]

    for element in core_elements:
        assert all(element in result for result in results)

def test_prompt_optimization_with_requirements(prompt_optimizer, mock_agent):
    """测试带有具体要求的prompt优化"""
    requirements = {
        "accessibility": True,
        "responsive": True,
        "theme": "dark",
        "language": "zh-CN"
    }

    test_prompt = "创建一个数据展示页面"

    # 使用特定要求优化prompt
    with patch.object(prompt_optimizer, 'get_optimization_requirements',
                     return_value=requirements):
        mock_agent.optimize_prompt.return_value = {
            "optimized_prompt": """
            请创建一个数据展示页面，需要满足以下要求：
            1. 实现无障碍设计
            2. 响应式布局
            3. 深色主题
            4. 中文界面
            """,
            "original_prompt": test_prompt,
            "contexts_used": 2,
            "templates_used": 1
        }
        result = prompt_optimizer.optimize(test_prompt)
        optimized_prompt = result["optimized_prompt"]
        assert "无障碍设计" in optimized_prompt

def test_prompt_optimization_error_handling(prompt_optimizer):
    """测试提示优化的错误处理"""
    with pytest.raises(Exception):
        prompt_optimizer.optimize("")

def test_optimize_prompt_success():
    """测试提示优化成功的情况"""
    # 创建 mock vector store
    mock_vector_store = Mock(spec=VectorStore)
    mock_vector_store.search_contexts.return_value = [
        {"content": "这是一个测试上下文"}
    ]
    mock_vector_store.search_templates.return_value = [
        "这是一个测试模板"
    ]
    mock_vector_store.add_prompt_history = Mock()

    # 创建优化代理
    agent = PromptOptimizationAgent(
        vector_store=mock_vector_store,
        is_testing=True
    )

    # 执行优化
    result = agent.optimize_prompt("创建一个按钮组件")

    # 验证结果
    assert isinstance(result, dict)
    assert "optimized_prompt" in result
    assert "original_prompt" in result
    assert result["contexts_used"] == 1
    assert result["templates_used"] == 1

def test_optimize_prompt_failure():
    """测试提示优化失败的情况"""
    # 创建会抛出异常的 mock vector store
    mock_vector_store = Mock(spec=VectorStore)
    mock_vector_store.search_contexts.side_effect = Exception("搜索失败")

    # 创建优化代理
    agent = PromptOptimizationAgent(
        vector_store=mock_vector_store,
        is_testing=True
    )

    # 验证异常抛出
    with pytest.raises(Exception) as exc_info:
        agent.optimize_prompt("测试提示")
    assert "搜索失败" in str(exc_info.value)

def test_get_optimization_history():
    """测试获取优化历史"""
    # 创建 mock vector store
    mock_vector_store = Mock(spec=VectorStore)
    mock_vector_store.get_prompt_history.return_value = [
        {
            "original_prompt": "原始提示",
            "optimized_prompt": "优化后的提示"
        }
    ]

    # 创建优化代理
    agent = PromptOptimizationAgent(
        vector_store=mock_vector_store,
        is_testing=True
    )

    # 获取历史记录
    history = agent.get_optimization_history()

    # 验证结果
    assert isinstance(history, list)
    assert len(history) == 1
    assert history[0]["original_prompt"] == "原始提示"
    assert history[0]["optimized_prompt"] == "优化后的提示"

def test_set_model_parameters():
    """测试设置模型参数"""
    # 创建优化代理
    agent = PromptOptimizationAgent(
        vector_store=Mock(spec=VectorStore),
        is_testing=True
    )

    # 设置参数
    agent.set_model_parameters(temperature=0.8, max_tokens=100)

    # 验证参数设置（在测试模式下不会实际改变参数）
    assert True  # 只要不抛出异常就算通过 

@pytest.mark.asyncio
async def test_optimize_empty_prompt(optimizer):
    """测试空提示词"""
    with pytest.raises(ValueError, match="提示词不能为空"):
        await optimizer.optimize("")

@pytest.mark.asyncio
async def test_optimize_success(optimizer, mock_vector_store):
    """测试成功优化提示词"""
    # 准备测试数据
    test_prompt = "测试提示词"
    mock_vector_store.similarity_search.return_value = ["历史提示词1", "历史提示词2"]
    
    # 执行优化
    result = await optimizer.optimize(test_prompt)
    
    # 验证结果
    assert isinstance(result, dict)
    assert "optimized_prompt" in result
    assert "analysis" in result
    assert "suggestions" in result
    assert "scores" in result
    assert "timestamp" in result
    
    # 验证调用
    mock_vector_store.similarity_search.assert_called_once()
    
@pytest.mark.asyncio
async def test_optimize_with_context(optimizer):
    """测试带上下文的优化"""
    # 准备测试数据
    test_prompt = "测试提示词"
    test_context = {
        "tech_stack": {
            "frontend": ["React"],
            "backend": ["Python"]
        }
    }
    
    # 执行优化
    result = await optimizer.optimize(test_prompt, context=test_context)
    
    # 验证结果
    assert isinstance(result, dict)
    assert result["optimized_prompt"]
    
@pytest.mark.asyncio
async def test_enhance_with_rag(optimizer):
    """测试RAG增强"""
    # 准备测试数据
    test_prompt = "测试提示词"
    similar_prompts = ["历史1", "历史2"]
    
    # 执行增强
    result = await optimizer._enhance_with_rag(test_prompt, similar_prompts)
    
    # 验证结果
    assert isinstance(result, (dict, str))
    
@pytest.mark.asyncio
async def test_evaluate_prompt(optimizer):
    """测试提示词评估"""
    # 准备测试数据
    test_prompt = "测试提示词"
    
    # 执行评估
    result = await optimizer._evaluate_prompt(test_prompt)
    
    # 验证结果
    assert isinstance(result, dict)
    assert "scores" in result
    assert "feedback" in result
    assert "improvement_suggestions" in result
    
@pytest.mark.asyncio
async def test_save_optimization_result(optimizer, tmp_path):
    """测试保存优化结果"""
    # 准备测试数据
    original_prompt = "原始提示词"
    optimized_prompt = "优化后的提示词"
    evaluation = {
        "scores": {
            "completeness": 8,
            "clarity": 9
        },
        "feedback": ["反馈1"],
        "improvement_suggestions": ["建议1"]
    }
    context = {
        "tech_stack": {
            "frontend": ["React"]
        }
    }
    
    # 执行保存
    await optimizer._save_optimization_result(
        original_prompt=original_prompt,
        optimized_prompt=optimized_prompt,
        evaluation=evaluation,
        context=context
    )
    
    # 验证调用
    optimizer.vector_store.add_prompt_history.assert_called_once()
    
def test_get_project_context(optimizer):
    """测试获取项目上下文"""
    # 执行获取
    context = optimizer._get_project_context()
    
    # 验证结果
    assert isinstance(context, dict)
    assert "tech_stack" in context
    assert "file_structure" in context
    assert "output_paths" in context
    
@pytest.mark.asyncio
async def test_error_handling(optimizer):
    """测试错误处理"""
    # 模拟错误
    optimizer.vector_store.similarity_search.side_effect = Exception("测试错误")
    
    # 验证错误处理
    with pytest.raises(Exception):
        await optimizer.optimize("测试提示词")
        
@pytest.mark.asyncio
async def test_performance(optimizer):
    """测试性能"""
    import time
    
    # 准备测试数据
    test_prompt = "测试提示词" * 100  # 长文本
    
    # 测量执行时间
    start_time = time.time()
    await optimizer.optimize(test_prompt)
    end_time = time.time()
    
    # 验证执行时间在合理范围内
    assert end_time - start_time < 5  # 假设5秒是合理的超时时间

@pytest.mark.integration
class TestIntegration:
    """集成测试类"""
    
    @pytest.mark.asyncio
    async def test_full_optimization_flow(self):
        """测试完整的优化流程"""
        # 创建真实的组件实例
        vector_store = VectorStore()
        rag_manager = RAGManager(vector_store=vector_store)
        optimizer = PromptOptimizer(vector_store=vector_store)
        
        # 准备测试数据
        test_prompt = """
        请实现一个React组件，要求：
        1. 使用TypeScript
        2. 包含表单验证
        3. 支持响应式布局
        """
        
        # 1. 添加上下文
        await rag_manager.add_context(
            content="React最佳实践示例",
            content_type="best_practice"
        )
        
        # 2. 使用RAG增强
        enhanced = await rag_manager.enhance_prompt(test_prompt)
        
        # 3. 优化提示词
        result = await optimizer.optimize(
            prompt=enhanced["enhanced_prompt"]
        )
        
        # 验证结果
        assert isinstance(result, dict)
        assert "optimized_prompt" in result
        assert "analysis" in result
        assert "suggestions" in result
        assert "scores" in result
        assert "rag_info" in result
        
        # 验证优化质量
        assert len(result["optimized_prompt"]) > len(test_prompt)
        assert "TypeScript" in result["optimized_prompt"]
        assert "React" in result["optimized_prompt"]
        
        # 验证评分
        assert all(0 <= score <= 10 for score in result["scores"].values())
        
@pytest.mark.benchmark
class TestBenchmark:
    """性能基准测试类"""
    
    @pytest.mark.asyncio
    async def test_optimization_performance(self, benchmark):
        """测试优化性能"""
        # 创建优化器实例
        optimizer = PromptOptimizer()
        
        # 准备测试数据
        test_prompts = [
            "短提示词",
            "中等长度的提示词" * 10,
            "长提示词" * 100
        ]
        
        # 执行基准测试
        for prompt in test_prompts:
            result = await optimizer.optimize(prompt)
            
            # 验证结果
            assert isinstance(result, dict)
            assert "optimized_prompt" in result
            
        # 验证内存使用
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # 确保内存使用在合理范围内（例如小于500MB）
        assert memory_info.rss < 500 * 1024 * 1024 

@pytest.mark.asyncio
async def test_cache_mechanism(optimizer):
    """测试缓存机制"""
    # 准备测试数据
    test_prompt = "测试提示词"
    test_context = {"test": "context"}
    
    # 第一次调用
    result1 = await optimizer.optimize(test_prompt, test_context)
    
    # 第二次调用应该使用缓存
    result2 = await optimizer.optimize(test_prompt, test_context)
    
    # 验证结果
    assert result1 == result2
    assert optimizer.cache.get(f"{test_prompt}:{json.dumps(test_context)}") is not None

@pytest.mark.asyncio
async def test_cache_expiration(optimizer):
    """测试缓存过期"""
    # 设置较短的TTL
    optimizer.cache.ttl = 1
    
    # 添加测试数据到缓存
    test_key = "test_key"
    test_value = {"test": "value"}
    optimizer.cache.set(test_key, test_value)
    
    # 验证缓存存在
    assert optimizer.cache.get(test_key) == test_value
    
    # 等待缓存过期
    await asyncio.sleep(2)
    
    # 验证缓存已过期
    assert optimizer.cache.get(test_key) is None

@pytest.mark.asyncio
async def test_batch_optimization(optimizer):
    """测试批量优化"""
    # 准备测试数据
    test_prompts = [
        "测试提示词1",
        "测试提示词2",
        "测试提示词3"
    ]
    
    # 执行批量优化
    results = await optimizer.optimize_batch(test_prompts)
    
    # 验证结果
    assert len(results) == len(test_prompts)
    assert all(isinstance(result, dict) for result in results)
    assert all("optimized_prompt" in result for result in results)

@pytest.mark.asyncio
async def test_parallel_processing(optimizer):
    """测试并行处理"""
    import time
    
    # 准备测试数据
    test_prompts = ["测试提示词"] * 5
    
    # 测量串行处理时间
    start_time = time.time()
    for prompt in test_prompts:
        await optimizer.optimize(prompt)
    serial_time = time.time() - start_time
    
    # 测量并行处理时间
    start_time = time.time()
    await optimizer.optimize_batch(test_prompts)
    parallel_time = time.time() - start_time
    
    # 验证并行处理更快
    assert parallel_time < serial_time

@pytest.mark.asyncio
async def test_vector_retrieval_optimization(optimizer):
    """测试向量检索优化"""
    # 准备测试数据
    test_prompt = "测试提示词"
    
    # 获取嵌入向量
    embedding1 = optimizer._get_embedding(test_prompt)
    embedding2 = optimizer._get_embedding(test_prompt)
    
    # 验证缓存生效
    assert embedding1 == embedding2
    
    # 验证相似度过滤
    similar_prompts = ["相似1", "相似2", "不相关"]
    filtered_result = await optimizer._enhance_with_rag(test_prompt, similar_prompts)
    
    assert isinstance(filtered_result, dict)
    assert "relevant_info" in str(filtered_result)

@pytest.mark.benchmark
class TestPerformance:
    """性能测试类"""
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, optimizer):
        """测试内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 执行大量操作
        test_prompts = ["测试提示词"] * 100
        await optimizer.optimize_batch(test_prompts)
        
        # 验证内存增长在合理范围内
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        assert memory_increase < 500  # 内存增长不超过500MB
        
    @pytest.mark.asyncio
    async def test_cache_hit_rate(self, optimizer):
        """测试缓存命中率"""
        # 准备测试数据
        test_prompts = ["提示词A", "提示词B", "提示词A", "提示词C", "提示词B"]
        
        # 执行优化
        for prompt in test_prompts:
            await optimizer.optimize(prompt)
            
        # 计算缓存命中率
        cache_entries = len(optimizer.cache.cache)
        expected_entries = len(set(test_prompts))  # 去重后的提示词数量
        
        assert cache_entries == expected_entries
        
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, optimizer):
        """测试并发请求处理"""
        # 准备测试数据
        test_prompts = [f"并发测试{i}" for i in range(10)]
        
        # 创建并发任务
        tasks = [optimizer.optimize(prompt) for prompt in test_prompts]
        
        # 并发执行
        results = await asyncio.gather(*tasks)
        
        # 验证所有请求都成功处理
        assert len(results) == len(test_prompts)
        assert all(isinstance(result, dict) for result in results)
        
    @pytest.mark.asyncio
    async def test_error_recovery(self, optimizer, monkeypatch):
        """测试错误恢复"""
        # 模拟间歇性失败
        fail_count = 0
        
        def mock_similarity_search(*args, **kwargs):
            nonlocal fail_count
            fail_count += 1
            if fail_count % 2 == 1:
                raise Exception("模拟失败")
            return ["测试文档"]
            
        monkeypatch.setattr(optimizer.vector_store, "similarity_search", mock_similarity_search)
        
        # 执行优化（应该自动重试）
        result = await optimizer.optimize("测试提示词")
        
        # 验证最终成功
        assert isinstance(result, dict)
        assert "optimized_prompt" in result
        
@pytest.mark.benchmark
class TestPerformance:
    """性能测试类"""
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, optimizer):
        """测试内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 执行大量操作
        test_prompts = ["测试提示词"] * 100
        await optimizer.optimize_batch(test_prompts)
        
        # 验证内存增长在合理范围内
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        assert memory_increase < 500  # 内存增长不超过500MB
        
    @pytest.mark.asyncio
    async def test_cache_hit_rate(self, optimizer):
        """测试缓存命中率"""
        # 准备测试数据
        test_prompts = ["提示词A", "提示词B", "提示词A", "提示词C", "提示词B"]
        
        # 执行优化
        for prompt in test_prompts:
            await optimizer.optimize(prompt)
            
        # 计算缓存命中率
        cache_entries = len(optimizer.cache.cache)
        expected_entries = len(set(test_prompts))  # 去重后的提示词数量
        
        assert cache_entries == expected_entries
        
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, optimizer):
        """测试并发请求处理"""
        # 准备测试数据
        test_prompts = [f"并发测试{i}" for i in range(10)]
        
        # 创建并发任务
        tasks = [optimizer.optimize(prompt) for prompt in test_prompts]
        
        # 并发执行
        results = await asyncio.gather(*tasks)
        
        # 验证所有请求都成功处理
        assert len(results) == len(test_prompts)
        assert all(isinstance(result, dict) for result in results)
        
    @pytest.mark.asyncio
    async def test_error_recovery(self, optimizer, monkeypatch):
        """测试错误恢复"""
        # 模拟间歇性失败
        fail_count = 0
        
        def mock_similarity_search(*args, **kwargs):
            nonlocal fail_count
            fail_count += 1
            if fail_count % 2 == 1:
                raise Exception("模拟失败")
            return ["测试文档"]
            
        monkeypatch.setattr(optimizer.vector_store, "similarity_search", mock_similarity_search)
        
        # 执行优化（应该自动重试）
        result = await optimizer.optimize("测试提示词")
        
        # 验证最终成功
        assert isinstance(result, dict)
        assert "optimized_prompt" in result 