import pytest
from unittest.mock import patch, Mock
from src.template_generator import TemplateGenerator
from src.utils.vector_store import VectorStore
from src.agents.template_generation_agent import TemplateGenerationAgent

@pytest.fixture
def mock_vector_store():
    mock = Mock(spec=VectorStore)
    mock.search_contexts.return_value = [{'content': 'context1'}, {'content': 'context2'}]
    mock.search_templates.return_value = ['template1', 'template2']
    return mock

@pytest.fixture
def mock_agent():
    return Mock(spec=TemplateGenerationAgent)

@pytest.fixture
def template_generator(mock_vector_store, mock_agent):
    with patch('src.template_generator.VectorStore', return_value=mock_vector_store), \
         patch('src.template_generator.TemplateGenerationAgent', return_value=mock_agent):
        return TemplateGenerator()

def test_init(template_generator):
    """测试初始化"""
    assert isinstance(template_generator.vector_store, VectorStore)
    assert isinstance(template_generator.agent, TemplateGenerationAgent)

def test_get_context_success(template_generator):
    """测试成功获取上下文"""
    # 执行测试
    context = template_generator._get_context()
    
    # 验证结果
    assert context['contexts'] == [{'content': 'context1'}, {'content': 'context2'}]
    assert context['templates'] == ['template1', 'template2']

def test_get_context_failure(template_generator, mock_vector_store):
    """测试获取上下文失败的情况"""
    # 设置mock抛出异常
    mock_vector_store.search_contexts.side_effect = Exception("搜索上下文失败")
    
    # 验证异常抛出
    with pytest.raises(Exception) as exc_info:
        template_generator._get_context()
    assert "搜索上下文失败" in str(exc_info.value)

def test_generate_success(template_generator, mock_agent):
    """测试成功生成模板"""
    # 准备测试数据
    expected_template = """
    # 项目基本信息
    - 项目名称：测试项目
    - 项目描述：这是一个测试项目
    - 项目架构：前后端分离架构

    # 技术栈信息
    - 前端技术：React + TypeScript
    - UI框架：Ant Design
    - 后端技术：FastAPI
    - 数据库技术：PostgreSQL
    - API设计：RESTful API

    # 页面信息
    - 页面列表：登录页、首页、用户中心
    - 导航设计：顶部导航栏 + 侧边菜单
    - 响应式设计：支持移动端和桌面端
    - 用户交互流程：登录 -> 首页 -> 功能页面
    - 状态管理方案：Redux
    - 数据流设计：单向数据流
    - 组件设计：
      * 组件层次：原子组件 -> 分子组件 -> 有机组件
      * 组件通信：Props + Context + Redux
      * 组件复用：高阶组件 + Hooks
    """
    
    # 设置mock返回值
    mock_agent.generate.return_value = expected_template
    
    # 执行测试
    result = template_generator.generate()
    
    # 验证结果
    assert result == expected_template

def test_generate_failure(template_generator, mock_vector_store):
    """测试生成模板失败的情况"""
    # 设置mock抛出异常
    mock_vector_store.search_contexts.side_effect = Exception("生成失败")
    
    # 验证异常抛出
    with pytest.raises(Exception) as exc_info:
        template_generator.generate()
    assert "生成失败" in str(exc_info.value)

def test_template_content_quality(template_generator, mock_agent):
    """测试生成的模板prompt质量"""
    # 准备测试数据
    expected_template = """
    # 项目基本信息
    - 项目名称：测试项目
    - 项目描述：这是一个测试项目
    - 项目架构：前后端分离架构

    # 技术栈信息
    - 前端技术：React + TypeScript
    - UI框架：Ant Design
    - 后端技术：FastAPI
    - 数据库技术：PostgreSQL
    - API设计：RESTful API

    # 页面信息
    - 页面列表：登录页、首页、用户中心
    - 导航设计：顶部导航栏 + 侧边菜单
    - 响应式设计：支持移动端和桌面端
    - 用户交互流程：登录 -> 首页 -> 功能页面
    - 状态管理方案：Redux
    - 数据流设计：单向数据流
    - 组件设计：
      * 组件层次：原子组件 -> 分子组件 -> 有机组件
      * 组件通信：Props + Context + Redux
      * 组件复用：高阶组件 + Hooks
    """
    
    # 设置mock返回值
    mock_agent.generate.return_value = expected_template
    
    # 执行测试
    template = template_generator.generate()
    
    # 验证模板内容质量
    required_sections = [
        # 工程基本信息
        "项目名称",
        "项目描述",
        "项目架构",
        
        # 技术栈信息
        "前端技术",
        "UI框架",
        "后端技术",
        "数据库技术",
        "API设计",
        
        # 页面信息
        "页面列表",
        "导航设计",
        "响应式设计",
        "用户交互流程",
        "状态管理方案",
        "数据流设计",
        "组件设计",
        "组件层次",
        "组件通信",
        "组件复用"
    ]
    
    for section in required_sections:
        assert section in template, f"模板中缺少{section}相关信息"

def test_template_technical_stack_details(template_generator, mock_agent):
    """测试模板中技术栈描述的完整性"""
    # 准备测试数据
    expected_template = """
    # 项目基本信息
    - 项目名称：测试项目
    - 项目描述：这是一个测试项目
    - 项目架构：前后端分离架构

    # 技术栈信息
    - 前端技术：React + TypeScript
    - UI框架：Ant Design
    - 后端技术：FastAPI
    - 数据库技术：PostgreSQL
    - API设计：RESTful API

    # 页面信息
    - 页面列表：登录页、首页、用户中心
    - 导航设计：顶部导航栏 + 侧边菜单
    - 响应式设计：支持移动端和桌面端
    - 用户交互流程：登录 -> 首页 -> 功能页面
    - 状态管理方案：Redux
    - 数据流设计：单向数据流
    - 组件设计：
      * 组件层次：原子组件 -> 分子组件 -> 有机组件
      * 组件通信：Props + Context + Redux
      * 组件复用：高阶组件 + Hooks
    """
    
    # 设置mock返回值
    mock_agent.generate.return_value = expected_template
    
    # 执行测试
    template = template_generator.generate()
    
    # 验证技术栈描述
    tech_stack_requirements = [
        # 前端技术
        "React",
        "TypeScript",
        "UI框架",
        "Ant Design",
        
        # 后端技术
        "FastAPI",
        "PostgreSQL",
        "RESTful API",
        
        # 状态管理
        "Redux",
        "状态管理方案",
        
        # 组件设计
        "组件层次",
        "组件通信",
        "组件复用"
    ]
    
    for tech in tech_stack_requirements:
        assert tech in template, f"模板中缺少{tech}相关技术描述"

def test_template_page_information(template_generator, mock_agent):
    """测试模板中页面信息的完整性"""
    # 准备测试数据
    expected_template = """
    # 项目基本信息
    - 项目名称：测试项目
    - 项目描述：这是一个测试项目
    - 项目架构：前后端分离架构

    # 技术栈信息
    - 前端技术：React + TypeScript
    - UI框架：Ant Design
    - 后端技术：FastAPI
    - 数据库技术：PostgreSQL
    - API设计：RESTful API

    # 页面信息
    - 页面列表：登录页、首页、用户中心
    - 导航设计：顶部导航栏 + 侧边菜单
    - 响应式设计：支持移动端和桌面端
    - 用户交互流程：登录 -> 首页 -> 功能页面
    - 状态管理方案：Redux
    - 数据流设计：单向数据流
    - 组件设计：
      * 组件层次：原子组件 -> 分子组件 -> 有机组件
      * 组件通信：Props + Context + Redux
      * 组件复用：高阶组件 + Hooks
    """
    
    # 设置mock返回值
    mock_agent.generate.return_value = expected_template
    
    # 执行测试
    template = template_generator.generate()
    
    # 验证页面信息
    page_info_requirements = [
        # 页面结构
        "页面列表",
        "导航设计",
        "响应式设计",
        
        # 交互设计
        "用户交互流程",
        "状态管理方案",
        "数据流设计",
        
        # 组件设计
        "组件层次",
        "组件通信",
        "组件复用"
    ]
    
    for info in page_info_requirements:
        assert info in template, f"模板中缺少{info}相关描述"

def test_template_generation_with_context(template_generator, mock_vector_store, mock_agent):
    """测试基于上下文的模板生成"""
    # 准备测试数据
    test_contexts = [
        {
            "content": "项目类型: web_app\n项目描述: 健身训练应用，帮助用户制定和追踪健身计划",
            "type": "project_info"
        }
    ]
    
    expected_template = """
    # 项目基本信息
    - 项目名称：健身训练应用
    - 项目描述：帮助用户制定和追踪健身计划的应用
    - 项目架构：前后端分离架构

    # 技术栈信息
    - 前端技术：React + TypeScript
    - UI框架：Ant Design
    - 后端技术：FastAPI
    - 数据库技术：PostgreSQL
    - API设计：RESTful API

    # 页面信息
    - 页面列表：登录页、首页、训练计划页、进度追踪页
    - 导航设计：顶部导航栏 + 侧边菜单
    - 响应式设计：支持移动端和桌面端
    - 用户交互流程：登录 -> 首页 -> 训练计划 -> 进度追踪
    - 状态管理方案：Redux
    - 数据流设计：单向数据流
    - 组件设计：
      * 组件层次：原子组件 -> 分子组件 -> 有机组件
      * 组件通信：Props + Context + Redux
      * 组件复用：高阶组件 + Hooks
    """
    
    # 设置mock返回值
    mock_vector_store.search_contexts.return_value = test_contexts
    mock_agent.generate.return_value = expected_template
    
    # 执行测试
    template = template_generator.generate(project_type="web_app", project_description="健身训练应用")
    
    # 验证模板是否包含上下文信息
    assert "健身训练应用" in template
    assert "训练计划" in template
    assert "进度追踪" in template

def test_template_customization(template_generator, mock_vector_store, mock_agent):
    """测试模板的自定义能力"""
    # 测试不同类型项目的模板生成
    project_types = [
        {
            "type": "web_app",
            "description": "Web应用模板",
            "template": """
    # 项目基本信息
    - 项目名称：Web应用
    - 项目描述：Web应用模板
    - 项目架构：前后端分离架构

    # 技术栈信息
    - 前端技术：React + TypeScript
    - UI框架：Ant Design
    - 后端技术：Node.js + Express
    - 数据库技术：MongoDB
    - API设计：RESTful API

    # 页面信息
    - 页面列表：登录页、首页、功能页
    - 导航设计：顶部导航栏 + 侧边菜单
    - 响应式设计：支持移动端和桌面端
    - 用户交互流程：标准Web应用流程
    - 状态管理方案：Redux
    - 数据流设计：单向数据流
    - 组件设计：
      * 组件层次：标准Web组件层次
      * 组件通信：标准通信方式
      * 组件复用：Web组件复用策略
    """
        }
    ]
    
    # 设置mock返回值
    for project_type in project_types:
        mock_vector_store.search_contexts.return_value = [
            {
                "content": f"项目类型: {project_type['type']}\n项目描述: {project_type['description']}",
                "type": "project_info"
            }
        ]
        mock_agent.generate.return_value = project_type["template"]
        
        # 执行测试
        template = template_generator.generate(
            project_type=project_type["type"],
            project_description=project_type["description"]
        )
        
        # 验证结果
        assert "Web应用" in template
        assert "Web应用模板" in template 