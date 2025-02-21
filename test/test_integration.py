import pytest
import asyncio
from pathlib import Path
import json
from src.file_processor import FileProcessor
from src.utils.vector_store import VectorStore
from src.template_generator import TemplateGenerator

@pytest.fixture
def vector_store():
    """创建测试用的向量存储"""
    return VectorStore(use_mock=True)

@pytest.fixture
def file_processor(vector_store):
    """创建测试用的文件处理器"""
    upload_dir = Path("test_uploads")
    return FileProcessor(upload_dir, vector_store)

@pytest.fixture
def template_generator(vector_store):
    """创建测试用的模板生成器"""
    generator = TemplateGenerator()
    generator.vector_store = vector_store  # 使用mock的向量存储
    generator.llm = None  # 使用默认模板
    return generator

@pytest.mark.asyncio
async def test_file_upload_and_processing(file_processor):
    """测试文件上传和处理流程"""
    # 创建测试文件
    test_file = Path("test_file.txt")
    test_content = """
    前端技术: React, TypeScript, Ant Design
    后端技术: FastAPI, Python
    数据库技术: PostgreSQL
    API设计: RESTful API
    导航设计: 响应式导航栏
    响应式设计: 移动端优先
    用户交互流程: 登录-首页-功能页
    状态管理方案: Redux
    数据流设计: 单向数据流
    组件设计: 原子设计模式
    """
    test_file.write_text(test_content)

    try:
        # 处理文件
        result = await file_processor.process_file(test_file)
        
        # 验证结果
        assert result["status"] == "success"
        assert result["file_type"] == "text"
        assert result["chunks"] > 0
        
        # 验证向量数据库插入
        vector_store = file_processor.vector_store
        assert vector_store.verify_insertion(test_content)
        
    finally:
        # 清理测试文件
        if test_file.exists():
            test_file.unlink()

@pytest.mark.asyncio
async def test_template_generation(template_generator):
    """测试模板生成流程"""
    # 生成模板
    template = template_generator.generate()  # 使用默认模板
    
    # 验证模板包含所有必要字段
    is_valid, missing_fields = template_generator.validate_template(template)
    assert is_valid, f"缺少字段: {missing_fields}"
    assert len(missing_fields) == 0
    
    # 验证模板已保存到向量数据库
    vector_store = template_generator.vector_store
    assert vector_store.verify_insertion(template, "templates")

@pytest.mark.asyncio
async def test_full_workflow(file_processor, template_generator):
    """测试完整工作流程"""
    # 1. 上传并处理文件
    test_file = Path("test_project.txt")
    test_content = """
    项目名称: 测试项目
    前端技术: React, TypeScript
    后端技术: FastAPI
    数据库技术: MongoDB
    API设计: GraphQL
    导航设计: 侧边栏导航
    响应式设计: Bootstrap Grid
    用户交互流程: 注册-登录-仪表盘
    状态管理方案: MobX
    数据流设计: 事件驱动
    组件设计: 组件库
    """
    test_file.write_text(test_content)

    try:
        # 处理文件
        result = await file_processor.process_file(test_file)
        assert result["status"] == "success"
        
        # 2. 生成模板
        template = template_generator.generate()  # 使用默认模板
        is_valid, missing_fields = template_generator.validate_template(template)
        assert is_valid, f"缺少字段: {missing_fields}"
        
        # 3. 验证整个流程
        vector_store = file_processor.vector_store
        assert vector_store.verify_insertion(test_content)
        assert vector_store.verify_insertion(template, "templates")
        
    finally:
        # 清理测试文件
        if test_file.exists():
            test_file.unlink()

def test_error_handling(file_processor, template_generator):
    """测试错误处理"""
    # 1. 测试不支持的文件类型
    assert not file_processor.is_supported_file("test.xyz")
    
    # 2. 测试空文件处理
    empty_file = Path("empty.txt")
    empty_file.write_text("")
    
    try:
        result = asyncio.run(file_processor.process_file(empty_file))
        assert result["status"] == "warning"
        assert "文件内容为空" in result["message"]
    finally:
        if empty_file.exists():
            empty_file.unlink()
    
    # 3. 测试缺失必要字段的模板
    invalid_template = "这是一个无效的模板"
    is_valid, missing_fields = template_generator.validate_template(invalid_template)
    assert not is_valid
    assert len(missing_fields) == len(template_generator.REQUIRED_FIELDS)

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 