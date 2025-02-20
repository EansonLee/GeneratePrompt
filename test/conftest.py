import os
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置测试环境变量
os.environ.setdefault('PYTHONPATH', str(project_root))
os.environ.setdefault('TESTING', 'True')
os.environ.setdefault('OPENAI_API_KEY', 'test-key')
os.environ.setdefault('MOCK_OPENAI', 'True')
os.environ.setdefault('MOCK_EMBEDDINGS', 'True')

@pytest.fixture(autouse=True)
def mock_env_vars():
    """自动mock所有外部依赖"""
    with patch.dict(os.environ, {
        'TESTING': 'True',
        'MOCK_OPENAI': 'True',
        'MOCK_EMBEDDINGS': 'True',
        'OPENAI_API_KEY': 'test-key'
    }):
        yield

@pytest.fixture
def mock_openai():
    """Mock OpenAI API调用"""
    with patch('langchain_community.chat_models.ChatOpenAI') as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        mock_instance.invoke.return_value = {"text": "优化后的提示"}
        yield mock_instance

@pytest.fixture
def mock_embeddings():
    """Mock Embeddings API调用"""
    with patch('langchain_community.embeddings.OpenAIEmbeddings') as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        mock_instance.embed_documents.return_value = [[0.1] * 1536]
        mock_instance.embed_query.return_value = [0.1] * 1536
        yield mock_instance

@pytest.fixture
def mock_faiss():
    """Mock FAISS向量存储"""
    with patch('langchain_community.vectorstores.FAISS') as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        mock.from_texts.return_value = mock_instance
        mock_instance.similarity_search.return_value = [
            Mock(page_content="测试内容", metadata={"type": "test"})
        ]
        yield mock_instance 