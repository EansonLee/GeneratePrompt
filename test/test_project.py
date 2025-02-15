import unittest
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from functools import wraps

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

def ensure_env_loaded(func):
    """确保环境变量正确加载的装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 确保加载.env文件
        dotenv_path = os.path.join(project_root, '.env')
        load_dotenv(dotenv_path)
        
        # 验证环境变量
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY 未在环境变量中设置")
        if api_key == "your-api-key-here":
            raise ValueError("OPENAI_API_KEY 使用了默认值，请在.env文件中设置正确的值")
            
        return func(*args, **kwargs)
    return wrapper

from src.prompt_optimizer import PromptOptimizer
from src.agents.prompt_optimization_agent import PromptOptimizationAgent
from utils.vector_store import VectorStore
from config.config import LOG_LEVEL

class TestProjectWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """测试类初始化时的设置"""
        super().setUpClass()
        # 确保环境变量已加载
        dotenv_path = os.path.join(project_root, '.env')
        load_dotenv(dotenv_path)
        
        # 验证环境变量
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY 未在环境变量中设置")
        if api_key == "your-api-key-here":
            raise ValueError("OPENAI_API_KEY 使用了默认值，请在.env文件中设置正确的值")

    @ensure_env_loaded
    def setUp(self):
        """测试前的设置"""
        self.optimizer = PromptOptimizer()
        self.test_prompt = "生成一个用户管理页面"
        
    @ensure_env_loaded
    def test_prompt_optimizer_initialization(self):
        """测试PromptOptimizer初始化"""
        self.assertIsNotNone(self.optimizer)
        
    @ensure_env_loaded
    def test_add_and_optimize_prompt(self):
        """测试添加和优化提示的功能"""
        # 添加React代码示例
        self.optimizer.add_react_code(
            code="""
            function UserList({ users }) {
                return (
                    <div className="user-list">
                        {users.map(user => (
                            <UserCard key={user.id} user={user} />
                        ))}
                    </div>
                );
            }
            """,
            metadata={
                "description": "用户列表组件",
                "author": "test"
            }
        )
        
        # 添加最佳实践
        self.optimizer.add_best_practice(
            practice="React组件应该遵循单一职责原则",
            category="架构"
        )
        
        # 测试优化功能
        result = self.optimizer.optimize(self.test_prompt)
        self.assertIsNotNone(result)
        self.assertIn('original_prompt', result)
        self.assertIn('optimized_prompt', result)
        
    @ensure_env_loaded
    def test_project_processing(self):
        """测试项目处理功能"""
        test_project_path = os.path.join(project_root, "test_data")
        if not os.path.exists(test_project_path):
            os.makedirs(test_project_path)
            
        try:
            stats = self.optimizer.process_project_directory(test_project_path)
            self.assertIsInstance(stats, dict)
            self.assertIn('react_component', stats)
            self.assertIn('test', stats)
            self.assertIn('config', stats)
        finally:
            # 清理测试数据
            if os.path.exists(test_project_path):
                import shutil
                shutil.rmtree(test_project_path)
                
    @ensure_env_loaded
    def test_vector_store(self):
        """测试向量存储功能"""
        vector_store = VectorStore()
        test_text = "这是一个测试文本"
        vector_store.add_texts([test_text], [{"type": "test"}])
        results = vector_store.similarity_search(test_text, k=1)
        self.assertGreater(len(results), 0)

if __name__ == '__main__':
    unittest.main() 