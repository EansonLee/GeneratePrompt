import unittest
import sys
from pathlib import Path
from functools import wraps

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from config.config import settings
from src.prompt_optimizer import PromptOptimizer
from src.agents.prompt_optimization_agent import PromptOptimizationAgent
from utils.vector_store import VectorStore

def validate_api_key(func):
    """验证API密钥的装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY 未设置")
        if api_key == "your-api-key-here":
            raise ValueError("OPENAI_API_KEY 使用了默认值，请在config.py文件中设置正确的值")
        return func(*args, **kwargs)
    return wrapper

class TestProjectWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """测试类初始化时的设置"""
        super().setUpClass()
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY 未设置")
        if api_key == "your-api-key-here":
            raise ValueError("OPENAI_API_KEY 使用了默认值，请在config.py文件中设置正确的值")

    @validate_api_key
    def setUp(self):
        """测试前的设置"""
        self.optimizer = PromptOptimizer()
        self.test_prompt = "生成一个用户管理页面"
        
    @validate_api_key
    def test_prompt_optimizer_initialization(self):
        """测试PromptOptimizer初始化"""
        self.assertIsNotNone(self.optimizer)
        
    @validate_api_key
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
        
    @validate_api_key
    def test_project_processing(self):
        """测试项目处理功能"""
        test_project_path = Path(project_root) / "test_data"
        test_project_path.mkdir(exist_ok=True)
            
        try:
            stats = self.optimizer.process_project_directory(str(test_project_path))
            self.assertIsInstance(stats, dict)
            self.assertIn('react_component', stats)
            self.assertIn('test', stats)
            self.assertIn('config', stats)
        finally:
            # 清理测试数据
            if test_project_path.exists():
                import shutil
                shutil.rmtree(test_project_path)
                
    @validate_api_key
    def test_vector_store(self):
        """测试向量存储功能"""
        vector_store = VectorStore()
        test_text = "这是一个测试文本"
        vector_store.add_texts([test_text], [{"type": "test"}])
        results = vector_store.similarity_search(test_text, k=1)
        self.assertGreater(len(results), 0)

if __name__ == '__main__':
    unittest.main() 