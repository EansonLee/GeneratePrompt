import logging
import argparse
from pathlib import Path
from typing import Dict, Any
from src.prompt_optimizer import PromptOptimizer
from config.config import LOG_LEVEL
from utils.vector_store import VectorStore
from agents.prompt_optimization_agent import PromptOptimizationAgent

# 配置日志
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_project(project_path: str) -> Dict[str, int]:
    """处理整个项目工程
    
    Args:
        project_path: 项目路径
        
    Returns:
        Dict[str, int]: 处理统计信息
    """
    try:
        optimizer = PromptOptimizer()
        logger.info(f"开始处理项目: {project_path}")
        
        # 处理项目目录
        stats = optimizer.process_project_directory(project_path)
        
        # 打印处理统计
        logger.info("项目处理完成！统计信息：")
        logger.info("-" * 40)
        logger.info(f"React组件: {stats['react_component']}个")
        logger.info(f"测试文件: {stats['test']}个")
        logger.info(f"配置文件: {stats['config']}个")
        logger.info(f"文档文件: {stats['doc']}个")
        logger.info(f"其他文件: {stats['other']}个")
        if stats['failed'] > 0:
            logger.warning(f"处理失败: {stats['failed']}个")
        logger.info("-" * 40)
        
        return stats
        
    except Exception as e:
        logger.error(f"项目处理失败: {str(e)}")
        raise

def main():
    """主程序入口"""
    try:
        # 创建优化器实例
        optimizer = PromptOptimizer()
        
        # 示例：添加一些React代码和最佳实践
        optimizer.add_react_code(
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
                "author": "demo"
            }
        )
        
        optimizer.add_best_practice(
            practice="""
            React组件应该遵循单一职责原则，每个组件只负责一个特定的功能。
            这样可以提高代码的可维护性和复用性。
            """,
            category="架构"
        )
        
        # 示例：优化提示
        result = optimizer.optimize("生成一个用户管理页面")
        print("\n优化结果:")
        print(f"原始提示: {result['original_prompt']}")
        print(f"优化后提示: {result['optimized_prompt']}")
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 