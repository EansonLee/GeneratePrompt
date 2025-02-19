import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from src.prompt_optimizer import PromptOptimizer
from src.template_generator import TemplateGenerator
from config.config import LOG_LEVEL
from utils.vector_store import VectorStore
from agents.prompt_optimization_agent import PromptOptimizationAgent

# 配置日志
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_files(file_paths: list[str]) -> Dict[str, Any]:
    """处理上传的文件
    
    Args:
        file_paths: 文件路径列表
        
    Returns:
        Dict[str, Any]: 处理结果
    """
    try:
        vector_store = VectorStore()
        results = {
            'processed': 0,
            'failed': 0,
            'contexts': []
        }
        
        for file_path in file_paths:
            try:
                # 处理文件并存入向量数据库
                context = vector_store.add_file(file_path)
                results['processed'] += 1
                results['contexts'].append(context)
            except Exception as e:
                logger.error(f"处理文件失败 {file_path}: {str(e)}")
                results['failed'] += 1
                
        return results
    except Exception as e:
        logger.error(f"文件处理过程失败: {str(e)}")
        raise

def generate_template(file_paths: Optional[list[str]] = None) -> str:
    """生成模板prompt
    
    Args:
        file_paths: 可选的上下文文件路径列表
        
    Returns:
        str: 生成的模板prompt
    """
    try:
        generator = TemplateGenerator()
        
        # 如果提供了文件，先处理文件
        if file_paths:
            process_files(file_paths)
            
        # 生成模板
        template = generator.generate()
        logger.info("模板生成成功")
        return template
        
    except Exception as e:
        logger.error(f"生成模板失败: {str(e)}")
        raise

def optimize_prompt(prompt: str, file_paths: Optional[list[str]] = None) -> str:
    """优化用户输入的prompt
    
    Args:
        prompt: 用户输入的prompt
        file_paths: 可选的上下文文件路径列表
        
    Returns:
        str: 优化后的prompt
    """
    try:
        optimizer = PromptOptimizer()
        
        # 如果提供了文件，先处理文件
        if file_paths:
            process_files(file_paths)
            
        # 优化prompt
        optimized = optimizer.optimize(prompt)
        logger.info("Prompt优化成功")
        return optimized
        
    except Exception as e:
        logger.error(f"优化prompt失败: {str(e)}")
        raise

def main():
    """主程序入口"""
    try:
        print("\n欢迎使用Prompt优化系统！")
        print("\n请选择功能：")
        print("1. 生成模板prompt")
        print("2. 优化用户输入的prompt")
        
        choice = input("\n请输入选项（1或2）：")
        
        if choice not in ['1', '2']:
            print("无效的选择！请输入1或2")
            return
            
        # 询问是否需要上传文件
        need_files = input("\n是否需要上传上下文文件？(y/n): ").lower() == 'y'
        file_paths = []
        
        if need_files:
            print("\n请输入文件路径（每行一个，输入空行结束）：")
            while True:
                path = input().strip()
                if not path:
                    break
                if Path(path).exists():
                    file_paths.append(path)
                else:
                    print(f"警告：文件 {path} 不存在")
        
        if choice == '1':
            # 生成模板prompt
            template = generate_template(file_paths if need_files else None)
            print("\n生成的模板prompt：")
            print("-" * 40)
            print(template)
            print("-" * 40)
            
            # 询问是否需要修改
            if input("\n是否需要修改模板？(y/n): ").lower() == 'y':
                print("\n请输入修改后的模板：")
                template = input().strip()
            
            # 保存到向量数据库
            vector_store = VectorStore()
            vector_store.add_template(template)
            print("\n模板已保存到向量数据库")
            
        else:
            # 优化用户输入的prompt
            print("\n请输入需要优化的prompt：")
            user_prompt = input().strip()
            
            optimized = optimize_prompt(user_prompt, file_paths if need_files else None)
            print("\n优化后的prompt：")
            print("-" * 40)
            print(optimized)
            print("-" * 40)
            
            # 询问是否需要修改
            if input("\n是否需要修改优化后的prompt？(y/n): ").lower() == 'y':
                print("\n请输入修改后的prompt：")
                optimized = input().strip()
            
            # 保存到向量数据库
            vector_store = VectorStore()
            vector_store.add_optimized_prompt(optimized)
            print("\n优化后的prompt已保存到向量数据库")
            
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 