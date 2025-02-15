import os
import sys
import pytest
from pathlib import Path
from dotenv import load_dotenv

def main():
    """运行测试的主函数"""
    # 设置项目根目录
    project_root = Path(__file__).resolve().parent
    
    # 加载环境变量
    env_path = project_root / '.env'
    if not env_path.exists():
        print(f"错误: 未找到 .env 文件在 {env_path}")
        sys.exit(1)
        
    load_dotenv(env_path)
    
    # 验证环境变量
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("错误: OPENAI_API_KEY 未在环境变量中设置")
        sys.exit(1)
    if api_key == "your-api-key-here":
        print("错误: OPENAI_API_KEY 使用了默认值，请在.env文件中设置正确的值")
        sys.exit(1)
        
    # 运行测试
    pytest.main(["-v", "test/test_project.py"])

if __name__ == "__main__":
    main() 