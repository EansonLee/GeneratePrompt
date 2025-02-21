import os
import requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import Tuple, List
import json
import logging

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)

def get_available_models(api_key: str, base_url: str) -> List[str]:
    """获取可用的模型列表
    
    Args:
        api_key: API密钥
        base_url: API基础URL
    
    Returns:
        List[str]: 可用模型列表
    """
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        print("\n📡 获取可用模型列表...")
        response = requests.get(
            f"{base_url}/models",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            models = response.json().get("data", [])
            model_ids = [model["id"] for model in models]
            print("\n✅ 获取模型列表成功！")
            print("\n可用模型:")
            for model_id in model_ids:
                print(f"- {model_id}")
            return model_ids
        else:
            print(f"\n❌ 获取模型列表失败：状态码 {response.status_code}")
            print(f"响应内容: {response.text}")
            return []
            
    except Exception as e:
        print(f"\n❌ 获取模型列表失败：{str(e)}")
        return []

def test_openai_key() -> Tuple[bool, str]:
    """测试OpenAI API密钥是否可用
    
    Returns:
        Tuple[bool, str]: (密钥是否可用, 详细信息)
    """
    try:
        # 直接设置API配置
        api_key = "sk-FastAPITi1j0BFd0pIv5oonN0pozt1mDi0SRz5E3mKCje0DE"
        base_url = "https://api.free.fastapi.ai/v1"  # 添加 /v1 到base_url
            
        print("\n🔄 正在测试API密钥...")
        print(f"Base URL: {base_url}")
        print(f"API Key: {api_key}")

        # 1. 获取可用模型列表
        available_models = get_available_models(api_key, base_url)
        if not available_models:
            return False, "无法获取可用模型列表"

        # 2. 尝试直接调用API
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 使用第一个可用的模型
        model_name = available_models[0] if available_models else "free-gpt-4o"
        
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": "返回'测试成功'这三个字"}],
            "temperature": 0
        }
        
        print("\n📡 直接调用API测试...")
        print(f"使用模型: {model_name}")
        print(f"请求头: {json.dumps(headers, indent=2, ensure_ascii=False)}")
        print(f"请求体: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"\n📥 API响应状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        print(f"响应内容: {response.text}")

        if response.status_code == 200:
            print("\n✅ 直接API调用测试成功！")
            
            # 3. 使用LangChain测试
            print("\n🔄 使用LangChain测试...")
            
            chat = ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                temperature=0,
                timeout=30,
                max_retries=3,
                verbose=True,
                streaming=False,
                default_headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
            )
            
            # 发送测试消息
            test_message = "返回'测试成功'这三个字"
            messages = [HumanMessage(content=test_message)]
            
            print("📤 发送测试消息...")
            response = chat.invoke(messages)
            
            # 验证响应
            if response and response.content:
                print(f"📥 收到响应: {response.content}")
                print("\n✅ LangChain测试成功！")
                return True, "API测试成功（直接调用和LangChain都成功）"
            
        return False, f"API测试失败：状态码 {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        error_msg = f"HTTP请求错误: {str(e)}"
        print(f"\n❌ {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ API测试失败：{error_msg}")
        return False, f"API测试失败：{error_msg}"

if __name__ == "__main__":
    # 运行测试
    success, message = test_openai_key()
    print(f"\n测试结果: {message}")
    exit(0 if success else 1) 