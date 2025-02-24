import os
import requests
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
        api_key = "sk-FastAPIvE1M0Ktm0qjx1IZm4LIA1bVdR0mZ0aOtH3BCz2wjn"
        base_url = "https://api.fastapi.ai/v1"  # 添加 /v1 到base_url
            
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
        model_name = "gpt-3.5-turbo"
        
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

def test_embedding_model(api_key: str, base_url: str) -> Tuple[bool, str]:
    """测试文本嵌入模型是否可用
    
    Args:
        api_key: API密钥
        base_url: API基础URL
    
    Returns:
        Tuple[bool, str]: (是否成功, 详细信息)
    """
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "OpenAI/v1 PythonClient/1.0.0"
        }
        
        data = {
            "model": "text-embedding-ada-002",
            "input": "测试文本",
            "encoding_format": "float"
        }
        
        print("\n📡 测试文本嵌入接口...")
        print(f"请求URL: {base_url}/embeddings")
        print(f"请求头: {json.dumps(headers, indent=2, ensure_ascii=False)}")
        print(f"请求体: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        response = requests.post(
            f"{base_url}/embeddings",
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"\n📥 响应状态码: {response.status_code}")
        print(f"响应头: {json.dumps(dict(response.headers), indent=2, ensure_ascii=False)}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"响应内容摘要: {json.dumps({k: v for k, v in response_data.items() if k != 'data'}, indent=2, ensure_ascii=False)}")
            embedding_length = len(response_data.get('data', [{}])[0].get('embedding', []))
            print(f"嵌入向量维度: {embedding_length}")
            return True, f"嵌入模型测试成功，向量维度: {embedding_length}"
        else:
            try:
                error_data = response.json()
                error_msg = f"请求失败: {json.dumps(error_data, indent=2, ensure_ascii=False)}"
            except:
                error_msg = f"请求失败: HTTP {response.status_code}, {response.text}"
            print(f"❌ {error_msg}")
            return False, error_msg
            
    except Exception as e:
        error_msg = f"测试失败: {str(e)}"
        print(f"\n❌ {error_msg}")
        if hasattr(e, '__traceback__'):
            import traceback
            print(f"详细错误信息:\n{traceback.format_exc()}")
        return False, error_msg

def test_embedding_with_langchain(api_key: str, base_url: str) -> Tuple[bool, str]:
    """使用 LangChain API 测试文本嵌入模型
    
    Args:
        api_key: API密钥
        base_url: API基础URL
    
    Returns:
        Tuple[bool, str]: (是否成功, 详细信息)
    """
    try:
        print("\n📡 使用 LangChain API 测试文本嵌入...")
        
        # 初始化 OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=api_key,
            openai_api_base=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "OpenAI/v1 PythonClient/1.0.0"
            },
            timeout=30,
            max_retries=3,
            model_kwargs={
                "encoding_format": "float"
            }
        )
        
        # 测试文本
        test_texts = ["这是第一个测试文本", "这是第二个测试文本"]
        test_query = "测试查询文本"
        
        print("\n🔄 测试 embed_documents...")
        try:
            # 测试文档嵌入
            doc_embeddings = embeddings.embed_documents(test_texts)
            print(f"文档嵌入成功！维度: {len(doc_embeddings[0])}")
            print(f"嵌入数量: {len(doc_embeddings)}")
        except Exception as e:
            print(f"❌ 文档嵌入失败: {str(e)}")
            raise
            
        print("\n🔄 测试 embed_query...")
        try:
            # 测试查询嵌入
            query_embedding = embeddings.embed_query(test_query)
            print(f"查询嵌入成功！维度: {len(query_embedding)}")
        except Exception as e:
            print(f"❌ 查询嵌入失败: {str(e)}")
            raise
            
        # 测试向量相似度
        print("\n🔄 测试向量相似度计算...")
        try:
            import numpy as np
            similarity = np.dot(doc_embeddings[0], query_embedding) / (
                np.linalg.norm(doc_embeddings[0]) * np.linalg.norm(query_embedding)
            )
            print(f"相似度计算成功！相似度分数: {similarity}")
        except Exception as e:
            print(f"❌ 相似度计算失败: {str(e)}")
            raise
            
        return True, "LangChain Embeddings API 测试全部成功！"
        
    except Exception as e:
        error_msg = f"LangChain API 测试失败: {str(e)}"
        print(f"\n❌ {error_msg}")
        if hasattr(e, '__traceback__'):
            import traceback
            print(f"详细错误信息:\n{traceback.format_exc()}")
        return False, error_msg

if __name__ == "__main__":
    # 设置详细的HTTP请求日志
    import http.client as http_client
    http_client.HTTPConnection.debuglevel = 1
    
    # 配置requests的日志
    import logging
    logging.basicConfig(level=logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True
    
    # API配置
    api_key = "sk-FastAPIvE1M0Ktm0qjx1IZm4LIA1bVdR0mZ0aOtH3BCz2wjn"
    base_url = "https://api.fastapi.ai/v1"
    
    # 运行常规API测试
    print("\n=== 测试常规API ===")
    success, message = test_openai_key()
    print(f"\n常规API测试结果: {message}")
    
    # 运行直接嵌入模型测试
    print("\n=== 测试直接调用嵌入模型 ===")
    embedding_success, embedding_message = test_embedding_model(api_key, base_url)
    print(f"\n直接调用嵌入模型测试结果: {embedding_message}")
    
    # 运行 LangChain API 测试
    print("\n=== 测试 LangChain API ===")
    langchain_success, langchain_message = test_embedding_with_langchain(api_key, base_url)
    print(f"\nLangChain API 测试结果: {langchain_message}")
    
    # 根据所有测试的结果决定退出码
    exit(0 if success and embedding_success and langchain_success else 1) 