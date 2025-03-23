import requests
import json
import traceback

url = "http://localhost:8000/api/design/generate"
data = {
    "tech_stack": "Android",
    "design_image_id": "c74a4dfe-8740-4167-a5df-281dc60f5413",
    "design_image_path": "F:\\generatePrompt\\GeneratePrompt\\uploads\\design_images\\c74a4dfe-8740-4167-a5df-281dc60f5413.png",
    "rag_method": "similarity",
    "retriever_top_k": 3,
    "agent_type": "ReActAgent",
    "temperature": 0.5,
    "context_window_size": 4000,
    "prompt": "测试提示词"
}

try:
    print(f"发送请求到: {url}")
    print(f"请求数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
    
    response = requests.post(url, json=data)
    
    print(f"状态码: {response.status_code}")
    print(f"响应头: {response.headers}")
    
    if response.status_code == 422:
        print("422 错误详情:")
        error_detail = response.json()
        print(json.dumps(error_detail, ensure_ascii=False, indent=2))
    else:
        print(f"响应内容: {response.text}")
except Exception as e:
    print(f"发生错误: {str(e)}")
    traceback.print_exc() 