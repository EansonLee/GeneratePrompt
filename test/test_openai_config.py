import os
import unittest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

class TestOpenAIConfig(unittest.TestCase):
    """测试OpenAI配置"""

    @classmethod
    def setUpClass(cls):
        """测试前的设置"""
        # 加载环境变量
        load_dotenv()
        cls.api_key = os.getenv("OPENAI_API_KEY")
        cls.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        
        # 设置环境变量
        os.environ["OPENAI_API_KEY"] = cls.api_key
        print("\n开始运行OpenAI配置测试...")

    def test_api_key_exists(self):
        """测试API密钥是否存在"""
        self.assertIsNotNone(self.api_key, "OpenAI API密钥未设置")
        self.assertTrue(len(self.api_key) > 0, "OpenAI API密钥为空")
        print(f"\nAPI密钥验证成功")

    def test_api_key_format(self):
        """测试API密钥格式"""
        self.assertTrue(self.api_key.startswith("sk-"), "OpenAI API密钥格式不正确")
        print(f"API密钥格式验证成功")

    def test_chat_completion(self):
        """测试API调用是否成功"""
        try:
            print("\n开始测试API调用...")
            
            # 创建LangChain聊天模型，使用最简单的初始化方式
            chat = ChatOpenAI(
                model_name=self.model,
                openai_api_key=self.api_key,
                temperature=0
            )
            print(f"成功创建ChatOpenAI实例，使用模型: {self.model}")

            # 创建输出解析器
            output_parser = StrOutputParser()
            print("成功创建输出解析器")

            # 创建消息
            test_message = "你好，这是一个测试消息。请回复：'测试成功'"
            messages = [
                HumanMessage(content=test_message)
            ]
            print(f"发送测试消息: {test_message}")

            # 发送请求
            print("正在等待API响应...")
            response = chat.invoke(messages)
            result = output_parser.invoke(response.content)
            print(f"收到API响应: {result}")

            # 验证响应
            self.assertIsNotNone(response, "未收到API响应")
            self.assertTrue(len(response.content) > 0, "API响应内容为空")

            # 测试链式调用
            print("\n开始测试链式调用...")
            chain = chat | output_parser
            chain_message = "请回复：'链式调用测试成功'"
            print(f"发送链式调用测试消息: {chain_message}")
            chain_result = chain.invoke(chain_message)
            print(f"链式调用响应: {chain_result}")

            print("\nAPI调用测试全部成功！")

        except Exception as e:
            self.fail(f"API调用失败: {str(e)}")

if __name__ == '__main__':
    unittest.main(verbosity=2) 