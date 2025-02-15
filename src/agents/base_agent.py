from typing import Dict, Any, List
from abc import ABC, abstractmethod
from langchain.schema import BaseMessage
from langchain_openai import ChatOpenAI
from config.config import DEFAULT_MODEL_NAME

class BaseAgent(ABC):
    """基础Agent类"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.llm = ChatOpenAI(model_name=model_name)
        self.memory: List[BaseMessage] = []
        
    @abstractmethod
    def execute(self, task: str) -> str:
        """执行任务"""
        pass
    
    @abstractmethod
    def analyze(self, content: str) -> Dict[str, Any]:
        """分析内容"""
        pass
    
    def add_to_memory(self, message: BaseMessage):
        """添加消息到记忆"""
        self.memory.append(message)
        
    def clear_memory(self):
        """清除记忆"""
        self.memory = []

    def set_model_parameters(self, temperature: float = 0.7, max_tokens: int = 150):
        """设置模型参数"""
        self.llm.temperature = temperature
        self.llm.max_tokens = max_tokens 