from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from .base_agent import BaseAgent
from utils.vector_store import VectorStore
from config.config import REACT_AGENT_TEMPERATURE, REACT_AGENT_MAX_TOKENS
import os

class ReactAgent(BaseAgent):
    """React代码生成Agent"""
    
    def __init__(self, vector_store: VectorStore):
        super().__init__()
        self.vector_store = vector_store
        self.set_model_parameters(temperature=REACT_AGENT_TEMPERATURE, max_tokens=REACT_AGENT_MAX_TOKENS)
        self.analysis_template = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的React代码分析专家。
            你需要分析用户的需求，并提供详细的技术建议。
            分析应该包括：
            1. 组件架构建议
            2. 状态管理策略
            3. 性能优化方案
            4. 可能的技术难点
            5. 测试策略建议
            请确保建议是最新的，并符合当前的最佳实践。"""),  # 改进提示模板
            ("user", "{content}")
        ])
        
    def analyze(self, content: str) -> Dict[str, Any]:
        """分析React相关内容"""
        # 获取相关上下文
        code_examples = self.vector_store.search_react_code(content)
        best_practices = self.vector_store.similarity_search(
            content,
            filter_dict={"type": "best_practices"}
        )
        
        # 构建分析提示
        analysis_input = f"""
        需求内容：
        {content}
        
        相关代码示例：
        {'\n'.join(code_examples)}
        
        最佳实践参考：
        {'\n'.join(best_practices)}
        """
        
        # 使用LLM进行分析
        chain = self.analysis_template | self.llm
        result = chain.invoke({"content": analysis_input})
        
        # 将分析结果添加到记忆
        self.add_to_memory(
            SystemMessage(content="开始新的React组件分析")
        )
        self.add_to_memory(
            HumanMessage(content=content)
        )
        self.add_to_memory(
            AIMessage(content=result.content)
        )
        
        # 解析分析结果
        return {
            "analysis": result.content,
            "code_examples": code_examples,
            "best_practices": best_practices
        }
    
    def execute(self, task: str) -> str:
        """执行React相关任务"""
        # 首先进行分析
        analysis_result = self.analyze(task)
        
        # 构建执行提示
        execution_template = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的React开发专家。
            基于之前的分析结果和最佳实践，你需要：
            1. 提供详细的实现步骤
            2. 指出潜在的问题和解决方案
            3. 提供性能优化建议
            4. 建议适当的测试策略"""),
            ("user", """
            任务描述：{task}
            
            分析结果：{analysis}
            
            请提供具体的实现建议和注意事项。
            """)
        ])
        
        # 使用LLM生成执行建议
        chain = execution_template | self.llm
        result = chain.invoke({
            "task": task,
            "analysis": analysis_result["analysis"]
        })
        
        # 将执行结果添加到记忆
        self.add_to_memory(
            SystemMessage(content="生成执行建议")
        )
        self.add_to_memory(
            AIMessage(content=result.content)
        )
        
        return result.content
    
    def get_conversation_history(self) -> List[str]:
        """获取对话历史"""
        return [msg.content for msg in self.memory]

# 使用 os.path.join 处理路径
file_path = os.path.join('some', 'directory', 'file.txt') 