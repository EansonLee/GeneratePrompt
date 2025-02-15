from typing import Dict, Any, List
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.agent_toolkits.conversational_retrieval.tool import create_retriever_tool
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from utils.vector_store import VectorStore
from config.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    SYSTEM_TEMPLATE,
    AGENT_CONFIG,
    SEARCH_CONFIG,
    PROMPT_OPTIMIZATION_TEMPERATURE,
    PROMPT_OPTIMIZATION_MAX_TOKENS
)

class PromptOptimizationAgent:
    """使用LangChain对话式React代理实现的提示优化Agent"""
    
    def __init__(self, vector_store: VectorStore):
        """初始化代理
        
        Args:
            vector_store: 向量存储实例
        """
        self.vector_store = vector_store
        
        # 初始化LLM，使用特定的温度和最大token设置
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=PROMPT_OPTIMIZATION_TEMPERATURE,
            max_tokens=PROMPT_OPTIMIZATION_MAX_TOKENS
        )
        
        # 创建记忆组件，限制窗口大小以保持对话聚焦
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,  # 保留最近5轮对话
            return_messages=True
        )
        
        # 创建输出解析器
        self.output_parser = self._create_output_parser()
        
        # 创建工具
        self.tools = self._create_tools()
        
        # 创建代理
        self.agent = self._create_agent()
        
    def _create_output_parser(self):
        """创建结构化输出解析器"""
        response_schemas = [
            ResponseSchema(
                name="optimized_prompt",
                description="优化后的提示内容"
            ),
            ResponseSchema(
                name="reasoning",
                description="优化过程中的推理说明"
            ),
            ResponseSchema(
                name="technical_requirements",
                description="技术要求列表"
            ),
            ResponseSchema(
                name="design_guidelines",
                description="设计规范说明"
            )
        ]
        return StructuredOutputParser.from_response_schemas(response_schemas)
    
    def _create_tools(self) -> List[Tool]:
        """创建代理使用的工具集"""
        # 创建检索工具
        retriever_tool = create_retriever_tool(
            self.vector_store.vector_store.as_retriever(
                search_kwargs={"k": SEARCH_CONFIG["k"]}
            ),
            "search_context",
            "搜索项目相关的上下文信息，包括代码示例、最佳实践等"
        )
        
        # 创建React代码搜索工具
        react_code_tool = Tool(
            name="search_react_code",
            func=self.vector_store.search_react_code,
            description="搜索相关的React代码示例"
        )
        
        # 创建最佳实践搜索工具
        best_practices_tool = Tool(
            name="search_best_practices",
            func=self.vector_store.search_best_practices,
            description="搜索相关的React最佳实践"
        )
        
        # 创建历史提示搜索工具
        prompt_history_tool = Tool(
            name="search_prompt_history",
            func=self.vector_store.search_prompt_history,
            description="搜索历史优化过的提示"
        )
        
        return [
            retriever_tool,
            react_code_tool,
            best_practices_tool,
            prompt_history_tool
        ]
    
    def _create_agent(self):
        """创建对话式React代理"""
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,  # 启用详细输出以便调试
            handle_parsing_errors=True,  # 优雅处理解析错误
            max_iterations=5,  # 限制最大迭代次数
            early_stopping_method="generate"  # 使用生成式方法进行早停
        )
    
    def optimize_prompt(self, prompt: str) -> Dict[str, Any]:
        """优化用户输入的提示
        
        Args:
            prompt: 原始提示
            
        Returns:
            Dict[str, Any]: 包含优化后提示和其他信息的结构化输出
        """
        try:
            # 执行优化
            result = self.agent.run(
                f"""请基于项目上下文优化以下提示，并按照指定格式输出结果：
                
                原始提示：{prompt}
                
                请使用可用的工具搜索相关信息，并生成一个更具体、更符合项目需求的提示。
                在优化过程中，请特别注意：
                1. 技术要求的具体性
                2. 设计规范的明确性
                3. 性能和可维护性考虑
                4. 测试和文档要求
                
                {self.output_parser.get_format_instructions()}
                """
            )
            
            # 解析输出
            parsed_output = self.output_parser.parse(result)
            
            # 保存优化历史
            self.vector_store.add_prompt_history(
                prompt, 
                parsed_output["optimized_prompt"]
            )
            
            return parsed_output
            
        except Exception as e:
            raise Exception(f"提示优化失败: {str(e)}")
    
    def get_optimization_history(self) -> List[Dict[str, str]]:
        """获取优化历史
        
        Returns:
            List[Dict[str, str]]: 优化历史记录列表
        """
        return [
            {
                "input": messages[0].content,
                "output": messages[1].content
            }
            for messages in self.memory.chat_memory.messages[::2]
        ] 