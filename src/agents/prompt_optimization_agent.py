import logging
from typing import Dict, Any, List, TypedDict, Annotated
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_community.tools import Tool
from src.utils.vector_store import VectorStore
from config.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_BASE_URL,
    SYSTEM_TEMPLATE,
    AGENT_CONFIG,
    SEARCH_CONFIG,
    PROMPT_OPTIMIZATION_TEMPERATURE,
    PROMPT_OPTIMIZATION_MAX_TOKENS,
    PROMPT_OPTIMIZATION_SYSTEM_PROMPT
)
from pydantic import BaseModel, Field
from unittest.mock import Mock, MagicMock
import os
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """Agent状态"""
    messages: List[Dict[str, Any]]
    prompt: str
    contexts: List[Dict[str, str]]
    templates: List[str]
    optimized_prompt: str
    next_step: str

class OptimizedPrompt(BaseModel):
    """优化后的提示结构"""
    optimized_prompt: str = Field(description="优化后的提示内容")
    requirements: Dict[str, Any] = Field(description="提示要求", default_factory=dict)
    
class MockLLM(Runnable):
    """Mock LLM for testing"""
    def invoke(self, input: Any) -> Any:
        # 根据输入内容返回不同的测试响应
        if "按钮组件" in str(input):
            return AIMessage(content="请帮我创建一个React按钮组件，包含以下特性：样式可定制、支持点击事件、支持禁用状态")
        elif "健身训练页面" in str(input):
            return AIMessage(content="""
            请创建一个健身训练页面，包含以下内容：
            1. 健身训练页面详细设计
            2. 页面布局结构
            3. 组件层次关系
            4. 数据流设计
            5. 状态管理方案
            """)
        elif "训练计划页面" in str(input):
            return AIMessage(content="""
            请使用React和TypeScript创建训练计划页面，包含以下功能：
            1. 用户认证集成
            2. 训练计划展示
            3. 进度追踪功能
            """)
        elif "用户资料页面" in str(input):
            return AIMessage(content="""
            请创建用户资料页面，包含以下内容：
            1. 用户资料展示
            2. 页面布局设计
            3. 数据展示组件
            4. 编辑功能实现
            """)
        elif "数据展示页面" in str(input):
            return AIMessage(content="""
            请创建一个数据展示页面，需要满足以下要求：
            1. 实现无障碍设计
            2. 响应式布局
            3. 深色主题
            4. 中文界面
            """)
        else:
            return AIMessage(content="这是一个测试回复")
    
    def batch(self, inputs: List[Any]) -> List[Any]:
        return [self.invoke(input) for input in inputs]

class PromptOptimizationAgent:
    """Prompt优化Agent"""
    
    def __init__(self, vector_store, is_testing=False):
        """初始化优化代理

        Args:
            vector_store: 向量存储实例
            is_testing: 是否为测试模式
        """
        self.vector_store = vector_store
        self.is_testing = is_testing

        if self.is_testing:
            self.llm = MockLLM()
        else:
            self.llm = ChatOpenAI(
                temperature=0.7,
                model_name=OPENAI_MODEL,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL
            )

        # 创建工具列表
        self.tools = [
            Tool(
                name="search_contexts",
                func=self.vector_store.search_contexts,
                description="搜索相关上下文"
            ),
            Tool(
                name="search_templates",
                func=self.vector_store.search_templates,
                description="搜索相关模板"
            )
        ]

        # 创建工具节点
        self.tool_nodes = {
            "search_contexts": ToolNode(tools=[self.tools[0]]),
            "search_templates": ToolNode(tools=[self.tools[1]])
        }

        # 创建工作流图
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """创建工作流图"""
        # 创建状态图
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("get_contexts", self._get_contexts)
        workflow.add_node("get_templates", self._get_templates)
        workflow.add_node("optimize_prompt", self._optimize_prompt)

        # 添加边
        workflow.add_edge("get_contexts", "get_templates")
        workflow.add_edge("get_templates", "optimize_prompt")
        workflow.add_edge("optimize_prompt", END)

        # 设置入口节点
        workflow.set_entry_point("get_contexts")

        # 编译工作流
        return workflow.compile()

    def _get_contexts(self, state: AgentState) -> AgentState:
        """获取上下文"""
        try:
            contexts = self.vector_store.search_contexts(limit=3)
            state["contexts"] = contexts
            state["next_step"] = "get_templates"
            return state
        except Exception as e:
            logger.error(f"获取上下文失败: {str(e)}")
            raise

    def _get_templates(self, state: AgentState) -> AgentState:
        """获取模板"""
        try:
            templates = self.vector_store.search_templates(limit=2)
            state["templates"] = templates
            state["next_step"] = "optimize_prompt"
            return state
        except Exception as e:
            logger.error(f"获取模板失败: {str(e)}")
            raise

    def _optimize_prompt(self, state: AgentState) -> AgentState:
        """优化提示"""
        try:
            # 构建优化提示
            optimization_prompt = f"""
            请基于以下上下文和模板，优化用户的提示:

            上下文:
            {state['contexts']}

            模板:
            {state['templates']}

            用户提示:
            {state['prompt']}

            请生成优化后的提示。要求：
            1. 提示应该更加具体和详细
            2. 包含必要的技术要求和约束
            3. 明确输出的格式和质量标准
            4. 考虑项目的整体上下文
            """

            # 执行优化
            result = self.llm.invoke(optimization_prompt)

            # 提取优化后的提示
            optimized_prompt = str(result.content)

            # 更新状态
            state["optimized_prompt"] = optimized_prompt
            state["next_step"] = END

            # 保存优化历史
            self.vector_store.add_prompt_history(
                state["prompt"],
                optimized_prompt
            )

            return state

        except Exception as e:
            logger.error(f"优化提示失败: {str(e)}")
            raise

    def optimize_prompt(self, prompt: str) -> Dict[str, Any]:
        """优化用户输入的提示

        Args:
            prompt: 原始提示

        Returns:
            Dict[str, Any]: 包含优化后提示和其他信息的结构化输出
        """
        try:
            # 初始化状态
            initial_state: AgentState = {
                "messages": [],
                "prompt": prompt,
                "contexts": [],
                "templates": [],
                "optimized_prompt": "",
                "next_step": "get_contexts"
            }

            # 执行工作流
            final_state = self.workflow.invoke(initial_state)

            # 构造返回结果
            output = {
                "optimized_prompt": final_state["optimized_prompt"],
                "original_prompt": prompt,
                "contexts_used": len(final_state["contexts"]),
                "templates_used": len(final_state["templates"])
            }

            return output

        except Exception as e:
            logger.error(f"提示优化失败: {str(e)}")
            raise Exception(f"提示优化失败: {str(e)}")
            
    def get_optimization_history(self) -> List[Dict[str, str]]:
        """获取优化历史"""
        return self.vector_store.get_prompt_history()
        
    def set_model_parameters(self, temperature: float, max_tokens: int):
        """设置模型参数"""
        if not self.is_testing:
            self.llm.temperature = temperature
            self.llm.max_tokens = max_tokens 