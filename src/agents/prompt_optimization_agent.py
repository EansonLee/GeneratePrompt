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
from datetime import datetime

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
    """提示词优化Agent"""
    
    def __init__(
            self,
            vector_store: VectorStore,
            is_testing: bool = False,
            temperature: float = PROMPT_OPTIMIZATION_TEMPERATURE,
            max_tokens: int = PROMPT_OPTIMIZATION_MAX_TOKENS
        ):
        self.vector_store = vector_store
        self.is_testing = is_testing
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.optimization_history = []
        
        # 初始化LLM
        self.llm = ChatOpenAI(
            model_name=OPENAI_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        
    async def optimize_prompt(self, prompt: str) -> str:
        """优化提示词
        
        Args:
            prompt: 原始提示词
            
        Returns:
            str: 优化后的提示词
        """
        try:
            # 获取相关上下文
            contexts = self.vector_store.search_contexts(limit=5)
            
            # 构建提示
            system_prompt = PROMPT_OPTIMIZATION_SYSTEM_PROMPT
            user_prompt = f"""请根据以下信息优化提示词：

原始提示词：
{prompt}

相关上下文：
{self._format_contexts(contexts)}

请确保优化后的提示词：
1. 更加清晰和具体
2. 包含必要的上下文信息
3. 结构合理，易于理解
4. 符合最佳实践
"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 调用LLM
            response = await self.llm.ainvoke(messages)
            optimized = response.content
            
            # 记录历史
            self.optimization_history.append({
                "input": prompt,
                "output": optimized,
                "timestamp": datetime.now().isoformat()
            })
            
            return optimized
            
        except Exception as e:
            logger.error(f"优化提示词失败: {str(e)}")
            raise
            
    async def execute(self, prompt: str) -> str:
        """执行优化
        
        Args:
            prompt: 原始提示词
            
        Returns:
            str: 优化后的提示词
        """
        try:
            # 分析提示词
            analysis = await self._analyze_prompt(prompt)
            
            # 获取相关最佳实践
            best_practices = self.vector_store.search_best_practices(prompt)
            
            # 优化提示词
            optimized = await self.optimize_prompt(prompt)
            
            # 验证优化结果
            validation = await self._validate_optimization(prompt, optimized)
            
            if not validation["is_valid"]:
                logger.warning(f"优化结果验证失败: {validation['reason']}")
                return prompt
                
            return optimized
            
        except Exception as e:
            logger.error(f"执行优化失败: {str(e)}")
            return prompt
            
    async def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """分析提示词
        
        Args:
            prompt: 提示词
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            messages = [
                {"role": "system", "content": "你是一个提示词分析专家。"},
                {"role": "user", "content": f"请分析以下提示词的优缺点：\n\n{prompt}"}
            ]
            
            response = await self.llm.ainvoke(messages)
            
            return {
                "analysis": response.content,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"分析提示词失败: {str(e)}")
            return {
                "analysis": "分析失败",
                "error": str(e)
            }
            
    async def _validate_optimization(self, original: str, optimized: str) -> Dict[str, Any]:
        """验证优化结果
        
        Args:
            original: 原始提示词
            optimized: 优化后的提示词
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        try:
            messages = [
                {"role": "system", "content": "你是一个提示词优化验证专家。"},
                {"role": "user", "content": f"""请验证优化结果是否有效：

原始提示词：
{original}

优化后的提示词：
{optimized}

请判断：
1. 优化后的提示词是否保留了原始意图
2. 是否有效改进了原始提示词
3. 是否引入了不必要的复杂性"""}
            ]
            
            response = await self.llm.ainvoke(messages)
            content = response.content.lower()
            
            is_valid = "有效" in content or "改进" in content
            reason = response.content
            
            return {
                "is_valid": is_valid,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"验证优化结果失败: {str(e)}")
            return {
                "is_valid": True,  # 出错时默认通过
                "reason": f"验证过程出错: {str(e)}"
            }
            
    def _format_contexts(self, contexts: List[Dict[str, str]]) -> str:
        """格式化上下文
        
        Args:
            contexts: 上下文列表
            
        Returns:
            str: 格式化后的上下文
        """
        result = []
        for ctx in contexts:
            result.append(f"- {ctx.get('content', '')}")
        return "\n".join(result)
        
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """获取优化历史
        
        Returns:
            List[Dict[str, Any]]: 优化历史记录
        """
        return self.optimization_history
        
    def clear_memory(self):
        """清除历史记录"""
        self.optimization_history.clear()
        
    def set_model_parameters(self, temperature: float, max_tokens: int):
        """设置模型参数
        
        Args:
            temperature: 温度参数
            max_tokens: 最大token数
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = ChatOpenAI(
            model_name=OPENAI_MODEL,
            temperature=temperature,
            max_tokens=max_tokens
        ) 