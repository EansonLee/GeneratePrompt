import logging
from typing import Dict, Any, List, TypedDict, Annotated, Optional, Tuple, Literal
from datetime import datetime
import uuid
import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from src.utils.vector_store import VectorStore
from src.utils.design_image_processor import DesignImageProcessor
from config.config import settings

logger = logging.getLogger(__name__)

# 定义Agent状态
class DesignPromptState(TypedDict):
    """设计图Prompt生成Agent状态"""
    messages: List[Dict[str, Any]]  # 消息历史
    tech_stack: str  # 技术栈
    design_image_id: str  # 设计图ID
    design_image_path: str  # 设计图路径
    similar_designs: List[Dict[str, Any]]  # 相似设计图
    history_prompts: List[Dict[str, Any]]  # 历史Prompt
    rag_method: str  # RAG方法
    retriever_top_k: int  # 检索结果数量
    agent_type: str  # Agent类型
    temperature: float  # 温度
    context_window_size: int  # 上下文窗口大小
    generated_prompt: str  # 生成的Prompt
    next_step: str  # 下一步
    design_analysis: str  # 设计图分析结果

# 定义系统提示模板
SYSTEM_PROMPT = """你是一个专业的设计图Prompt生成专家，擅长将设计图转换为详细的开发提示词。
你的任务是根据用户上传的设计图和选择的技术栈({tech_stack})，生成一个详细的开发提示词。

请遵循以下指南：
1. 仔细分析设计图的UI元素、布局和交互
2. 考虑{tech_stack}技术栈的特性和最佳实践
3. 生成的提示词应包含：
   - 设计图的详细描述
   - UI组件的层次结构
   - 布局和排版要求
   - 交互和动画效果
   - 技术实现建议
   - 适配和响应式设计要求

历史相似设计图的提示词可以作为参考，但请确保生成的提示词针对当前设计图的特点。

相似设计图提示词参考：
{similar_designs}
"""

class DesignPromptAgent:
    """设计图Prompt生成Agent"""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        design_processor: Optional[DesignImageProcessor] = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        """初始化Agent
        
        Args:
            vector_store: 向量存储实例
            design_processor: 设计图处理器
            temperature: 温度参数
            max_tokens: 最大token数
        """
        self.vector_store = vector_store or VectorStore()
        self.design_processor = design_processor or DesignImageProcessor(vector_store=self.vector_store)
        
        # 使用传入的参数或配置文件中的默认值
        self.temperature = temperature or settings.PROMPT_OPTIMIZATION_CONFIG["temperature"]
        self.max_tokens = max_tokens or settings.PROMPT_OPTIMIZATION_CONFIG["max_tokens"]
        
        # 初始化LLM
        self.llm = ChatOpenAI(
            model_name=settings.DESIGN_PROMPT_CONFIG.get("model_name", settings.DEFAULT_MODEL_NAME),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        )
        
        # 初始化工作流图
        self.workflow = self._build_workflow()
        
        logger.info("设计图Prompt生成Agent初始化完成")
    
    def _build_workflow(self) -> StateGraph:
        """构建工作流图
        
        Returns:
            StateGraph: 工作流图
        """
        # 创建工作流图
        workflow = StateGraph(DesignPromptState)
        
        # 添加节点
        workflow.add_node("retrieve_similar_designs", self._retrieve_similar_designs)
        workflow.add_node("retrieve_history_prompts", self._retrieve_history_prompts)
        workflow.add_node("analyze_design", self._analyze_design)
        workflow.add_node("generate_prompt", self._generate_prompt)
        
        # 定义工作流
        workflow.add_edge("retrieve_similar_designs", "retrieve_history_prompts")
        workflow.add_edge("retrieve_history_prompts", "analyze_design")
        workflow.add_edge("analyze_design", "generate_prompt")
        workflow.add_edge("generate_prompt", END)
        
        # 设置入口节点
        workflow.set_entry_point("retrieve_similar_designs")
        
        # 编译工作流
        return workflow.compile()
    
    async def _retrieve_similar_designs(self, state: DesignPromptState) -> DesignPromptState:
        """检索相似设计图
        
        Args:
            state: 当前状态
            
        Returns:
            DesignPromptState: 更新后的状态
        """
        try:
            logger.info(f"检索相似设计图，技术栈: {state['tech_stack']}")
            
            # 检索相似设计图
            similar_designs = await self.design_processor.search_similar_designs(
                tech_stack=state["tech_stack"],
                limit=state["retriever_top_k"]
            )
            
            # 更新状态
            state["similar_designs"] = similar_designs
            state["next_step"] = "retrieve_history_prompts"
            
            # 添加系统消息
            messages = state.get("messages", [])
            messages.append({
                "role": "system",
                "content": f"已检索到 {len(similar_designs)} 个相似设计图"
            })
            state["messages"] = messages
            
            return state
            
        except Exception as e:
            logger.error(f"检索相似设计图失败: {str(e)}")
            
            # 更新状态
            state["similar_designs"] = []
            state["next_step"] = "retrieve_history_prompts"
            
            # 添加错误消息
            messages = state.get("messages", [])
            messages.append({
                "role": "system",
                "content": f"检索相似设计图失败: {str(e)}"
            })
            state["messages"] = messages
            
            return state
    
    async def _retrieve_history_prompts(self, state: DesignPromptState) -> DesignPromptState:
        """检索历史Prompt
        
        Args:
            state: 当前状态
            
        Returns:
            DesignPromptState: 更新后的状态
        """
        try:
            logger.info(f"检索历史Prompt，技术栈: {state['tech_stack']}")
            
            # 构建查询
            query = f"技术栈: {state['tech_stack']} 设计图Prompt"
            
            # 检索历史Prompt
            history_prompts = await self.vector_store.search_texts(
                query=query,
                limit=state["retriever_top_k"],
                search_type=state["rag_method"]
            )
            
            # 过滤结果，只返回设计图Prompt
            filtered_prompts = []
            for prompt in history_prompts:
                metadata = prompt.get("metadata", {})
                if metadata.get("type") == "design_prompt" and metadata.get("tech_stack") == state["tech_stack"]:
                    filtered_prompts.append(prompt)
            
            # 更新状态
            state["history_prompts"] = filtered_prompts
            state["next_step"] = "analyze_design"
            
            # 添加系统消息
            messages = state.get("messages", [])
            messages.append({
                "role": "system",
                "content": f"已检索到 {len(filtered_prompts)} 个历史Prompt"
            })
            state["messages"] = messages
            
            return state
            
        except Exception as e:
            logger.error(f"检索历史Prompt失败: {str(e)}")
            
            # 更新状态
            state["history_prompts"] = []
            state["next_step"] = "analyze_design"
            
            # 添加错误消息
            messages = state.get("messages", [])
            messages.append({
                "role": "system",
                "content": f"检索历史Prompt失败: {str(e)}"
            })
            state["messages"] = messages
            
            return state
    
    async def _analyze_design(self, state: DesignPromptState) -> DesignPromptState:
        """分析设计图
        
        Args:
            state: 当前状态
            
        Returns:
            DesignPromptState: 更新后的状态
        """
        try:
            logger.info(f"开始分析设计图: {state['design_image_id']}")
            
            # 验证设计图路径
            if not os.path.exists(state["design_image_path"]):
                raise ValueError(f"设计图文件不存在: {state['design_image_path']}")
            
            # 读取图片数据
            with open(state["design_image_path"], "rb") as f:
                image_data = f.read()
            
            # 分析设计图
            result = await self.design_processor.process_image(
                image_data=image_data,
                filename=os.path.basename(state["design_image_path"]),
                tech_stack=state["tech_stack"]
            )
            
            if not result.get("success", False):
                raise ValueError(result.get("error", "设计图分析失败"))
                
            # 添加分析结果到状态
            state["design_analysis"] = result["analysis"]
            
            # 添加系统消息
            messages = state.get("messages", [])
            messages.append({
                "role": "system",
                "content": f"设计图分析完成，分析结果长度: {len(result['analysis'])} 字符"
            })
            messages.append({
                "role": "assistant",
                "content": result["analysis"]
            })
            state["messages"] = messages
            state["next_step"] = "generate_prompt"
            
            return state
            
        except Exception as e:
            logger.error(f"分析设计图失败: {str(e)}")
            logger.error(f"设计图路径: {state.get('design_image_path')}")
            logger.error(f"技术栈: {state.get('tech_stack')}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            
            # 使用默认的分析结果
            default_analysis = f"""基于{state['tech_stack']}平台的界面分析：

1. 页面整体布局和结构：
   - 请检查设计图文件是否正确上传
   - 当前无法获取具体的设计图分析结果

2. UI组件和样式：
   - 建议重新上传设计图
   - 确保图片格式正确（支持jpg、png、webp）
   - 图片大小不超过5MB

3. 颜色主题：
   - 暂时无法分析具体的颜色主题
   - 建议使用{state['tech_stack']}标准设计规范

4. 交互设计：
   - 暂时无法分析具体的交互设计
   - 建议参考{state['tech_stack']}官方交互指南

5. 注意事项：
   - 请确保网络连接正常
   - 检查API密钥配置
   - 如果问题持续，请联系技术支持"""
            
            state["design_analysis"] = default_analysis
            messages = state.get("messages", [])
            messages.append({
                "role": "system",
                "content": f"分析设计图失败: {str(e)}\n使用默认分析模板。"
            })
            messages.append({
                "role": "assistant",
                "content": default_analysis
            })
            state["messages"] = messages
            state["next_step"] = "generate_prompt"
            return state
    
    async def _generate_prompt(self, state: DesignPromptState) -> DesignPromptState:
        """生成Prompt
        
        Args:
            state: 当前状态
            
        Returns:
            DesignPromptState: 更新后的状态
        """
        try:
            logger.info("生成设计图Prompt")
            
            # 验证是否有设计图分析结果
            if not state.get("design_analysis"):
                raise ValueError("没有获取到设计图分析结果")
            
            # 格式化相似设计图信息
            similar_designs_text = self._format_similar_designs(state["similar_designs"])
            
            # 格式化历史Prompt信息
            history_prompts_text = self._format_history_prompts(state["history_prompts"])
            
            # 创建提示模板
            system_prompt = """你是一个专业的设计图Prompt生成专家。请基于以下设计图分析结果，生成一个详细的开发提示词。
要求提示词必须：
1. 完全基于提供的设计图分析结果
2. 包含所有具体的UI组件和它们的确切位置、样式
3. 详细描述实际的颜色值、字体大小、间距等具体数值
4. 明确指出所有交互行为和动画效果的具体实现方式
5. 提供符合技术栈特性的具体实现建议
6. 说明适配要求和响应式设计方案

设计图分析结果：
{design_analysis}

技术栈：{tech_stack}

相似设计参考：
{similar_designs}

请生成一个结构化的开发提示词，包含以下部分：
1. 设计图详细描述
2. UI组件层次结构（包含具体的布局类型和属性）
3. 布局和排版要求（包含具体的尺寸、边距、对齐方式）
4. 交互和动画效果（包含具体的触发条件和动画参数）
5. 技术实现建议（针对该技术栈的具体实现方案）
6. 适配和响应式设计要求（具体的断点和适配策略）"""
            
            # 如果有历史Prompt，添加到系统提示中
            if history_prompts_text:
                system_prompt += "\n\n历史Prompt参考：\n{history_prompts}"
            
            prompt = ChatPromptTemplate.from_template(system_prompt).format(
                tech_stack=state["tech_stack"],
                similar_designs=similar_designs_text,
                history_prompts=history_prompts_text,
                design_analysis=state["design_analysis"]
            )
            
            # 使用较低的温度以获得更精确的结果
            self.llm.temperature = 0.3
            
            # 生成Prompt
            generated_prompt = await self.llm.ainvoke(prompt)
            
            # 更新状态
            state["generated_prompt"] = generated_prompt.content
            
            # 添加AI消息
            messages = state.get("messages", [])
            messages.append({
                "role": "ai",
                "content": generated_prompt.content
            })
            state["messages"] = messages
            
            return state
            
        except Exception as e:
            logger.error(f"生成Prompt失败: {str(e)}")
            
            # 更新状态
            state["generated_prompt"] = f"生成Prompt失败: {str(e)}"
            
            # 添加错误消息
            messages = state.get("messages", [])
            messages.append({
                "role": "system",
                "content": f"生成Prompt失败: {str(e)}"
            })
            state["messages"] = messages
            
            return state
    
    def _format_similar_designs(self, similar_designs: List[Dict[str, Any]]) -> str:
        """格式化相似设计图信息
        
        Args:
            similar_designs: 相似设计图列表
            
        Returns:
            str: 格式化后的相似设计图信息
        """
        if not similar_designs:
            return "没有找到相似的设计图。"
            
        formatted_text = ""
        for i, design in enumerate(similar_designs, 1):
            metadata = design.get("metadata", {})
            prompt = design.get("prompt", "无提示词")
            tech_stack = metadata.get("tech_stack", "未知")
            
            formatted_text += f"示例 {i}（技术栈: {tech_stack}）:\n{prompt}\n\n"
            
        return formatted_text
    
    def _format_history_prompts(self, history_prompts: List[Dict[str, Any]]) -> str:
        """格式化历史Prompt信息
        
        Args:
            history_prompts: 历史Prompt列表
            
        Returns:
            str: 格式化后的历史Prompt信息
        """
        if not history_prompts:
            return "没有找到相关的历史Prompt。"
            
        formatted_text = ""
        for i, prompt_data in enumerate(history_prompts, 1):
            metadata = prompt_data.get("metadata", {})
            prompt = prompt_data.get("text", "无提示词")
            tech_stack = metadata.get("tech_stack", "未知")
            created_at = metadata.get("created_at", "未知时间")
            user_modified = metadata.get("user_modified", False)
            
            formatted_text += f"历史Prompt {i}（技术栈: {tech_stack}, 创建时间: {created_at}, 用户修改: {'是' if user_modified else '否'}）:\n{prompt}\n\n"
            
        return formatted_text
    
    async def generate_design_prompt(
        self,
        tech_stack: str,
        design_image_id: str,
        design_image_path: str,
        rag_method: str = "similarity",
        retriever_top_k: int = 3,
        agent_type: str = "ReActAgent",
        temperature: float = 0.7,
        context_window_size: int = 4000
    ) -> Dict[str, Any]:
        """生成设计图Prompt
        
        Args:
            tech_stack: 技术栈
            design_image_id: 设计图ID
            design_image_path: 设计图路径
            rag_method: RAG方法
            retriever_top_k: 检索结果数量
            agent_type: Agent类型
            temperature: 温度
            context_window_size: 上下文窗口大小
            
        Returns:
            Dict[str, Any]: 生成结果
        """
        try:
            # 设置初始状态
            initial_state = DesignPromptState(
                messages=[],
                tech_stack=tech_stack,
                design_image_id=design_image_id,
                design_image_path=design_image_path,
                similar_designs=[],
                history_prompts=[],
                rag_method=rag_method,
                retriever_top_k=retriever_top_k,
                agent_type=agent_type,
                temperature=temperature,
                context_window_size=context_window_size,
                generated_prompt="",
                next_step="retrieve_similar_designs",
                design_analysis=""
            )
            
            # 执行工作流
            result = await self.workflow.ainvoke(initial_state)
            
            # 保存生成的Prompt到向量数据库
            if result["generated_prompt"]:
                await self.vector_store.add_texts(
                    texts=[result["generated_prompt"]],
                    metadatas=[{
                        "id": str(uuid.uuid4()),
                        "tech_stack": tech_stack,
                        "design_image_id": design_image_id,
                        "type": "design_prompt",
                        "created_at": datetime.now().isoformat()
                    }]
                )
            
            return {
                "generated_prompt": result["generated_prompt"],
                "tech_stack": tech_stack,
                "design_image_id": design_image_id,
                "messages": result["messages"],
                "similar_designs": result["similar_designs"],
                "history_prompts": result["history_prompts"],
                "has_history_context": len(result["history_prompts"]) > 0,
                "design_analysis": result["design_analysis"]
            }
            
        except Exception as e:
            logger.error(f"生成设计图Prompt失败: {str(e)}")
            return {
                "error": str(e),
                "tech_stack": tech_stack,
                "design_image_id": design_image_id
            }
    
    async def save_user_modified_prompt(
        self,
        prompt: str,
        tech_stack: str,
        design_image_id: str
    ) -> Dict[str, Any]:
        """保存用户修改后的Prompt
        
        Args:
            prompt: 修改后的Prompt
            tech_stack: 技术栈
            design_image_id: 设计图ID
            
        Returns:
            Dict[str, Any]: 保存结果
        """
        try:
            # 生成唯一ID
            prompt_id = str(uuid.uuid4())
            
            # 保存到向量数据库
            await self.vector_store.add_texts(
                texts=[prompt],
                metadatas=[{
                    "id": prompt_id,
                    "tech_stack": tech_stack,
                    "design_image_id": design_image_id,
                    "type": "design_prompt",
                    "user_modified": True,
                    "created_at": datetime.now().isoformat()
                }],
                ids=[prompt_id]
            )
            
            return {
                "id": prompt_id,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"保存用户修改后的Prompt失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            } 