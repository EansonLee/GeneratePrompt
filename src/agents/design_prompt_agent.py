import logging
from typing import Dict, Any, List, TypedDict, Annotated, Optional, Tuple, Literal
from datetime import datetime
import uuid
import json
import os
import time
import hashlib
from pathlib import Path
import asyncio

from langchain_openai import ChatOpenAI
from openai import OpenAI
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

# 确保必要的目录存在
def ensure_directories():
    """确保所有必要的目录存在"""
    dirs_to_check = [
        settings.DATA_DIR,
        settings.DATA_DIR / "vector_store",
        settings.DATA_DIR / "vector_store/designs",
        settings.DATA_DIR / "vector_store/prompts",
        settings.DATA_DIR / "vector_store/templates",
        settings.DATA_DIR / "vector_store/contexts",
        settings.UPLOAD_DIR,
        settings.LOG_DIR
    ]
    
    for directory in dirs_to_check:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"确认目录存在: {directory}")
        except Exception as e:
            logger.error(f"创建目录失败: {directory}, 错误: {str(e)}")
    
    logger.info("已确认所有必要的目录存在")

# 定义Agent状态
class DesignPromptState(TypedDict):
    """设计图Prompt生成Agent状态"""
    messages: List[Dict[str, Any]]  # 消息历史
    tech_stack: str  # 技术栈
    design_image_id: str  # 设计图ID
    design_image_path: str  # 设计图路径
    similar_designs: List[Dict[str, Any]]  # 相似设计图列表
    history_prompts: List[Dict[str, Any]]  # 历史Prompt列表
    rag_method: str  # RAG方法
    retriever_top_k: int  # 检索结果数量
    agent_type: str  # Agent类型
    temperature: float  # 温度
    context_window_size: int  # 上下文窗口大小
    generated_prompt: str  # 生成的Prompt
    next_step: str  # 下一步
    design_analysis: str  # 设计图分析
    image_hash: Optional[str]  # 设计图哈希（用于缓存）

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
    
    # 添加缓存相关的类变量
    _prompt_cache = {}  # 内存缓存：{图像哈希}_{技术栈} -> 生成的Prompt
    _prompt_cache_file = None  # 缓存文件路径
    _cache_expiry = 30 * 24 * 60 * 60  # 缓存过期时间（30天）
    _max_cache_size = 50  # 最大缓存条目数
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        design_processor: Optional[DesignImageProcessor] = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        """初始化设计图Prompt生成Agent
        
        Args:
            vector_store: 向量存储对象
            design_processor: 设计图处理器
            temperature: 温度
            max_tokens: 最大Token数
        """
        # 确保所有必要的目录存在
        ensure_directories()
        
        # 记录模型配置信息
        logger.info("初始化DesignPromptAgent")
        logger.info(f"环境配置检查 - OPENAI_API_KEY: {'已设置' if settings.OPENAI_API_KEY else '未设置'}")
        logger.info(f"环境配置检查 - DESIGN_PROMPT_MODEL: {settings.DESIGN_PROMPT_MODEL or '未设置'}")
        logger.info(f"环境配置检查 - OPENAI_BASE_URL: {settings.OPENAI_BASE_URL or '未设置'}")
        
        # 检查必要的配置
        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY未设置，这将影响提示词生成功能")
            
        if not settings.DESIGN_PROMPT_MODEL:
            logger.warning("DESIGN_PROMPT_MODEL未设置，这将导致提示词生成失败")
        
        self.vector_store = vector_store if vector_store else VectorStore()
        self.design_processor = design_processor if design_processor else DesignImageProcessor(self.vector_store)
        self.temperature = temperature if temperature is not None else settings.DESIGN_PROMPT_CONFIG["temperature"]
        self.max_tokens = max_tokens if max_tokens is not None else settings.DESIGN_PROMPT_CONFIG["max_tokens"]
        
        # 记录使用的最终配置
        logger.info(f"使用的配置 - temperature: {self.temperature}, max_tokens: {self.max_tokens}")
        
        # 初始化LangChain ChatOpenAI客户端 (替代OpenAI客户端)
        try:
            # 获取超时参数
            timeout = 60.0  # 默认值
            try:
                timeout = settings.OPENAI_TIMEOUT
            except AttributeError:
                logger.warning("未找到OPENAI_TIMEOUT配置，使用默认值60秒")
                
            self.llm = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
                model_name=settings.DESIGN_PROMPT_MODEL,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=timeout,
                max_retries=3,
                verbose=True,
                streaming=False,
                default_headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}"
                }
            )
            logger.info(f"LangChain ChatOpenAI客户端初始化成功，使用模型: {settings.DESIGN_PROMPT_MODEL}")
            logger.info(f"基础URL: {settings.OPENAI_BASE_URL}")
        except Exception as e:
            logger.error(f"ChatOpenAI客户端初始化失败: {str(e)}")
            raise
            
        # 确保设置了模型名称
        if not hasattr(settings, 'DESIGN_PROMPT_MODEL') or not settings.DESIGN_PROMPT_MODEL:
            logger.error("DESIGN_PROMPT_MODEL未在配置中设置，将使用配置文件中的默认值")
            # 尝试从配置中获取模型名称
            model_name = settings.DESIGN_PROMPT_CONFIG.get("model", settings.DEFAULT_MODEL_NAME)
            if not model_name:
                logger.error("无法确定要使用的模型名称，请检查配置")
            else:
                logger.info(f"将使用配置文件中指定的模型: {model_name}")
        else:
            logger.info(f"将使用环境变量设置的模型: {settings.DESIGN_PROMPT_MODEL}")
        
        # 初始化工作流图
        self.workflow = self._build_workflow()
        
        # 加载提示词缓存
        self._prompt_cache_file = Path(settings.DATA_DIR) / "prompt_cache.json"
        # 确保数据目录存在
        self._prompt_cache_file.parent.mkdir(parents=True, exist_ok=True)
        # 设置类变量
        DesignPromptAgent._prompt_cache_file = self._prompt_cache_file
        self._load_prompt_cache()
        
        logger.info("设计图Prompt生成Agent初始化完成")
    
    def _load_prompt_cache(self):
        """加载Prompt缓存"""
        try:
            if DesignPromptAgent._prompt_cache_file.exists():
                with open(DesignPromptAgent._prompt_cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    # 过滤掉过期的缓存项
                    current_time = time.time()
                    DesignPromptAgent._prompt_cache = {
                        k: v for k, v in cache_data.items()
                        if current_time - v.get('timestamp', 0) < self._cache_expiry
                    }
                logger.info(f"加载了 {len(DesignPromptAgent._prompt_cache)} 个设计图Prompt缓存")
            else:
                logger.info("没有找到设计图Prompt缓存文件，将创建新缓存")
                DesignPromptAgent._prompt_cache = {}
        except Exception as e:
            logger.error(f"加载设计图Prompt缓存失败: {str(e)}")
            DesignPromptAgent._prompt_cache = {}
    
    def _save_prompt_cache(self):
        """保存Prompt缓存"""
        try:
            # 确保缓存文件已初始化
            if DesignPromptAgent._prompt_cache_file is None:
                logger.warning("缓存文件路径未初始化，无法保存缓存")
                return
                
            # 确保目录存在
            DesignPromptAgent._prompt_cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(DesignPromptAgent._prompt_cache_file, 'w', encoding='utf-8') as f:
                json.dump(DesignPromptAgent._prompt_cache, f, ensure_ascii=False)
            logger.info(f"已保存 {len(DesignPromptAgent._prompt_cache)} 个设计图Prompt缓存")
        except Exception as e:
            logger.error(f"保存设计图Prompt缓存失败: {str(e)}")
            # 记录更详细的错误信息以便调试
            import traceback
            logger.error(f"缓存保存错误详情: {traceback.format_exc()}")
    
    def _get_cache_key(self, image_hash: str, tech_stack: str) -> str:
        """获取缓存键
        
        Args:
            image_hash: 图像哈希
            tech_stack: 技术栈
            
        Returns:
            str: 缓存键
        """
        return f"{image_hash}_{tech_stack}"
    
    def _prune_cache_if_needed(self):
        """如果缓存大小超过限制，清理最旧的缓存项"""
        if len(DesignPromptAgent._prompt_cache) > self._max_cache_size:
            # 按时间戳排序
            sorted_items = sorted(
                DesignPromptAgent._prompt_cache.items(),
                key=lambda x: x[1].get('timestamp', 0)
            )
            # 移除最旧的项，直到缓存大小符合限制
            items_to_remove = len(sorted_items) - self._max_cache_size
            for i in range(items_to_remove):
                del DesignPromptAgent._prompt_cache[sorted_items[i][0]]
            logger.info(f"清理了 {items_to_remove} 个旧的缓存项")
            
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
        """生成设计图提示词"""
        logger.info(f"开始生成设计图提示词，使用模型: {settings.DESIGN_PROMPT_MODEL}")
        logger.info(f"模型配置参数: temperature={state['temperature']}, max_tokens={self.max_tokens}")
        
        if not settings.DESIGN_PROMPT_MODEL:
            logger.error("设计提示词模型未配置! 请检查.env文件中的DESIGN_PROMPT_MODEL设置")
            raise ValueError("设计提示词模型未配置，无法生成提示词")
            
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
6. 适配和响应式设计要求（具体的断点和适配策略）

历史Prompt参考：
{history_prompts}"""
            
            # 使用设计图分析和上下文替换占位符
            system_prompt_filled = system_prompt.format(
                design_analysis=state["design_analysis"],
                tech_stack=state["tech_stack"],
                similar_designs=similar_designs_text,
                history_prompts=history_prompts_text
            )
            
            # 创建LangChain消息列表
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [SystemMessage(content=system_prompt_filled)]
            
            # 使用LangChain ChatOpenAI模型生成Prompt
            try:
                logger.info(f"使用LangChain方式调用模型 {settings.DESIGN_PROMPT_MODEL} 生成Prompt")
                
                # 设置超时参数，确保安全获取
                timeout = 60.0  # 默认值
                try:
                    timeout = settings.OPENAI_TIMEOUT
                    logger.info(f"使用配置的超时设置: {timeout}秒")
                except AttributeError:
                    logger.warning("未找到OPENAI_TIMEOUT配置，使用默认值60秒")
                
                logger.info(f"请求参数: temperature={state['temperature']}, max_tokens={self.max_tokens}, timeout={timeout}")
                
                # 添加重试机制
                max_retries = 3
                retry_delay = 1  # 初始延迟1秒
                last_error = None
                
                for attempt in range(max_retries):
                    try:
                        if attempt > 0:
                            logger.warning(f"第{attempt+1}次尝试调用LangChain生成提示词 (重试 {attempt}/{max_retries})")
                            # 使用指数退避策略
                            await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))
                            
                        # 检查模型是否存在
                        if not settings.DESIGN_PROMPT_MODEL:
                            raise ValueError("DESIGN_PROMPT_MODEL未设置，请检查环境变量配置")
                        
                        # 动态更新温度参数，确保使用请求中的temperature
                        self.llm.temperature = state["temperature"]

                        # 使用LangChain调用模型
                        logger.info("调用LangChain ChatOpenAI开始...")
                        response = await self.llm.ainvoke(messages)
                        logger.info(f"LangChain调用完成，获取到响应类型: {type(response)}")
                        
                        # 提取生成的内容
                        generated_prompt = response.content
                        
                        # 记录令牌使用信息（如果可用）
                        if hasattr(response, 'token_usage'):
                            token_usage = response.token_usage
                            logger.info(f"令牌使用情况: {token_usage}")
                        else:
                            # Claude模型没有token_usage属性，尝试估算
                            if "claude" in settings.DESIGN_PROMPT_MODEL.lower():
                                # 粗略估计Claude模型的token用量
                                # 使用字符数作为简单估算（大约4个字符=1个token）
                                prompt_text = "".join([m.content for m in messages])
                                prompt_tokens = len(prompt_text) // 4
                                completion_tokens = len(generated_prompt) // 4
                                total_tokens = prompt_tokens + completion_tokens
                                
                                token_usage = {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens, 
                                    "total_tokens": total_tokens
                                }
                                logger.info(f"估算的Claude令牌使用情况: {token_usage}")
                            else:
                                # 其他模型设置默认的token使用情况
                                token_usage = {
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0, 
                                    "total_tokens": 0
                                }
                                logger.info("无法从响应中获取令牌使用信息")
                            
                        logger.info(f"成功获取LangChain响应，使用模型: {settings.DESIGN_PROMPT_MODEL}")
                        break
                        
                    except Exception as e:
                        last_error = e
                        logger.warning(f"LangChain调用失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
                        
                        # 如果已经是最后一次尝试，则不再继续
                        if attempt == max_retries - 1:
                            logger.error(f"所有重试都失败，最后错误: {str(last_error)}")
                            raise last_error
                
                if not generated_prompt or len(generated_prompt.strip()) < 100:
                    raise ValueError(f"生成的Prompt过短或为空: {generated_prompt}")
                
                logger.info(f"成功生成Prompt, 估计总Token用量: {token_usage.get('total_tokens', 'unknown')}")
                prompt = generated_prompt
                
            except Exception as e:
                # 记录错误但不中断流程
                logger.error(f"生成Prompt失败: {str(e)}")
                
                # 创建一个更丰富的默认Prompt
                tech_stack = state["tech_stack"]
                # 提取设计分析中的关键信息（如果有）
                design_preview = ""
                if state.get("design_analysis") and len(state["design_analysis"]) > 200:
                    design_preview = state["design_analysis"][:200] + "..."
                
                prompt = f"""# {tech_stack}应用开发提示词

## 设计图详细描述
基于提供的设计分析，这是一个{tech_stack}应用界面，需要根据分析结果实现相应的UI和功能。

{design_preview}

**注意：** 此提示词是因LangChain API调用失败而自动生成的默认内容。
错误信息: {str(e)}

## UI组件层次结构
请参考设计图分析中的UI组件清单，构建合适的布局结构。

## 布局和排版要求
根据设计图分析中的布局结构分析部分实现布局。

## 交互和动画效果
参考设计图分析中的交互设计细节实现相应的交互效果。

## 技术实现建议
使用{tech_stack}标准开发实践和组件库实现此界面。

## 适配和响应式设计要求
确保应用在不同尺寸的设备上都能良好显示。

*此提示词为系统默认生成，请结合设计图分析完善具体实现细节。*
"""
                # 添加警告消息
                messages.append(HumanMessage(content=f"警告: 使用LLM生成Prompt失败: {str(e)}。已使用默认模板生成Prompt。"))
                
                # 设置默认token使用量
                token_usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            
            # 更新状态
            state["generated_prompt"] = prompt
            state["messages"] = state.get("messages", []) + [{"role": "assistant", "content": prompt}]
            state["token_usage"] = token_usage
            state["next_step"] = "complete"
            
            return state
            
        except Exception as e:
            logger.error(f"生成Prompt失败: {str(e)}")
            import traceback
            logger.error(f"生成Prompt详细错误: {traceback.format_exc()}")
            
            # 构建一个更详细的错误响应
            error_message = f"生成过程发生错误: {str(e)}"
            error_prompt = f"""# {state.get('tech_stack', 'Android')}应用开发提示词

## 生成失败

无法基于设计图生成开发提示词，请检查以下可能的原因：
1. API调用超时或连接问题
2. 模型配置错误（如使用了不兼容的模型）
3. 服务器内部错误

详细错误信息: {str(e)}

### 建议解决方案
1. 检查网络连接和API服务状态
2. 确认.env文件中的模型配置是否正确
3. 查看服务器日志获取更多信息
4. 尝试使用不同的设计图或技术栈
"""
            
            # 返回错误结果
            return {
                "error": error_message,
                "generated_prompt": error_prompt,
                "tech_stack": state.get("tech_stack", "Android"),
                "design_image_id": state.get("design_image_id", "未知"),
                "cache_hit": False
            }
    
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
        context_window_size: int = 4000,
        skip_cache: bool = False  # 新增参数，允许跳过缓存
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
            skip_cache: 是否跳过缓存（强制重新生成）
            
        Returns:
            Dict[str, Any]: 生成结果
        """
        # 记录基本信息和使用的关键参数
        logger.info(f"开始生成设计图Prompt, 技术栈: {tech_stack}, 设计图ID: {design_image_id}")
        logger.info(f"生成参数: temperature={temperature}, context_window_size={context_window_size}, skip_cache={skip_cache}")
        logger.info(f"使用模型: {settings.DESIGN_PROMPT_MODEL or settings.DESIGN_PROMPT_CONFIG.get('model', '未设置')}")
        
        # 检查OpenAI API密钥是否设置
        if not settings.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY未设置，无法生成Prompt")
            return {
                "error": "OPENAI_API_KEY未设置，请检查环境变量配置",
                "generated_prompt": f"# {tech_stack}应用开发提示词\n\n## 错误\nOpenAI API密钥未设置，无法生成Prompt。"
            }
            
        # 检查生成模型是否设置
        model_to_use = settings.DESIGN_PROMPT_MODEL or settings.DESIGN_PROMPT_CONFIG.get('model')
        if not model_to_use:
            logger.error("生成模型未设置，无法生成Prompt")
            return {
                "error": "生成模型未设置，请检查环境变量或配置文件",
                "generated_prompt": f"# {tech_stack}应用开发提示词\n\n## 错误\n生成模型未设置，无法生成Prompt。"
            }
            
        try:
            # 计算设计图哈希值用于缓存查找
            image_hash = self._calculate_image_hash(design_image_path)
            if image_hash:
                logger.info(f"设计图哈希值: {image_hash}")
                
                # 检查缓存（除非明确要求跳过）
                if not skip_cache:
                    cache_key = self._get_cache_key(image_hash, tech_stack)
                    if cache_key in self._prompt_cache:
                        cache_entry = self._prompt_cache[cache_key]
                        # 检查缓存是否过期
                        if time.time() - cache_entry["timestamp"] < self._cache_expiry:
                            logger.info(f"从缓存获取Prompt: {cache_key}")
                            
                            # 构建响应
                            return {
                                "generated_prompt": cache_entry["prompt"],
                                "tech_stack": tech_stack,
                                "design_image_id": design_image_id,
                                "similar_designs": cache_entry.get("similar_designs", []),
                                "history_prompts": cache_entry.get("history_prompts", []),
                                "has_history_context": cache_entry.get("has_history_context", False),
                                "design_analysis": cache_entry.get("design_analysis", ""),
                                "rag_info": cache_entry.get("rag_info", {}),
                                "generation_time": cache_entry.get("generation_time", 0),
                                "cache_hit": True
                            }
                        else:
                            logger.info(f"缓存已过期: {cache_key}")
                    else:
                        logger.info(f"缓存未命中: {cache_key}")
                else:
                    logger.info("跳过缓存检查，强制重新生成Prompt")
                
            # 初始化状态
            state = DesignPromptState(
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
                design_analysis="",
                image_hash=image_hash
            )
            
            # 记录工作流开始
            generation_start_time = time.time()
            logger.info(f"启动设计图Prompt生成工作流, 起始步骤: {state['next_step']}")
            
            # 执行设计图提示词生成工作流
            try:
                # 各个步骤的详细日志会在相应的方法内部记录
                state = await self._retrieve_similar_designs(state)
                state = await self._retrieve_history_prompts(state)
                state = await self._analyze_design(state)
                state = await self._generate_prompt(state)
                
            except Exception as e:
                logger.error(f"生成过程发生错误: {str(e)}")
                logger.exception("详细错误信息:")
                
                # 生成一个默认提示词（提示失败）
                fallback_prompt = f"""# {tech_stack}应用开发提示词

## 生成过程发生错误
无法生成完整的提示词，请尝试以下解决方案：
1. 检查API密钥配置
2. 跳过缓存重试
3. 使用不同的设计图
4. 更换技术栈

错误信息: {str(e)}

## 部分分析结果
{state.get('design_analysis', '无可用的设计分析')}
"""
                state["generated_prompt"] = fallback_prompt
                
                # 返回带有错误信息的结果
                return {
                    "error": str(e),
                    "generated_prompt": fallback_prompt,
                    "tech_stack": tech_stack,
                    "design_image_id": design_image_id,
                    "design_analysis": state.get("design_analysis", ""),
                    "similar_designs": state.get("similar_designs", []),
                    "history_prompts": state.get("history_prompts", []),
                    "has_history_context": len(state.get("history_prompts", [])) > 0,
                    "cache_hit": False
                }
            
            # 计算生成时间
            generation_time = time.time() - generation_start_time
            logger.info(f"成功完成设计图Prompt生成，耗时: {generation_time:.2f}秒")
            
            # 只有成功生成的提示词才缓存
            if image_hash and state.get("generated_prompt"):
                cache_key = self._get_cache_key(image_hash, tech_stack)
                self._prompt_cache[cache_key] = {
                    "prompt": state["generated_prompt"],
                    "timestamp": time.time(),
                    "tech_stack": tech_stack,
                    "similar_designs": state.get("similar_designs", []),
                    "history_prompts": state.get("history_prompts", []),
                    "has_history_context": len(state.get("history_prompts", [])) > 0,
                    "design_analysis": state.get("design_analysis", ""),
                    "generation_time": generation_time,
                    "rag_info": {}  # 可以添加更多RAG相关信息
                }
                # 修剪缓存以确保不超过大小限制
                self._prune_cache_if_needed()
                # 持久化缓存
                self._save_prompt_cache()
                logger.info(f"已将生成的Prompt添加到缓存: {cache_key}")
            
            # 构建返回结果
            result = {
                "generated_prompt": state["generated_prompt"],
                "tech_stack": tech_stack,
                "design_image_id": design_image_id,
                "similar_designs": state.get("similar_designs", []),
                "history_prompts": state.get("history_prompts", []),
                "has_history_context": len(state.get("history_prompts", [])) > 0,
                "design_analysis": state.get("design_analysis", ""),
                "rag_info": {},  # 可以添加更多RAG相关信息
                "generation_time": generation_time,
                "cache_hit": False
            }
            
            # 添加Token用量信息（如果有）
            if hasattr(state, "token_usage") and state["token_usage"]:
                result["token_usage"] = state["token_usage"]
                
            return result
            
        except Exception as e:
            logger.error(f"生成Prompt失败: {str(e)}")
            import traceback
            logger.error(f"生成Prompt详细错误: {traceback.format_exc()}")
            
            # 构建一个更详细的错误响应
            error_message = f"生成过程发生错误: {str(e)}"
            error_prompt = f"""# {tech_stack}应用开发提示词

## 生成失败

无法基于设计图生成开发提示词，请检查以下可能的原因：
1. API调用超时或连接问题
2. 模型配置错误（如使用了不兼容的模型）
3. 服务器内部错误

详细错误信息: {str(e)}

### 建议解决方案
1. 检查网络连接和API服务状态
2. 确认.env文件中的模型配置是否正确
3. 查看服务器日志获取更多信息
4. 尝试使用不同的设计图或技术栈
"""
            
            # 返回错误结果
            return {
                "error": error_message,
                "generated_prompt": error_prompt,
                "tech_stack": tech_stack,
                "design_image_id": design_image_id,
                "cache_hit": False
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
            
            # 尝试获取原始设计图的路径和哈希值
            design_image_path = None
            image_hash = None
            
            # 尝试从缓存查找设计图信息
            for cache_key, cache_entry in DesignPromptAgent._prompt_cache.items():
                if not cache_key.startswith('fail_'):  # 跳过失败记录
                    cache_result = cache_entry.get('result', {})
                    if cache_result.get('design_image_id') == design_image_id:
                        # 找到了相关的缓存项
                        image_hash = cache_result.get('image_hash')
                        if 'design_image_path' in cache_result:
                            design_image_path = cache_result.get('design_image_path')
                        break
            
            # 保存到向量数据库
            metadata = {
                "id": prompt_id,
                "tech_stack": tech_stack,
                "design_image_id": design_image_id,
                "type": "design_prompt",
                "user_modified": True,
                "created_at": datetime.now().isoformat()
            }
            
            # 如果有图像哈希，添加到元数据
            if image_hash:
                metadata["image_hash"] = image_hash
            
            # 保存到向量数据库
            await self.vector_store.add_texts(
                texts=[prompt],
                metadatas=[metadata],
                ids=[prompt_id]
            )
            
            logger.info(f"已将用户修改的Prompt保存到向量数据库，ID: {prompt_id}")
            
            # 如果有图像哈希，更新缓存
            if image_hash:
                cache_key = self._get_cache_key(image_hash, tech_stack)
                # 更新或创建缓存条目
                DesignPromptAgent._prompt_cache[cache_key] = {
                    'result': {
                        "generated_prompt": prompt,
                        "tech_stack": tech_stack,
                        "design_image_id": design_image_id,
                        "image_hash": image_hash,
                        "design_image_path": design_image_path
                    },
                    'timestamp': time.time(),
                    'tech_stack': tech_stack,
                    'user_modified': True
                }
                # 清理缓存（如果需要）
                self._prune_cache_if_needed()
                # 保存缓存到磁盘
                self._save_prompt_cache()
                logger.info(f"已将用户修改的Prompt添加到缓存: {cache_key}")
                
                # 如果有失败记录，移除它
                fail_cache_key = f"fail_{cache_key}"
                if fail_cache_key in DesignPromptAgent._prompt_cache:
                    del DesignPromptAgent._prompt_cache[fail_cache_key]
                    logger.info(f"已移除失败记录: {fail_cache_key}")
                    self._save_prompt_cache()
            
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
    
    def _calculate_image_hash(self, image_path: str) -> Optional[str]:
        """计算设计图的哈希值
        
        Args:
            image_path: 设计图路径
            
        Returns:
            Optional[str]: 哈希值，如果无法计算则返回None
        """
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            return hashlib.sha256(image_data).hexdigest()
        except Exception as e:
            logger.error(f"计算设计图哈希失败: {str(e)}")
            return None 