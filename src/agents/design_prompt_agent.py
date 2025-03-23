import logging
from typing import Dict, Any, List, TypedDict, Annotated, Optional, Tuple, Literal
from datetime import datetime
import uuid
import json
import os
from pathlib import Path
import hashlib
import time
import asyncio
import importlib
import inspect

# 创建 logger
logger = logging.getLogger(__name__)

try:
    from packaging import version
except ImportError:
    logger.warning("缺少 packaging 模块，版本比较功能将不可用")
    # 创建一个简单的版本类以避免错误
    class version:
        @staticmethod
        def parse(ver_str):
            return ver_str

from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.utils.vector_store import VectorStore
from src.utils.design_image_processor import DesignImageProcessor
from config.config import settings

# 检测 langgraph 版本
try:
    import langgraph
    LANGGRAPH_VERSION = getattr(langgraph, "__version__", "0.0.0")
    logger.info(f"检测到 langgraph 版本: {LANGGRAPH_VERSION}")
except (ImportError, AttributeError):
    LANGGRAPH_VERSION = "0.0.0"
    logger.warning("无法检测 langgraph 版本，使用默认值 0.0.0")

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
    design_analysis: str  # 设计图分析结果
    skip_cache: bool  # 是否跳过缓存

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

# 请求类定义
class GenerateDesignPromptRequest(TypedDict, total=False):
    """设计提示词生成请求"""
    design_image_id: str
    tech_stack: str
    user_feedback: Optional[str]

class DesignPromptAgent:
    """设计图Prompt生成Agent"""
    
    # 添加缓存相关的类变量
    _prompt_cache = {}  # 内存缓存：{图像哈希}_{技术栈} -> 生成的Prompt
    _prompt_cache_file = None  # 缓存文件路径，在__init__方法中初始化
    _cache_expiry = 30 * 24 * 60 * 60  # 缓存过期时间（30天）
    _max_cache_size = 50  # 最大缓存条目数
    
    @staticmethod
    def _get_langgraph_version_info():
        """获取 langgraph 版本信息"""
        try:
            langgraph_spec = importlib.util.find_spec("langgraph")
            if not langgraph_spec:
                logger.warning("未找到 langgraph 模块")
                return None
            
            langgraph_location = langgraph_spec.origin
            logger.info(f"langgraph 模块位置: {langgraph_location}")
            
            # 尝试导入 langgraph 并获取版本
            import langgraph
            lg_version = getattr(langgraph, "__version__", None)
            
            if lg_version:
                logger.info(f"langgraph 版本: {lg_version}")
            else:
                logger.warning("未能确定 langgraph 版本")
            
            # 检查是否有 astream_events 方法
            from langgraph.graph import StateGraph
            has_astream_events = hasattr(StateGraph, "astream_events")
            logger.info(f"langgraph StateGraph 是否支持 astream_events: {has_astream_events}")
            
            return lg_version
            
        except Exception as e:
            logger.error(f"检查 langgraph 版本时出错: {e}")
            return None
            
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
        # 记录日志
        logger.info("初始化设计图Prompt生成Agent")
        logger.info(f"环境配置检查 - OPENAI_API_KEY: {'已设置' if settings.OPENAI_API_KEY else '未设置'}")
        logger.info(f"环境配置检查 - DESIGN_PROMPT_MODEL: {settings.DESIGN_PROMPT_CONFIG['model_name'] or '未设置'}")
        logger.info(f"环境配置检查 - OPENAI_BASE_URL: {settings.OPENAI_BASE_URL or '未设置'}")
        
        # 检查 langgraph 版本
        langgraph_info = self._get_langgraph_version_info()
        logger.info(f"LangGraph 版本: {langgraph_info}")
        self.langgraph_version = langgraph_info
        
        # 初始化基本属性
        self.vector_store = vector_store
        self.design_processor = design_processor or DesignImageProcessor(vector_store)
        self.temperature = temperature if temperature is not None else settings.DESIGN_PROMPT_CONFIG["temperature"]
        self.max_tokens = max_tokens if max_tokens is not None else settings.DESIGN_PROMPT_CONFIG["max_tokens"]
        
        # 初始化提示词缓存文件路径
        if DesignPromptAgent._prompt_cache_file is None:
            DesignPromptAgent._prompt_cache_file = Path(settings.DATA_DIR) / "design_prompt_cache.json"
            logger.info(f"设置提示词缓存文件路径: {DesignPromptAgent._prompt_cache_file}")
            
        # 加载缓存
        self._load_prompt_cache()
        
        # 初始化工作流
        self.workflow = self._build_workflow()
        
        # 初始化OpenAI客户端用于提示词生成
        try:
            if not settings.OPENAI_API_KEY:
                raise ValueError("OpenAI API密钥未设置")
            
            self.llm = OpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL
            )
            logger.info(f"OpenAI客户端初始化成功，使用模型: {settings.DESIGN_PROMPT_CONFIG['model_name']}")
        except Exception as e:
            logger.error(f"OpenAI客户端初始化失败: {str(e)}")
            self.llm = None
        
        logger.info("设计图Prompt生成Agent初始化完成")
    
    def _load_prompt_cache(self) -> None:
        """加载提示词缓存"""
        try:
            if DesignPromptAgent._prompt_cache_file.exists():
                with open(DesignPromptAgent._prompt_cache_file, "r", encoding="utf-8") as f:
                    DesignPromptAgent._prompt_cache = json.load(f)
                logger.info(f"已加载提示词缓存: {len(DesignPromptAgent._prompt_cache)}条记录")
            else:
                logger.info("提示词缓存文件不存在，将创建新缓存")
                DesignPromptAgent._prompt_cache = {}
        except Exception as e:
            logger.error(f"加载提示词缓存失败: {str(e)}")
            DesignPromptAgent._prompt_cache = {}
    
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
            logger.info(f"设计图路径: {state['design_image_path']}")
            logger.info(f"技术栈: {state['tech_stack']}")
            
            # 验证设计图路径
            if not os.path.exists(state["design_image_path"]):
                logger.error(f"设计图文件不存在: {state['design_image_path']}")
                # 使用默认的分析结果
                state["design_analysis"] = self._generate_default_analysis(state)
                state["next_step"] = "generate_prompt"
                return self._add_error_message(state, f"设计图文件不存在: {state['design_image_path']}")
            else:
                logger.info(f"设计图文件存在: {state['design_image_path']}")
                file_size = os.path.getsize(state["design_image_path"])
                logger.info(f"设计图文件大小: {file_size} 字节")
            
            # 读取图片数据
            try:
                with open(state["design_image_path"], "rb") as f:
                    image_data = f.read()
                logger.info(f"成功读取图片数据，大小: {len(image_data)} 字节")
            except Exception as read_error:
                logger.error(f"读取图片数据失败: {str(read_error)}")
                import traceback
                logger.error(f"详细错误信息: {traceback.format_exc()}")
                # 使用默认的分析结果
                state["design_analysis"] = self._generate_default_analysis(state)
                state["next_step"] = "generate_prompt"
                return self._add_error_message(state, f"读取图片数据失败: {str(read_error)}")
            
            # 分析设计图
            try:
                logger.info("开始调用 design_processor.process_image 方法")
                result = await self.design_processor.process_image(
                    file_content=image_data,
                    file_name=os.path.basename(state["design_image_path"])
                )
                logger.info(f"design_processor.process_image 方法调用完成，结果: {result.get('id', '')}")
            except Exception as process_error:
                logger.error(f"调用 design_processor.process_image 方法失败: {str(process_error)}")
                import traceback
                logger.error(f"详细错误信息: {traceback.format_exc()}")
                # 使用默认的分析结果
                state["design_analysis"] = self._generate_default_analysis(state)
                state["next_step"] = "generate_prompt"
                return self._add_error_message(state, f"调用 design_processor.process_image 方法失败: {str(process_error)}")
                
            # 添加分析结果到状态
            state["design_analysis"] = result["analysis"]
            logger.info(f"分析结果长度: {len(result['analysis'])} 字符")
            
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
            state["design_analysis"] = self._generate_default_analysis(state)
            state["next_step"] = "generate_prompt"
            return self._add_error_message(state, f"分析设计图失败: {str(e)}")
            
    def _generate_default_analysis(self, state: DesignPromptState) -> str:
        """生成默认分析结果
        
        Args:
            state: 当前状态
            
        Returns:
            str: 默认分析结果
        """
        tech_stack = state["tech_stack"]
        design_image_path = state["design_image_path"]
        filename = os.path.basename(design_image_path)
        
        return f"""基于{tech_stack}平台的界面分析：

1. 页面整体布局和结构：
   - 请检查设计图文件是否正确上传
   - 当前无法获取具体的设计图分析结果
   - 文件名: {filename}
   - 文件路径: {design_image_path}

2. UI组件和样式：
   - 建议重新上传设计图
   - 确保图片格式正确（支持jpg、png、webp）
   - 图片大小不超过5MB

3. 颜色主题：
   - 暂时无法分析具体的颜色主题
   - 建议使用{tech_stack}标准设计规范

4. 交互设计：
   - 暂时无法分析具体的交互设计
   - 建议参考{tech_stack}官方交互指南

5. 注意事项：
   - 请确保网络连接正常
   - 检查API密钥配置
   - 如果问题持续，请联系技术支持"""
            
    def _add_error_message(self, state: DesignPromptState, error_message: str) -> DesignPromptState:
        """添加错误消息到状态
        
        Args:
            state: 当前状态
            error_message: 错误消息
            
        Returns:
            DesignPromptState: 更新后的状态
        """
        messages = state.get("messages", [])
        messages.append({
            "role": "system",
            "content": f"分析设计图失败: {error_message}\n使用默认分析模板。"
        })
        messages.append({
            "role": "assistant",
            "content": state["design_analysis"]
        })
        state["messages"] = messages
        return state
    
    async def _generate_prompt(self, state: DesignPromptState) -> DesignPromptState:
        """生成设计图提示词"""
        logger.info(f"开始生成设计图提示词，使用模型: {settings.DESIGN_PROMPT_CONFIG['model_name']}")
        logger.info(f"模型配置参数: temperature={state['temperature']}, max_tokens={self.max_tokens}")
        
        if not settings.DESIGN_PROMPT_CONFIG['model_name']:
            logger.error("设计提示词模型未配置! 请检查.env文件中的DESIGN_PROMPT_MODEL设置")
            raise ValueError("设计提示词模型未配置，无法生成提示词")
            
        if not self.llm:
            logger.error("LLM客户端未初始化，请检查API密钥和基础URL配置")
            raise ValueError("LLM客户端未初始化，无法生成提示词")
            
        # 检查是否有图片分析结果
        design_analysis = state.get('design_analysis')
        if not design_analysis:
            error_message = "设计图分析结果为空，无法生成提示词。请检查设计图分析步骤。"
            logger.error(error_message)
            return self._add_error_message(state, error_message)
            
        # 获取相似设计图和历史Prompt
        similar_designs = state.get('similar_designs', [])
        history_prompts = state.get('history_prompts', [])
        tech_stack = state.get('tech_stack', '').lower()
        
        # 检查缓存是否有该设计图的提示词
        image_hash = None
        
        try:
            # 获取图片哈希，用于缓存查找
            design_image_path = state.get('design_image_path')
            if design_image_path:
                image_hash = self._calculate_image_hash(design_image_path)
                logger.info(f"计算得到图片哈希: {image_hash}")
                
                if image_hash and not state.get('skip_cache', False):
                    # 从缓存中查找提示词
                    cached_prompt = self._get_prompt_from_cache(image_hash, tech_stack)
                    if cached_prompt:
                        logger.info(f"从缓存中获取到提示词，跳过生成")
                        state['generated_prompt'] = cached_prompt
                        state['next_step'] = 'finalize'
                        return state
        except Exception as e:
            logger.warning(f"缓存检查失败，将重新生成提示词: {str(e)}")
            
        try:
            # 准备消息列表
            system_content = (
                "你是一个专业的前端开发提示词生成专家，擅长分析UI设计图并生成精确的"
                f"实现指导。你将根据{tech_stack}技术栈，生成详细的开发提示词。请使用中文生成所有内容。"
            )
            
            user_content = (
                f"请根据以下设计图分析，生成一个详细的{tech_stack}实现提示词。所有内容必须使用中文。提示词应该包含："
                "\n\n1. 布局结构的实现方式，包括容器组件和布局技术"
                "\n2. UI组件的实现细节，包括自定义组件和库组件"
                "\n3. 样式规范，包括颜色、字体、间距等"
                "\n4. 交互逻辑，包括状态管理和事件处理"
                "\n5. 性能优化建议"
                "\n\n重要：请确保生成的提示词完全使用中文，不要使用英文。"
                "\n\n设计图分析如下:\n\n"
                f"{design_analysis}"
            )
            
            # 添加相似设计图和历史Prompt
            if similar_designs:
                formatted_similar_designs = self._format_similar_designs(similar_designs)
                user_content += f"\n\n参考相似设计图分析:\n{formatted_similar_designs}"
                
            if history_prompts:
                formatted_history_prompts = self._format_history_prompts(history_prompts)
                user_content += f"\n\n参考历史Prompt:\n{formatted_history_prompts}"
                
            # 创建消息列表
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            
            state['messages'] = messages
            
            # 设置最大重试次数
            max_retries = 3
            retry_count = 0
            base_delay = 2
            
            while retry_count < max_retries:
                try:
                    # 调用API生成Prompt
                    logger.info(f"调用OpenAI API (尝试 {retry_count + 1}/{max_retries})...")
                    
                    response = self.llm.chat.completions.create(
                        model=settings.DESIGN_PROMPT_CONFIG['model_name'],
                        messages=messages,
                        temperature=state['temperature'],
                        max_tokens=self.max_tokens
                    )
                    
                    logger.info(f"API调用成功，获取到响应")
                    
                    # 提取生成的Prompt
                    if hasattr(response, 'choices') and response.choices:
                        generated_prompt = response.choices[0].message.content
                    else:
                        raise ValueError("无法从响应中获取生成的Prompt")
                    
                    # 估算token用量
                    token_usage = {"total_tokens": 0}
                    if hasattr(response, 'usage') and response.usage:
                        token_usage = {
                            "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                            "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                            "total_tokens": getattr(response.usage, 'total_tokens', 0)
                        }
                    else:
                        # Claude模型没有token_usage属性，尝试估算
                        if "claude" in settings.DESIGN_PROMPT_CONFIG['model_name'].lower():
                            # 粗略估计Claude模型的token用量
                            # 使用字符数作为简单估算（大约4个字符=1个token）
                            prompt_text = "\n".join([msg["content"] for msg in messages])
                            prompt_tokens = len(prompt_text) // 4
                            completion_tokens = len(generated_prompt) // 4
                            total_tokens = prompt_tokens + completion_tokens
                            token_usage = {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": total_tokens
                            }
                            logger.info(f"已估算Claude模型的token用量: {token_usage}")
                        else:
                            logger.info("无法从响应中获取令牌使用信息")
                    
                    logger.info(f"成功生成Prompt, 字符长度: {len(generated_prompt)}, 估计总Token用量: {token_usage.get('total_tokens', 'unknown')}")
                    
                    # 保存生成的Prompt
                    state['generated_prompt'] = generated_prompt
                    
                    # 将生成的Prompt添加到缓存
                    if image_hash and tech_stack:
                        self._add_prompt_to_cache(image_hash, tech_stack, generated_prompt)
                    
                    # 设置下一步
                    state['next_step'] = 'finalize'
                    return state
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"生成Prompt失败，已达到最大重试次数: {str(e)}")
                        error_message = f"生成Prompt失败: {str(e)}"
                        return self._add_error_message(state, error_message)
                    
                    # 指数退避重试
                    delay = base_delay * (2 ** (retry_count - 1))
                    logger.warning(f"生成Prompt失败，{delay}秒后重试 ({retry_count}/{max_retries}): {str(e)}")
                    await asyncio.sleep(delay)
            
            # 如果达到这里，表示所有重试都失败了
            error_message = "生成Prompt失败，已达到最大重试次数"
            logger.error(error_message)
            return self._add_error_message(state, error_message)
        
        except Exception as e:
            error_message = f"生成Prompt时发生错误: {str(e)}"
            logger.error(error_message, exc_info=True)
            return self._add_error_message(state, error_message)
    
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
    
    async def generate_design_prompt(self, request: GenerateDesignPromptRequest) -> Dict[str, Any]:
        start_time = time.time()
        design_image_id = request.design_image_id
        tech_stack = request.tech_stack
        user_feedback = request.user_feedback
        prompts = []
        
        # 获取设计图路径
        design_image_processor = self.design_processor
        # 添加自定义方法获取设计图路径
        design_image_path = self._get_design_image_path(design_image_id)
        
        # 检查设计图是否存在
        if not os.path.exists(design_image_path):
            logger.error(f"设计图不存在: {design_image_path}")
            return {
                "status": "error",
                "error": f"设计图不存在: {design_image_id}",
                "design_image_id": design_image_id,
                "tech_stack": tech_stack,
                "processing_time": time.time() - start_time
            }
            
        element_details_dict = {}
        extracted_text = ''
        
        # Load cache
        cache_file_path = self.get_cache_file_path(design_image_id, tech_stack)
        cache_used = False
        generated_prompt = None
        
        # 检查是否应该跳过缓存
        skip_cache = hasattr(request, 'skip_cache') and request.skip_cache
        if not skip_cache and os.path.exists(cache_file_path):
            try:
                with open(cache_file_path, 'r') as f:
                    cached_data = json.load(f)
                    
                    # 验证缓存的技术栈是否匹配
                    if cached_data.get('tech_stack') == tech_stack:
                        logging.info(f"找到缓存的设计提示词: {design_image_id}, 技术栈: {tech_stack}")
                        
                        # 兼容两种缓存字段名
                        generated_prompt = cached_data.get('prompt') or cached_data.get('generated_prompt')
                        
                        if generated_prompt:
                            cache_used = True
                            logging.info(f"使用缓存的提示词，长度: {len(generated_prompt)}")
                            
                            # 直接返回缓存结果
                            return {
                                "status": "success",
                                "prompt": generated_prompt,
                                "design_image_id": design_image_id,
                                "tech_stack": tech_stack,
                                "cache_used": True,
                                "processing_time": 0
                            }
                        else:
                            logging.warning(f"缓存文件中没有找到有效的prompt数据: {cache_file_path}")
                    else:
                        logging.info(f"缓存的技术栈不匹配: 期望={tech_stack}, 实际={cached_data.get('tech_stack')}")
            except Exception as e:
                logging.error(f"读取缓存时出错: {str(e)}")
        
        # 如果代码执行到这里，说明没有使用缓存
        logging.info(f"为设计图 {design_image_id} 和技术栈 {tech_stack} 生成新的提示词")
        
        # Continue with regular processing if cache is not used
        if not cache_used:
            logging.info(f"Generating design prompt for image {design_image_id} with tech stack {tech_stack}")
            
            # Get workflow
            config = {"design_image_id": design_image_id, "tech_stack": tech_stack, "user_feedback": user_feedback}
            workflow = self.get_workflow()

            # 检查 langgraph 版本并使用适当的方法
            lg_version = getattr(self, 'langgraph_version', '0.0.0')
            logging.info(f"使用 langgraph 版本: {lg_version} 处理事件流")
            
            final_state = None
            generated_prompt = None
            last_event = None
            
            try:
                # 根据版本选择不同的处理方法
                if hasattr(workflow, 'astream_events') and (
                    not lg_version or 
                    version.parse(lg_version) >= version.parse('0.3.0')
                ):
                    logging.info("使用 astream_events 方法处理事件流")
                    try:
                        events_stream = workflow.astream_events(config)
                        async for event in events_stream:
                            event_type = event.get("event", "unknown")
                            logging.debug(f"Event type: {event_type}")
                            
                            if event_type == "on_chain_end":
                                logging.debug(f"Chain output: {event.get('data', {}).get('output', 'No output')}")
                            
                            # 保存最后一个事件，作为备用
                            last_event = event
                            
                            if event_type == "end":
                                final_state = event.get("data", {}).get("state", {})
                                logging.info(f"Got final state from end event: {type(final_state)}")
                    except Exception as e:
                        logging.error(f"Error in astream_events: {e}")
                        if last_event and last_event.get("event") == "end":
                            final_state = last_event.get("data", {}).get("state", {})
                            logging.info(f"Using last event state as final state after error")
                else:
                    logging.info("使用传统方法处理工作流")
                    # 对于旧版本，使用 invoke 或 run 方法
                    try:
                        if hasattr(workflow, 'invoke'):
                            logging.info("使用 invoke 方法")
                            final_state = workflow.invoke(config)
                        else:
                            logging.info("使用 run 方法")
                            final_state = workflow.run(config)
                        logging.info(f"Got final state directly: {type(final_state)}")
                    except Exception as e:
                        logging.error(f"Error in traditional workflow execution: {e}")
                
                # 现在尝试从最终状态中提取生成的提示
                if final_state is not None:
                    logging.info(f"Final state type: {type(final_state)}")
                    
                    # 尝试提取 generated_prompt
                    if isinstance(final_state, dict):
                        if "generated_prompt" in final_state:
                            generated_prompt = final_state["generated_prompt"]
                            logging.info("Found generated_prompt directly in state dictionary")
                        else:
                            # 递归搜索字典
                            def search_dict(d, key):
                                if key in d:
                                    return d[key]
                                for k, v in d.items():
                                    if isinstance(v, dict):
                                        result = search_dict(v, key)
                                        if result is not None:
                                            return result
                                return None
                            
                            generated_prompt = search_dict(final_state, "generated_prompt")
                            if generated_prompt:
                                logging.info("Found generated_prompt through dictionary search")
                    else:
                        # 可能是对象而不是字典
                        try:
                            if hasattr(final_state, "generated_prompt"):
                                generated_prompt = final_state.generated_prompt
                                logging.info("Found generated_prompt as attribute of state object")
                        except Exception as e:
                            logging.error(f"Error accessing generated_prompt attribute: {e}")
                
                # 如果从最终状态中找不到生成的提示，尝试从最后一个事件中获取
                if generated_prompt is None and last_event is not None:
                    logging.info("Attempting to extract generated_prompt from last event")
                    try:
                        # 尝试从最后一个事件中的不同位置提取
                        if "data" in last_event and "state" in last_event["data"]:
                            state = last_event["data"]["state"]
                            if isinstance(state, dict):
                                if "generated_prompt" in state:
                                    generated_prompt = state["generated_prompt"]
                                    logging.info("Found generated_prompt in last event's state")
                                elif "values" in state:
                                    values = state["values"]
                                    if isinstance(values, dict) and "generated_prompt" in values:
                                        generated_prompt = values["generated_prompt"]
                                        logging.info("Found generated_prompt in last event's state.values")
                    except Exception as e:
                        logging.error(f"Error extracting from last event: {e}")
                
                # 如果经过足够长的处理时间仍未获得生成的提示，尝试直接生成
                elapsed_time = time.time() - start_time
                if generated_prompt is None and elapsed_time > 10:  # 10秒后仍未获得结果
                    logging.warning(f"Could not extract generated_prompt after {elapsed_time}s, using backup method")
                    try:
                        # 尝试直接从工作流中的相关组件生成提示
                        # 这取决于工作流的具体实现，可能需要根据实际情况调整
                        pass
                    except Exception as e:
                        logging.error(f"Error in backup generation: {e}")
                
                # 最终检查生成的提示
                if generated_prompt:
                    logging.info(f"Successfully extracted generated prompt: {len(generated_prompt)} chars")
                elif last_event and isinstance(last_event, dict):
                    # 更强大的提取方法
                    logging.info("尝试从最后一个事件中提取内容")
                    try:
                        # 尝试提取内容
                        if 'data' in last_event and 'state' in last_event['data']:
                            state_data = last_event['data']['state']
                            if isinstance(state_data, dict) and 'generated_prompt' in state_data:
                                generated_prompt = state_data['generated_prompt']
                                logging.info(f"从last_event.data.state中提取到生成的提示，长度: {len(generated_prompt)}")
                        # 尝试从任何键中提取
                        if not generated_prompt:
                            def deep_search(obj, key):
                                if isinstance(obj, dict):
                                    if key in obj:
                                        return obj[key]
                                    for k, v in obj.items():
                                        result = deep_search(v, key)
                                        if result:
                                            return result
                                elif isinstance(obj, list):
                                    for item in obj:
                                        result = deep_search(item, key)
                                        if result:
                                            return result
                                return None
                            
                            found_prompt = deep_search(last_event, 'generated_prompt')
                            if found_prompt:
                                generated_prompt = found_prompt
                                logging.info(f"通过深度搜索从事件中找到生成的提示，长度: {len(generated_prompt)}")
                            
                            # 尝试从content中提取
                            content = deep_search(last_event, 'content')
                            if content and not generated_prompt:
                                generated_prompt = content
                                logging.info(f"从content字段找到内容，长度: {len(generated_prompt)}")
                    except Exception as e:
                        logging.error(f"从last_event提取内容时出错: {e}")
                
                if not generated_prompt:
                    # 如果所有方法都失败，生成一个默认提示
                    logging.error("无法从工作流结果中提取生成的提示，使用默认提示")
                    generated_prompt = (
                        f"这是一个为{tech_stack}技术栈生成的默认设计提示。\n\n"
                        f"由于无法处理设计图{design_image_id}，无法提供详细的实现描述。\n\n"
                        f"建议：\n"
                        f"1. 请确保设计图文件存在且可访问\n"
                        f"2. 检查文件格式是否支持（支持的格式：jpg, png, jpeg, webp, bmp）\n"
                        f"3. 文件大小不应超过5MB\n"
                        f"4. 尝试重新上传设计图\n\n"
                        f"技术栈：{tech_stack}\n"
                        f"时间戳：{datetime.now().isoformat()}"
                    )
                
            except Exception as e:
                logging.error(f"Error in workflow execution: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "design_image_id": design_image_id, 
                    "tech_stack": tech_stack,
                    "processing_time": time.time() - start_time
                }
            
            # 如果成功获取到生成的提示，更新缓存和向量存储
            if generated_prompt:
                # 添加到向量存储
                try:
                    await self.vector_store.add_texts(
                        texts=[generated_prompt],
                        metadatas=[{
                            "design_image_id": design_image_id,
                            "tech_stack": tech_stack,
                            "timestamp": datetime.now().isoformat()
                        }]
                    )
                    logging.info(f"Added design prompt to vector store for image {design_image_id}")
                except Exception as e:
                    logging.error(f"Error adding to vector store: {e}")
                
                # 更新缓存
                try:
                    os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
                    with open(cache_file_path, 'w') as f:
                        json.dump({
                            "prompt": generated_prompt,
                            "tech_stack": tech_stack,
                            "timestamp": datetime.now().isoformat()
                        }, f)
                    logging.info(f"Cached design prompt for image {design_image_id} and tech stack {tech_stack}")
                except Exception as e:
                    logging.error(f"Error updating cache: {e}")
            
            # 返回结果
            return {
                "status": "success",
                "prompt": generated_prompt,
                "design_image_id": design_image_id,
                "tech_stack": tech_stack,
                "cache_used": cache_used,
                "processing_time": time.time() - start_time
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
    
    def _get_cache_key(self, image_hash: str, tech_stack: str) -> str:
        """生成缓存键
        
        Args:
            image_hash: 图像哈希
            tech_stack: 技术栈
            
        Returns:
            str: 缓存键
        """
        return f"{image_hash}_{tech_stack}"
    
    def _save_prompt_cache(self) -> None:
        """保存提示词缓存"""
        try:
            # 确保缓存文件的父目录存在
            if DesignPromptAgent._prompt_cache_file.parent:
                DesignPromptAgent._prompt_cache_file.parent.mkdir(parents=True, exist_ok=True)
                
            # 保存缓存
            with open(DesignPromptAgent._prompt_cache_file, "w", encoding="utf-8") as f:
                json.dump(DesignPromptAgent._prompt_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"已保存提示词缓存到: {DesignPromptAgent._prompt_cache_file}")
        except Exception as e:
            logger.error(f"保存提示词缓存失败: {str(e)}")
            
    def _prune_cache_if_needed(self) -> None:
        """如果缓存大小超过限制，清理最旧的缓存项"""
        if len(DesignPromptAgent._prompt_cache) <= self._max_cache_size:
            return
            
        # 按时间戳排序，保留最新的条目
        sorted_entries = sorted(
            DesignPromptAgent._prompt_cache.items(),
            key=lambda x: x[1].get("timestamp", 0),
            reverse=True
        )
        
        # 保留最新的条目
        DesignPromptAgent._prompt_cache = {
            k: v for k, v in sorted_entries[:self._max_cache_size]
        }
        
        logger.info(f"缓存清理完成，保留了 {len(DesignPromptAgent._prompt_cache)} 个条目")
        
    def _get_prompt_from_cache(self, image_hash: str, tech_stack: str) -> Optional[str]:
        """从缓存中获取设计图提示词
        
        Args:
            image_hash: 图像哈希
            tech_stack: 技术栈
            
        Returns:
            Optional[str]: 缓存的设计图提示词，如果没有则返回None
        """
        cache_key = self._get_cache_key(image_hash, tech_stack)
        
        if cache_key in DesignPromptAgent._prompt_cache:
            cache_entry = DesignPromptAgent._prompt_cache[cache_key]
            # 检查是否过期
            if time.time() - cache_entry.get("timestamp", 0) < self._cache_expiry:
                # 更新时间戳
                cache_entry["timestamp"] = time.time()
                DesignPromptAgent._prompt_cache[cache_key] = cache_entry
                
                logger.info(f"从缓存中获取设计图提示词: {cache_key}")
                return cache_entry.get("prompt")
                
        return None
        
    def _add_prompt_to_cache(self, image_hash: str, tech_stack: str, prompt: str) -> None:
        """将设计图提示词添加到缓存
        
        Args:
            image_hash: 图像哈希
            tech_stack: 技术栈
            prompt: 生成的提示词
        """
        if not image_hash or not tech_stack or not prompt:
            logger.warning("缺少缓存所需的参数，跳过缓存")
            return
            
        cache_key = self._get_cache_key(image_hash, tech_stack)
        
        DesignPromptAgent._prompt_cache[cache_key] = {
            "prompt": prompt,
            "tech_stack": tech_stack,
            "timestamp": time.time()
        }
        
        # 检查是否需要清理缓存
        self._prune_cache_if_needed()
        
        # 保存缓存
        self._save_prompt_cache()
        
        logger.info(f"已将设计图提示词添加到缓存: {cache_key}")
        
    def get_cache_file_path(self, design_image_id: str, tech_stack: str) -> str:
        """获取缓存文件路径
        
        Args:
            design_image_id: 设计图ID
            tech_stack: 技术栈
            
        Returns:
            str: 缓存文件路径
        """
        # 创建缓存目录
        cache_dir = Path("./cache/design_prompts")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建缓存文件名
        cache_filename = f"{design_image_id}_{tech_stack.replace(' ', '_')}.json"
        
        # 返回完整路径
        return str(cache_dir / cache_filename)
        
    def _check_environment(self):
        # 实现 _check_environment 方法
        pass 

    def get_workflow(self):
        """获取工作流实例
        
        Returns:
            工作流实例
        """
        try:
            # 导入 langgraph 组件
            from langgraph.graph import StateGraph
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI
            
            # 定义状态类型
            class State(TypedDict):
                design_image_id: str
                tech_stack: str
                user_feedback: Optional[str]
                generated_prompt: Optional[str]
            
            # 创建模型
            model_name = os.getenv("DESIGN_PROMPT_MODEL", "gpt-4o")
            model = ChatOpenAI(model_name=model_name, temperature=0.7)
            
            # 创建提示模板
            template = """你是一个精通前端开发的 AI 助手。
            
            设计图 ID: {design_image_id}
            目标技术栈: {tech_stack}
            用户反馈（如有）: {user_feedback}
            
            请为这个设计图生成一个详细的描述，以便另一个 AI 系统可以根据这个描述生成前端代码。
            你的描述应该包含：
            1. 整体布局和设计的详细描述
            2. 颜色方案和主题
            3. 组件和元素的详细描述
            4. UI/UX 元素和交互
            5. 排版和文本内容
            6. 适合指定技术栈的具体实现建议
            
            请生成一个全面、详细且结构化的描述，以便能够准确实现这个设计。
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            
            # 定义生成提示词的函数
            def generate_prompt(state):
                """生成设计提示词"""
                if not state.get("design_image_id") or not state.get("tech_stack"):
                    return {"generated_prompt": "缺少必要的设计图 ID 或技术栈信息"}
                
                # 获取设计图路径
                design_image_id = state["design_image_id"]
                image_path = self._get_design_image_path(design_image_id)
                
                # 检查图像是否存在
                if not os.path.exists(image_path):
                    logger.error(f"设计图文件不存在: {image_path}")
                    return {"generated_prompt": f"设计图文件不存在: {design_image_id}。请确保图片已上传到正确位置。"}
                
                # 图像转base64处理
                try:
                    import base64
                    import mimetypes
                    
                    # 获取MIME类型
                    mime_type, _ = mimetypes.guess_type(image_path)
                    if not mime_type:
                        mime_type = "image/png"  # 默认类型
                        
                    # 读取图像并编码
                    with open(image_path, "rb") as img_file:
                        image_data = img_file.read()
                        image_base64 = base64.b64encode(image_data).decode('utf-8')
                    
                    logger.info(f"成功读取并编码设计图: {image_path}")
                    
                    # 创建带有图像的消息
                    messages = [
                        {
                            "role": "system", 
                            "content": f"你是一个专业的设计分析专家。请分析以下{state['tech_stack']}设计图并生成详细的实现描述。"
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"请为这个{state['tech_stack']}设计图生成一个详细的描述，以便另一个AI系统可以根据这个描述生成前端代码。\n\n描述应包含：\n1. 整体布局和设计的详细描述\n2. 颜色方案和主题\n3. 组件和元素的详细描述\n4. UI/UX元素和交互\n5. 排版和文本内容\n6. 适合{state['tech_stack']}技术栈的具体实现建议"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ]
                    
                    # 使用带图像的消息直接调用模型
                    result = model.invoke(messages)
                    
                except Exception as e:
                    logger.error(f"处理设计图时出错: {str(e)}")
                    # 如果无法处理图像，回退到不带图像的方法
                    inputs = {
                        "design_image_id": state["design_image_id"],
                        "tech_stack": state["tech_stack"],
                        "user_feedback": state.get("user_feedback", "无")
                    }
                    
                    chain = prompt | model
                    result = chain.invoke(inputs)
                
                # 提取结果
                generated_prompt = result.content if hasattr(result, "content") else str(result)
                
                # 返回更新的状态
                return {"generated_prompt": generated_prompt}
            
            # 创建工作流图
            workflow = StateGraph(State)
            
            # 添加节点
            workflow.add_node("generate_prompt", generate_prompt)
            
            # 设置入口点
            workflow.set_entry_point("generate_prompt")
            
            # 编译工作流
            return workflow.compile()
            
        except Exception as e:
            logger.error(f"创建工作流时出错: {e}")
            raise 

    def _get_design_image_path(self, design_image_id: str) -> str:
        """根据设计图ID获取设计图路径
        
        Args:
            design_image_id: 设计图ID
            
        Returns:
            str: 设计图路径
        """
        # 尝试构造可能的路径
        # 通常设计图会保存在上传目录中
        upload_dir = self.design_processor.upload_dir if hasattr(self.design_processor, 'upload_dir') else str(settings.UPLOAD_DIR)
        
        # 检查上传目录中是否有此ID的图片
        for ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
            potential_path = os.path.join(upload_dir, f"{design_image_id}{ext}")
            if os.path.exists(potential_path):
                logger.info(f"找到设计图路径: {potential_path}")
                return potential_path
        
        # 如果找不到，直接返回ID作为路径（可能是完整路径）
        if os.path.exists(design_image_id):
            logger.info(f"设计图ID本身是一个存在的路径: {design_image_id}")
            return design_image_id
            
        logger.warning(f"无法找到设计图: {design_image_id}")
        return design_image_id  # 返回原始ID，调用方需要处理路径不存在的情况 