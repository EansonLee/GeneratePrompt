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
import re
import traceback
import base64
import io
from pydantic import BaseModel
from enum import Enum
from openai import OpenAI
import imghdr

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
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# 导入配置
from config.config import settings

# 推迟可能导致循环引用的导入
# from src.utils.vector_store import VectorStore
# from src.utils.design_image_processor import DesignImageProcessor
from src.templates.tech_stack_templates import TECH_STACK_SYSTEM_PROMPTS, TECH_STACK_OPTIMIZATION_TEMPLATES

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
    tech_stack_components: Optional[List[Dict[str, Any]]]  # 技术栈特定组件
    evaluation_result: Optional[Dict[str, Any]]  # 评估结果
    project_analysis: Optional[Dict[str, Any]]  # 项目分析结果
    design_image_base64: Optional[str]  # 设计图的base64编码
    design_analysis_raw: Optional[Dict[str, Any]]  # 设计图分析结果的原始数据

# ResponseStatus枚举
class ResponseStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"

# 请求类定义
class DesignPromptRequest(BaseModel):
    """设计图提示词生成请求"""
    tech_stack: str
    tech_stack_components: Optional[List[Dict[str, Any]]] = None
    project_id: Optional[str] = None
    agent_type: Optional[str] = None
    rag_method: Optional[str] = None
    retriever_top_k: Optional[int] = None
    temperature: Optional[float] = None
    context_window_size: Optional[int] = None
    skip_cache: bool = False

# 响应类定义
class DesignPromptResponse(BaseModel):
    """设计图提示词生成响应"""
    status: ResponseStatus = ResponseStatus.PENDING
    message: str = ""
    prompt: str = ""
    design_analysis: str = ""
    project_analysis: str = ""
    similar_designs: str = ""
    tech_stack: Optional[str] = None
    processing_time: Optional[float] = None
    
    class Config:
        """配置"""
        use_enum_values = True  # 确保枚举值序列化为字符串
        
        schema_extra = {
            "example": {
                "status": "success",
                "message": "成功生成设计图Prompt",
                "prompt": "基于上传的UI设计图，创建一个Android应用...",
                "design_analysis": "## 设计概述\n这是一个现代化的移动应用界面...",
                "project_analysis": "## 项目分析\n该项目使用MVVM架构...",
                "similar_designs": "## 相似设计\n1. 电商应用首页...",
                "tech_stack": "Android",
                "processing_time": 3.5
            }
        }

# 定义系统提示模板
SYSTEM_PROMPT = """你是一个专业的设计图Prompt生成专家，擅长将设计图转换为详细的开发提示词。
你的任务是根据用户上传的设计图和选择的技术栈({tech_stack})，生成一个详细的开发提示词。

请仔细查看以下设计图和分析结果，这是开发提示词生成的核心依据：

{design_analysis}

请基于项目分析结果，了解该项目的技术栈、组件和代码规范：

{project_analysis}

请充分利用下面的技术栈组件信息，确保开发提示词与现有技术架构兼容：

{tech_stack_components}

请遵循以下指南生成提示词：
1. 仔细分析设计图的UI元素、布局和交互特点
2. 关注{tech_stack}技术栈的特性和最佳实践
3. 优先使用项目分析中已有的组件和代码规范
4. 提示词应包含以下内容：
   - 设计图的详细描述和组织结构
   - UI组件的层次结构和对应的{tech_stack}标准组件
   - 具体的布局和排版要求（含尺寸、间距和对齐方式）
   - 配色方案和设计风格（使用具体的色值）
   - 交互和动画效果的具体实现方式
   - 技术实现建议，优先使用项目已有组件
   - 适配和响应式设计要求

可参考以下历史相似设计图的提示词，但确保生成的提示词针对当前设计图的特点：

## 相似设计图提示词参考
{similar_designs}

请按照以下结构生成最终提示词：
1. 设计概述：简要描述设计的整体风格和用途
2. 布局结构：详细描述UI布局层次和组织方式
3. UI组件：列出所有主要UI组件并说明对应的{tech_stack}标准组件
4. 样式规范：详细说明颜色、字体、尺寸、间距等设计规范
5. 交互行为：描述各组件的交互行为和动画效果
6. 实现建议：提供具体的{tech_stack}技术实现建议，包括代码组织和架构模式
7. 适配考虑：提供响应式设计和跨平台适配建议

确保你的提示词非常详细、明确，并且紧密贴合设计图的实际内容。"""

class DesignPromptAgent:
    """设计图Prompt生成Agent"""
    
    # 添加缓存相关的类变量
    _prompt_cache = {}  # 内存缓存：{图像哈希}_{技术栈} -> 生成的Prompt
    _prompt_cache_file = None  # 缓存文件路径，在__init__方法中初始化
    _cache_expiry = 30 * 24 * 60 * 60  # 缓存过期时间（30天）
    _max_cache_size = 50  # 最大缓存条目数
    
    @staticmethod
    def _get_langgraph_version_info():
        """获取LangGraph版本信息"""
        try:
            import langgraph
            version_str = getattr(langgraph, "__version__", "0.0.0")
            return version_str
        except ImportError:
            logger.warning("LangGraph未安装，请安装langgraph包")
            return "0.0.0"
        except Exception as e:
            logger.error(f"获取LangGraph版本信息失败: {str(e)}")
            return "0.0.0"
    
    def _initialize_llm(self, task_type: str = "default", temperature: Optional[float] = None) -> BaseChatModel:
        """初始化合适的LLM
        
        Args:
            task_type: 任务类型，可选值: "default", "design_analysis", "prompt_generation", "evaluation"
            temperature: 温度参数，如果为None则使用任务默认值
            
        Returns:
            BaseChatModel: 初始化的LLM
        """
        try:
            # 获取超时设置，默认30秒
            timeout = settings.OPENAI_TIMEOUT
            
            # 根据任务类型选择合适的模型和参数
            if task_type == "design_analysis":
                return ChatOpenAI(
                    temperature=temperature or settings.VISION_MODEL_CONFIG["temperature"],
                    model=settings.VISION_MODEL,
                    max_tokens=settings.VISION_MODEL_CONFIG["max_tokens"],
                    request_timeout=timeout
                )
            elif task_type == "prompt_generation":
                return ChatOpenAI(
                    temperature=temperature or settings.DESIGN_PROMPT_CONFIG["temperature"],
                    model=settings.DESIGN_PROMPT_CONFIG["model_name"],
                    max_tokens=settings.DESIGN_PROMPT_CONFIG["max_tokens"],
                    request_timeout=timeout
                )
            elif task_type == "evaluation":
                return ChatOpenAI(
                    temperature=temperature or 0.3,
                    model=settings.OPENAI_MODEL,
                    max_tokens=2000,
                    request_timeout=timeout
                )
            else:
                # 默认配置
                return ChatOpenAI(
                    temperature=temperature or 0.7,
                    model=settings.OPENAI_MODEL,
                    request_timeout=timeout
                )
        except Exception as e:
            logger.error(f"初始化LLM失败: {str(e)}")
            # 回退到基本配置
            try:
                return ChatOpenAI(
                    temperature=0.7,
                    model="gpt-3.5-turbo",
                    request_timeout=30
                )
            except Exception as e2:
                logger.error(f"回退到基础模型也失败: {str(e2)}")
                # 返回一个自定义的模拟LLM，用于降级处理
                return self._create_fallback_llm()
            
    def __init__(
        self,
        vector_store: Optional[Any] = None,
        design_processor: Optional[Any] = None,
        temperature: float = None,
        max_tokens: int = None,
        context_window_size: int = None,
        skip_cache: bool = False
    ):
        """初始化设计图Prompt生成Agent
        
        Args:
            vector_store: 向量存储
            design_processor: 设计图处理器
            temperature: 温度
            max_tokens: 最大token数
            context_window_size: 上下文窗口大小
            skip_cache: 是否跳过缓存
        """
        try:
            # 设置模型默认参数
            self.temperature = temperature or settings.DESIGN_PROMPT_CONFIG["temperature"]
            self.max_tokens = max_tokens or settings.DESIGN_PROMPT_CONFIG["max_tokens"]
            self.context_window_size = context_window_size or settings.DESIGN_PROMPT_CONFIG.get("default_context_window_size", 4096)
            self.skip_cache = skip_cache
            
            # 初始化服务
            self.vector_store = vector_store
            if not self.vector_store:
                from src.utils.vector_store import VectorStore
                self.vector_store = VectorStore()
                
            # 初始化设计图处理器
            self.design_processor = design_processor
            if not self.design_processor:
                from src.utils.design_image_processor import DesignImageProcessor
                self.design_processor = DesignImageProcessor(vector_store=self.vector_store)
                
            # 初始化状态
            self.state = {}
            self._init_state("")
            
            # 初始化缓存
            self._prompt_cache_file = settings.DESIGN_PROMPT_CONFIG.get(
                "prompt_cache_file", 
                os.path.join(settings.DATA_DIR, "design_prompt_cache.json")
            )
            self._load_prompt_cache()
            
            # 初始化工作流
            self.workflow = None
            
            # 初始化LLM
            self.llm = self._initialize_llm(temperature=self.temperature)
            
            logger.info("设计图Prompt生成Agent初始化成功")
        except Exception as e:
            logger.error(f"设计图Prompt生成Agent初始化失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise e
    
    def _load_prompt_cache(self) -> None:
        """加载提示词缓存"""
        try:
            if self._prompt_cache_file is None:
                # 如果缓存文件路径未设置，则在DATA_DIR中创建
                from config.config import settings
                cache_dir = Path(settings.DATA_DIR)
                cache_dir.mkdir(parents=True, exist_ok=True)
                self._prompt_cache_file = cache_dir / "design_prompt_cache.json"
                logger.info(f"设置提示词缓存文件路径: {self._prompt_cache_file}")
            
            if self._prompt_cache_file.exists():
                with open(self._prompt_cache_file, "r", encoding="utf-8") as f:
                    self._prompt_cache = json.load(f)
                logger.info(f"已加载提示词缓存: {len(self._prompt_cache)}个记录")
            else:
                logger.info(f"未找到提示词缓存文件: {self._prompt_cache_file}")
                self._prompt_cache = {}
        except Exception as e:
            logger.error(f"加载提示词缓存失败: {str(e)}")
            self._prompt_cache = {}
    
    def _build_workflow(self) -> StateGraph:
        """构建工作流
        
        Returns:
            StateGraph: 工作流图
        """
        # 检查LangGraph版本，并适配不同版本的API
        is_v1 = version.parse(self.langgraph_version) >= version.parse("0.1.0")
        logger.info(f"使用LangGraph {'v1' if is_v1 else 'v0'}风格的API构建工作流")
        
        # 创建同步版本的处理函数
        def sync_retrieve_similar_designs(state):
            return asyncio.run(self._retrieve_similar_designs(state))
            
        def sync_analyze_design(state):
            return asyncio.run(self._analyze_design(state))
            
        def sync_retrieve_history_prompts(state):
            return asyncio.run(self._retrieve_history_prompts(state))
            
        def sync_generate_prompt(state):
            return asyncio.run(self._generate_prompt(state))
        
        # 使用兼容的方式创建工作流
        if is_v1:
            # v1风格: 使用with语法
            workflow = StateGraph(DesignPromptState)
            
            # 添加节点 - 使用同步版本
            workflow.add_node("retrieve_similar_designs", sync_retrieve_similar_designs)
            workflow.add_node("analyze_design", sync_analyze_design)
            workflow.add_node("retrieve_history_prompts", sync_retrieve_history_prompts)  
            workflow.add_node("generate_prompt", sync_generate_prompt)
            
            # 设置边
            workflow.set_entry_point("retrieve_similar_designs")
            workflow.add_edge("retrieve_similar_designs", "analyze_design")
            workflow.add_edge("analyze_design", "retrieve_history_prompts")
            workflow.add_edge("retrieve_history_prompts", "generate_prompt")
            workflow.add_edge("generate_prompt", END)
            
            # 编译工作流
            return workflow.compile()
        else:
            # v0风格: 传统方式
            builder = StateGraph(DesignPromptState)
            
            # 添加节点 - 使用同步版本
            builder.add_node("retrieve_similar_designs", sync_retrieve_similar_designs)
            builder.add_node("analyze_design", sync_analyze_design)
            builder.add_node("retrieve_history_prompts", sync_retrieve_history_prompts)
            builder.add_node("generate_prompt", sync_generate_prompt)
            
            # 设置边
            builder.set_entry_point("retrieve_similar_designs")
            builder.add_edge("retrieve_similar_designs", "analyze_design")
            builder.add_edge("analyze_design", "retrieve_history_prompts")
            builder.add_edge("retrieve_history_prompts", "generate_prompt")
            builder.add_edge("generate_prompt", END)
            
            # 编译工作流
            return builder.compile()
    
    async def _retrieve_similar_designs(self, state: DesignPromptState) -> DesignPromptState:
        """检索相似设计图
        
        Args:
            state: Agent状态
            
        Returns:
            DesignPromptState: 更新后的状态
        """
        try:
            # 检索相似设计图的逻辑...
            
            # 设置下一步为设计分析
            state["next_step"] = "analyze_design"
            return state
        except Exception as e:
            # 错误处理...
            return state
    
    async def _analyze_design(self, design_image_id: str) -> Optional[Dict[str, Any]]:
        """分析设计图

        Args:
            design_image_id: 设计图ID或base64数据

        Returns:
            Optional[Dict[str, Any]]: 分析结果
        """
        try:
            logger.info(f"分析设计图: {design_image_id[:30]}...")
            if not design_image_id:
                logger.error("设计图ID为空")
                return {
                    "status": "error",
                    "error": "空ID",
                    "error_message": "设计图ID为空",
                    "summary": "无法分析设计图，因为未提供有效的ID或数据。"
                }
            
            # 如果design_image_id是base64数据
            if design_image_id.startswith('data:image'):
                logger.info("使用base64数据进行分析")
                # 保存base64数据到state中以供后续使用
                if self.state:
                    self.state["design_image_base64"] = design_image_id
                tech_stack = self.state.get("tech_stack", "Android") if self.state else "Android"
                analysis_result = await self.design_processor.analyze_design_image(design_image_id, tech_stack)
                return analysis_result
            
            # 获取设计图路径
            design_image_path = await self._get_design_image_path(design_image_id)
            if not design_image_path or not os.path.exists(design_image_path):
                logger.error(f"设计图路径无效或不存在: {design_image_path}")
                
                # 尝试将design_image_id当作base64数据处理
                if len(design_image_id) > 100 and ',' in design_image_id:
                    logger.info("尝试将ID作为base64数据处理")
                    # 保存base64数据到state中以供后续使用
                    if self.state:
                        self.state["design_image_base64"] = design_image_id
                    tech_stack = self.state.get("tech_stack", "Android") if self.state else "Android"
                    analysis_result = await self.design_processor.analyze_design_image(design_image_id, tech_stack)
                    return analysis_result
                
                # 返回错误响应
                return {
                    "status": "error",
                    "error": "文件不存在",
                    "error_message": f"设计图文件不存在: {design_image_path or design_image_id}",
                    "summary": "无法找到指定的设计图文件，请确认文件ID是否正确。"
                }
            
            # 获取技术栈
            tech_stack = self.state.get("tech_stack", "Android") if self.state else "Android"
            logger.info(f"使用技术栈 {tech_stack} 分析设计图: {design_image_path}")
            
            # 调用设计图处理器分析设计图
            analysis_result = await self.design_processor.analyze_design_image(design_image_path, tech_stack)
            if not analysis_result:
                logger.error(f"分析设计图失败，结果为空: {design_image_path}")
                return {
                    "status": "error",
                    "error": "分析失败",
                    "error_message": "设计图处理器返回空结果",
                    "summary": "设计图分析过程中出现错误，无法获取分析结果。"
                }
            
            logger.info("设计图分析完成")
            
            # 读取图像文件并转换为base64（如果结果中没有image_base64）
            if not analysis_result.get("image_base64") and os.path.exists(design_image_path):
                try:
                    import base64
                    with open(design_image_path, "rb") as image_file:
                        image_data = image_file.read()
                        image_type = os.path.splitext(design_image_path)[1][1:].lower()
                        if not image_type or image_type not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                            image_type = 'png'
                        base64_data = base64.b64encode(image_data).decode('utf-8')
                        analysis_result["image_base64"] = f"data:image/{image_type};base64,{base64_data}"
                        
                        # 保存base64数据到state中以供后续使用
                        if self.state:
                            self.state["design_image_base64"] = analysis_result["image_base64"]
                            logger.info("已将图像base64数据保存到state中")
                except Exception as e:
                    logger.warning(f"读取图像文件并转换为base64时出错: {str(e)}")
            
            return analysis_result
        except Exception as e:
            logger.error(f"分析设计图时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "error": "分析异常",
                "error_message": str(e),
                "summary": "在分析设计图过程中发生异常，无法完成分析。"
            }
    
    def _format_similar_designs(self, design_analysis: Optional[Dict[str, Any]]) -> str:
        """格式化相似设计提示词"""
        if not self.state.get("similar_designs"):
            return "没有找到相似的设计图。"
        
        try:
            formatted_result = ""
            similar_designs = self.state["similar_designs"]
            
            for idx, design in enumerate(similar_designs[:3], 1):  # 限制显示前3个相似设计
                prompt = design.get("prompt", "")
                tech_stack = design.get("tech_stack", "未知技术栈")
                
                if prompt:
                    # 截取提示词（避免过长）
                    if len(prompt) > 500:
                        prompt = prompt[:500] + "...(省略部分内容)"
                    
                    formatted_result += f"### 相似设计 {idx} (技术栈: {tech_stack})\n"
                    formatted_result += f"{prompt}\n\n"
            
            return formatted_result or "没有找到相似的设计图提示词。"
        
        except Exception as e:
            logger.error(f"格式化相似设计失败: {str(e)}")
            logger.error(traceback.format_exc())
            return "相似设计格式化失败。"
    
    async def _retrieve_history_prompts(self, state: DesignPromptState) -> DesignPromptState:
        """检索历史提示词
        
        Args:
            state: 当前状态
            
        Returns:
            DesignPromptState: 更新后的状态
        """
        try:
            # 检索历史提示词的逻辑...
            
            # 设置下一步为提取技术栈组件
            state["next_step"] = "extract_tech_stack_components"
            return state
        except Exception as e:
            # 错误处理...
            return state
                    
    async def _extract_tech_stack_components(self, state: DesignPromptState) -> DesignPromptState:
        """提取技术栈特定组件
        
        Args:
            state: Agent状态
            
        Returns:
            DesignPromptState: 更新后的状态
        """
        try:
            tech_stack = state.get("tech_stack", "")
            if not tech_stack or tech_stack not in settings.DESIGN_PROMPT_CONFIG.get("supported_tech_stacks", []):
                logger.warning(f"不支持的技术栈: {tech_stack}")
                state["tech_stack_components"] = []
                state["next_step"] = "generate_prompt"
                return state
            
            # 从配置中获取技术栈特定配置
            tech_stack_config = settings.DESIGN_PROMPT_CONFIG.get("tech_stack_config", {}).get(tech_stack, {})
            if not tech_stack_config:
                logger.warning(f"未找到技术栈特定配置: {tech_stack}")
                state["tech_stack_components"] = []
                state["next_step"] = "generate_prompt"
                return state
            
            # 从Git仓库分析结果中提取组件
            repo_analysis = state.get("repo_analysis", {})
            
            # 提取组件
            components = []
            
            # 从仓库中提取UI组件
            ui_components = repo_analysis.get("ui_components", [])
            for component in ui_components:
                if component.get("type") in tech_stack_config.get("ui_components", []):
                    components.append({
                        "name": component.get("name", ""),
                        "type": component.get("type", ""),
                        "description": component.get("description", ""),
                        "usage_example": component.get("usage_example", ""),
                        "source": "repo"
                    })
            
            # 根据技术栈特定配置添加默认组件
            for component_type in tech_stack_config.get("ui_components", []):
                # 检查是否已经有此类型的组件
                if not any(c.get("type") == component_type for c in components):
                    components.append({
                        "name": component_type,
                        "type": component_type,
                        "description": f"标准{tech_stack} {component_type}组件",
                        "usage_example": "",
                        "source": "default"
                    })
            
            # 更新状态
            state["tech_stack_components"] = components
            state["next_step"] = "generate_prompt"
            return state
        except Exception as e:
            logger.error(f"提取技术栈组件失败: {str(e)}")
            state["tech_stack_components"] = []
            state["next_step"] = "generate_prompt"
            return state
    
    async def _analyze_project(
        self, 
        project_id: str, 
        tech_stack: str, 
        tech_stack_components: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """分析项目代码
        
        Args:
            project_id: 项目ID
            tech_stack: 技术栈
            tech_stack_components: 技术栈组件
            
        Returns:
            Dict[str, Any]: 项目分析结果
        """
        if not project_id:
            logger.info("未提供项目ID，跳过项目分析")
            return None
            
        logger.info(f"项目分析暂不可用，项目ID: {project_id}")
        return {
            "status": "info",
            "message": "项目分析功能已禁用",
            "tech_stack": tech_stack,
            "components": tech_stack_components or []
        }

    async def _generate_prompt(
        self,
        design_image_id: str,
        tech_stack: str,
        design_analysis: str,
        project_analysis: str,
        similar_designs: str,
        tech_stack_components: Optional[List[Dict[str, Any]]] = None,
        skip_cache: bool = False
    ) -> str:
        """生成设计Prompt
        
        Args:
            design_image_id: 设计图ID
            tech_stack: 技术栈
            design_analysis: 设计图分析结果
            project_analysis: 项目分析结果
            similar_designs: 相似设计信息
            tech_stack_components: 技术栈组件列表
            skip_cache: 是否跳过缓存
            
        Returns:
            str: 生成的Prompt
        """
        from datetime import datetime
        import hashlib
        
        try:
            start_time = time.time()
            logger.info(f"开始生成Prompt: design_image_id={design_image_id}, tech_stack={tech_stack}")
            
            # 获取设计图路径
            design_image_path = await self._get_design_image_path(design_image_id)
            if not design_image_path:
                logger.warning(f"无法找到设计图: {design_image_id}")
            
            # 获取设计图的base64数据（如果分析结果中有）
            design_image_base64 = None
            if self.state and "design_analysis_raw" in self.state and self.state["design_analysis_raw"]:
                design_image_base64 = self.state["design_analysis_raw"].get("image_base64")
                if design_image_base64:
                    logger.info(f"从设计分析结果中获取到设计图base64数据，长度: {len(design_image_base64)//1000}KB")
            
            # 如果state中没有，检查design_analysis_raw字典是否可以直接获取
            if not design_image_base64 and isinstance(design_analysis, dict) and "image_base64" in design_analysis:
                design_image_base64 = design_analysis.get("image_base64")
                if design_image_base64:
                    logger.info(f"从设计分析字典中获取到设计图base64数据，长度: {len(design_image_base64)//1000}KB")
            
            # 如果现在仍然没有base64数据，尝试从文件路径读取
            if not design_image_base64 and design_image_path and os.path.exists(design_image_path):
                try:
                    from src.utils.design_image_processor import DesignImageProcessor
                    processor = DesignImageProcessor()
                    design_image_base64 = processor.get_image_base64(design_image_path)
                    if design_image_base64:
                        logger.info(f"从文件路径读取到设计图base64数据，长度: {len(design_image_base64)//1000}KB")
                        
                        # 保存到状态以便后续使用
                        if self.state:
                            self.state["design_image_base64"] = design_image_base64
                except Exception as e:
                    logger.warning(f"从文件读取设计图base64数据失败: {str(e)}")
            
            # 检查缓存
            if not skip_cache:
                # 计算缓存键
                if design_image_base64:
                    # 使用base64数据的哈希作为标识符
                    try:
                        # 提取base64数据部分
                        if ',' in design_image_base64:
                            _, b64_data = design_image_base64.split(',', 1)
                            # 使用截断的哈希，减少长度
                            image_hash = hashlib.md5(b64_data[:5000].encode('utf-8')).hexdigest()[:16]
                        else:
                            image_hash = hashlib.md5(design_image_base64[:5000].encode('utf-8')).hexdigest()[:16]
                    except Exception as e:
                        logger.warning(f"计算图像哈希失败: {str(e)}")
                        image_hash = design_image_id
                else:
                    image_hash = design_image_id
                
                cache_key = f"{image_hash}_{tech_stack}"
                
                # 检查缓存
                if cache_key in self._prompt_cache:
                    cached_prompt = self._prompt_cache[cache_key]
                    # 检查时间戳以确保不使用过期内容
                    timestamp = cached_prompt.get('timestamp', 0)
                    if time.time() - timestamp < self._cache_expiry:
                        prompt = cached_prompt.get('prompt', '')
                        if prompt:
                            logger.info(f"从缓存获取Prompt，缓存键: {cache_key}")
                            return prompt
            
            # 创建系统提示
            system_content = """你是一位专业的移动/Web应用开发专家，根据设计图和项目分析生成详细的开发提示。请生成一个全面的提示，其中包括：
1. 用户界面(UI)详细说明，包括所有组件、布局、颜色、字体等
2. 各组件的交互行为描述
3. 必要的数据结构和API接口
4. 实现细节和技术栈特定指导

请以清晰、结构化的方式提供这些信息，以便开发人员能够准确实现设计图所示的界面。
回答应当包含丰富的细节和技术实现指导，不要过于简略。
根据分析信息组织回答，特别关注设计元素和项目特性。"""

            # 构建用户提示
            user_content = f"### 技术栈\n{tech_stack}\n\n"
            
            if design_analysis:
                user_content += f"### 设计分析\n{design_analysis}\n\n"
            else:
                logger.warning("设计分析信息为空！这可能导致生成的提示词质量下降")
                user_content += "### 设计分析\n没有提供设计分析信息。\n\n"
            
            if project_analysis:
                user_content += f"### 项目分析\n{project_analysis}\n\n"
            
            if similar_designs:
                user_content += f"### 相似设计\n{similar_designs}\n\n"
            
            if tech_stack_components and isinstance(tech_stack_components, list):
                user_content += "### 技术栈组件\n"
                for component in tech_stack_components:
                    if isinstance(component, dict):
                        name = component.get('name', '')
                        version = component.get('version', '')
                        description = component.get('description', '')
                        if name:
                            user_content += f"- {name}"
                            if version:
                                user_content += f" (版本: {version})"
                            user_content += "\n"
                            if description:
                                user_content += f"  {description}\n"
                user_content += "\n"
            
            # 添加设计图的base64数据（如果有）
            if design_image_base64:
                user_content += f"### 设计图（Base64）\n"
                data_length = len(design_image_base64)
                
                # 添加一些图像信息
                user_content += f"设计图数据长度: {data_length//1000}KB\n"
                
                # 添加图像数据 - 不需要展示全部，确保提示词不会太长
                # OpenAI API对token有限制，所以我们最多包含一部分base64数据
                max_base64_length = 50000  # 截取前50KB
                if data_length > max_base64_length:
                    truncated_data = design_image_base64[:max_base64_length] + "..."
                    user_content += f"已截断的Base64数据: {truncated_data}\n\n"
                    logger.info(f"设计图base64数据已截断，原长度: {data_length//1000}KB")
                else:
                    user_content += f"{design_image_base64}\n\n"
            else:
                logger.warning("设计图base64数据不可用！这可能影响生成的提示词质量")
            
            # 添加命令
            user_content += "请基于以上信息，生成一个详细的开发提示，以帮助开发人员实现设计图所示的界面。"
            
            # 使用LLM生成Prompt
            chat_history = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            
            # 选择模型
            try:
                llm = self._initialize_llm("prompt_generation", temperature=self.temperature)
                logger.info(f"使用模型: {llm.model_name}")
            except Exception as e:
                logger.warning(f"初始化提示词生成模型失败: {str(e)}")
                llm = self._create_fallback_llm()
                logger.info("使用回退模型生成提示词")
            
            try:
                # 调用LLM
                result = await llm.apredict_messages(chat_history)
                response_content = result.content if hasattr(result, 'content') else str(result)
                
                # 后处理提示词
                processed_prompt = self._post_process_prompt(response_content)
                
                # 保存到缓存
                if not skip_cache and self._prompt_cache is not None:
                    self._prompt_cache[cache_key] = {
                        'prompt': processed_prompt,
                        'timestamp': time.time(),
                        'tech_stack': tech_stack
                    }
                    
                    # 清理缓存
                    self._save_prompt_cache()
                    logger.info(f"已保存生成的Prompt到缓存: {cache_key}")
                
                # 记录生成时间
                process_time = time.time() - start_time
                logger.info(f"Prompt生成完成，耗时: {process_time:.2f}秒")
                
                return processed_prompt
            except Exception as e:
                logger.error(f"生成Prompt时出错: {str(e)}")
                logger.error(traceback.format_exc())
                
                # 尝试使用备用方法
                try:
                    logger.info("尝试使用备用方法生成提示词...")
                    fallback_prompt = self._generate_fallback_prompt(system_content, user_content)
                    return fallback_prompt
                except Exception as e2:
                    logger.error(f"备用提示词生成也失败: {str(e2)}")
                    return f"无法生成设计提示词。错误: {str(e)}。请检查日志了解更多详情。"
        except Exception as e:
            logger.error(f"_generate_prompt方法发生未处理异常: {str(e)}")
            logger.error(traceback.format_exc())
            return f"生成提示词时发生错误: {str(e)}。请联系管理员或查看日志了解更多信息。"

    async def generate_design_prompt(
        self,
        design_image_id: str,
        design_prompt_request: DesignPromptRequest,
        skip_cache: bool = False,
    ) -> DesignPromptResponse:
        """生成设计图Prompt
        
        Args:
            design_image_id: 设计图ID
            design_prompt_request: 设计图Prompt生成请求
            skip_cache: 是否跳过缓存（强制重新生成）
            
        Returns:
            DesignPromptResponse: 包含生成的Prompt和相关信息的响应
        """
        start_time = time.time()
        logger.info(f"开始生成设计图Prompt: design_image_id={design_image_id}, tech_stack={design_prompt_request.tech_stack}")
        
        # 参数验证
        if not design_image_id:
            logger.error("设计图ID为空")
            return DesignPromptResponse(
                status=ResponseStatus.FAILED,
                message="设计图ID不能为空",
                processing_time=time.time() - start_time
            )
        
        if not design_prompt_request.tech_stack:
            logger.error("技术栈为空")
            return DesignPromptResponse(
                status=ResponseStatus.FAILED,
                message="技术栈不能为空",
                processing_time=time.time() - start_time
            )
        
        try:
            # 1. 尝试从缓存获取
            cache_key = f"{design_image_id}_{design_prompt_request.tech_stack}"
            if not skip_cache and cache_key in self._prompt_cache:
                cached_item = self._prompt_cache[cache_key]
                if time.time() - cached_item.get("timestamp", 0) < self._cache_expiry:
                    logger.info(f"从缓存中获取Prompt: {cache_key}")
                    cached_prompt = cached_item.get("prompt", "")
                    if cached_prompt:
                        return DesignPromptResponse(
                            status=ResponseStatus.SUCCESS,
                            message="成功从缓存获取设计图Prompt",
                            prompt=cached_prompt,
                            design_analysis=self._format_design_analysis(self.state.get("design_analysis_raw")),
                            project_analysis=self._format_project_analysis(self.state.get("project_analysis")),
                            similar_designs=self._format_similar_designs(self.state.get("similar_designs")),
                            tech_stack=design_prompt_request.tech_stack,
                            processing_time=time.time() - start_time
                        )
            
            # 2. 分析设计图
            try:
                design_analysis_raw = await self._analyze_design(design_image_id)
                if not design_analysis_raw:
                    logger.error(f"分析设计图失败: {design_image_id}")
                    return DesignPromptResponse(
                        status=ResponseStatus.FAILED,
                        message="无法分析设计图，请检查设计图ID是否正确",
                        processing_time=time.time() - start_time
                    )
                self.state["design_analysis_raw"] = design_analysis_raw
                design_analysis = self._format_design_analysis(design_analysis_raw)
            except Exception as e:
                logger.error(f"分析设计图时出错: {str(e)}")
                return DesignPromptResponse(
                    status=ResponseStatus.FAILED,
                    message=f"分析设计图时出错: {str(e)}",
                    processing_time=time.time() - start_time
                )
                
            # 3. 分析项目（如果提供了项目ID）
            project_analysis_raw = None
            project_analysis = ""
            if hasattr(design_prompt_request, 'project_id') and design_prompt_request.project_id:
                try:
                    project_analysis_raw = await self._analyze_project(
                        design_prompt_request.project_id,
                        design_prompt_request.tech_stack,
                        design_prompt_request.tech_stack_components
                    )
                    self.state["project_analysis"] = project_analysis_raw
                    project_analysis = self._format_project_analysis(project_analysis_raw)
                except Exception as e:
                    logger.warning(f"分析项目时出错（非致命）: {str(e)}")
                    # 项目分析失败不应阻止整个流程
            
            # 4. 获取相似设计
            similar_designs = ""
            try:
                self.state = await self._retrieve_similar_designs(self.state)
                similar_designs = self._format_similar_designs(self.state.get("similar_designs"))
            except Exception as e:
                logger.warning(f"获取相似设计时出错（非致命）: {str(e)}")
                # 获取相似设计失败不应阻止整个流程
            
            # 5. 生成Prompt
            try:
                prompt = await self._generate_prompt(
                    design_image_id=design_image_id,
                    tech_stack=design_prompt_request.tech_stack,
                    design_analysis=design_analysis,
                    project_analysis=project_analysis,
                    similar_designs=similar_designs,
                    tech_stack_components=design_prompt_request.tech_stack_components,
                    skip_cache=skip_cache
                )
                
                # 返回结果
                processing_time = time.time() - start_time
                logger.info(f"成功生成设计图Prompt，耗时: {processing_time:.2f}秒")
                
                return DesignPromptResponse(
                    status=ResponseStatus.SUCCESS,
                    message="成功生成设计图Prompt",
                    prompt=prompt,
                    design_analysis=design_analysis,
                    project_analysis=project_analysis,
                    similar_designs=similar_designs,
                    tech_stack=design_prompt_request.tech_stack,
                    processing_time=processing_time
                )
            except Exception as e:
                logger.error(f"生成Prompt时出错: {str(e)}")
                return DesignPromptResponse(
                    status=ResponseStatus.FAILED,
                    message=f"生成Prompt时出错: {str(e)}",
                    processing_time=time.time() - start_time
                )
                
        except Exception as e:
            error_msg = f"生成设计图Prompt过程中发生未预期错误: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            return DesignPromptResponse(
                status=ResponseStatus.FAILED,
                message=error_msg,
                processing_time=time.time() - start_time
            )

    def _init_state(self, design_image_id: str) -> None:
        """初始化代理状态"""
        self.state = {
            "design_image_id": design_image_id,
            "design_image_path": "",
            "design_analysis_raw": None,
            "image_base64": None,
            "similar_designs": [],
            "project_analysis_raw": None
        }

    def _format_project_analysis(self, project_analysis: Optional[Dict[str, Any]]) -> str:
        """格式化项目分析结果"""
        if not project_analysis:
            return "未提供项目分析数据。"
        
        try:
            formatted_result = "## 项目分析结果\n\n"
            
            # 添加项目分析状态
            status = project_analysis.get("status", "")
            if status:
                formatted_result += f"**分析状态**: {status}\n\n"
            
            # 添加技术栈信息
            tech_stack = project_analysis.get("tech_stack", {})
            if tech_stack:
                formatted_result += "### 技术栈信息\n"
                for key, value in tech_stack.items():
                    formatted_result += f"- **{key}**: {value}\n"
                formatted_result += "\n"
            
            # 添加组件信息
            components = project_analysis.get("components", [])
            if components:
                formatted_result += "### 组件信息\n"
                for component in components:
                    component_name = component.get("name", "未命名组件")
                    component_type = component.get("type", "未指定类型")
                    formatted_result += f"- **{component_name}** ({component_type})\n"
                    
                    # 添加组件属性
                    props = component.get("props", {})
                    if props:
                        formatted_result += "  属性:\n"
                        for prop_name, prop_value in props.items():
                            formatted_result += f"  - {prop_name}: {prop_value}\n"
                
                formatted_result += "\n"
            
            # 添加代码摘要
            summaries = project_analysis.get("summaries", [])
            if summaries:
                formatted_result += "### 代码摘要\n"
                for summary in summaries[:5]:  # 限制数量以避免过长
                    summary_file = summary.get("file", "未指定文件")
                    summary_content = summary.get("summary", "无摘要")
                    formatted_result += f"- **{summary_file}**:\n  {summary_content}\n"
                
                formatted_result += "\n"
            
            # 添加代码标准
            standards = project_analysis.get("standards", {})
            if standards:
                formatted_result += "### 代码标准\n"
                for standard_name, standard_desc in standards.items():
                    formatted_result += f"- **{standard_name}**: {standard_desc}\n"
                
                formatted_result += "\n"
            
            return formatted_result
        
        except Exception as e:
            logger.error(f"格式化项目分析结果失败: {str(e)}")
            logger.error(traceback.format_exc())
            return "项目分析结果格式化失败。"

    def _format_design_analysis(self, analysis: Optional[Dict[str, Any]]) -> str:
        """格式化设计图分析结果为字符串

        Args:
            analysis: 设计图分析结果

        Returns:
            str: 格式化后的设计图分析结果
        """
        if not analysis:
            logger.warning("设计分析结果为空")
            return "设计分析信息不可用。"

        try:
            # 检查分析结果状态
            if isinstance(analysis, dict) and analysis.get("status") == "error":
                error_msg = analysis.get("error_message", "未知错误")
                logger.warning(f"设计分析包含错误: {error_msg}")
                
                # 仍然尝试构造一些基本信息
                summary = "设计分析过程中出现错误，提供的信息可能不完整。"
                raw_analysis = analysis.get("raw_analysis", "")
                
                # 即使有错误，也尝试从分析结果中提取一些基本信息
                elements = analysis.get("elements", [])
                colors = analysis.get("colors", [])
                layout = analysis.get("layout", {})
                fonts = analysis.get("fonts", [])
            else:
                # 从分析结果中提取信息
                summary = analysis.get("summary", "无设计概述")
                elements = analysis.get("elements", [])
                colors = analysis.get("colors", [])
                layout = analysis.get("layout", {})
                fonts = analysis.get("fonts", [])
                raw_analysis = analysis.get("raw_analysis", "")
            
            # 检索设计图base64数据
            design_image_base64 = None
            # 首先尝试从分析结果中获取
            if "image_base64" in analysis:
                design_image_base64 = analysis.get("image_base64")
                logger.info("从分析结果中获取到设计图base64数据")
            # 如果分析结果中没有，尝试从状态中获取
            elif self.state and "design_image_base64" in self.state:
                design_image_base64 = self.state["design_image_base64"]
                logger.info("从状态中获取到设计图base64数据")
            
            # 如果没有找到base64数据，记录警告
            if not design_image_base64:
                logger.warning("未找到设计图base64数据")
            
            # 添加到状态中以供其他方法使用
            if design_image_base64 and self.state:
                self.state["design_image_base64"] = design_image_base64
                logger.info("已将设计图base64数据保存到状态中")
                
            # 构建格式化的分析结果
            formatted_analysis = "## 设计概述\n"
            if summary and len(summary) > 10:
                formatted_analysis += f"{summary}\n\n"
            else:
                if raw_analysis:
                    # 从原始分析中提取前100个字符作为概述
                    formatted_analysis += f"{raw_analysis[:100]}...\n\n"
                else:
                    formatted_analysis += "无法提取设计概述信息。\n\n"
            
            # 格式化UI组件
            formatted_analysis += "## UI组件\n"
            if elements and len(elements) > 0:
                for element in elements:
                    if isinstance(element, dict):
                        name = element.get("name", "未命名组件")
                        type_info = element.get("type", "未知类型")
                        description = element.get("description", "")
                        
                        formatted_analysis += f"- **{name}** ({type_info})"
                        if description:
                            formatted_analysis += f": {description}"
                        formatted_analysis += "\n"
            else:
                # 提供通用的UI组件信息作为回退
                formatted_analysis += """根据常见设计模式，该界面可能包含以下组件：
- **顶部应用栏/导航栏**：包含应用标题、导航图标和操作按钮
- **内容区域**：主要信息展示区
- **按钮/交互元素**：用于用户交互的可点击元素
- **输入控件**：可能包含文本输入、选择框等
- **列表/网格**：用于展示多项内容
- **底部导航栏**：提供主要功能区域的切换

由于分析不完整，请根据实际设计图补充具体UI组件信息。
"""
            
            # 格式化颜色方案
            formatted_analysis += "\n## 颜色方案\n"
            if colors and len(colors) > 0:
                for color in colors:
                    if isinstance(color, dict):
                        hex_code = color.get("hex", "#000000")
                        name = color.get("name", "未命名颜色")
                        usage = color.get("usage", "")
                        
                        formatted_analysis += f"- **{name}**: `{hex_code}`"
                        if usage:
                            formatted_analysis += f" - {usage}"
                        formatted_analysis += "\n"
            else:
                # 提供通用的颜色方案信息作为回退
                formatted_analysis += """常见的应用颜色方案包括：
- **主色调**：`#3F51B5`（靛蓝色）- 用于主要界面元素
- **强调色**：`#FF4081`（粉红色）- 用于突出重要操作
- **背景色**：`#FFFFFF`（白色）- 用于内容背景
- **文本色**：`#212121`（深灰色）- 用于主要文本
- **次要文本色**：`#757575`（中灰色）- 用于次要文本

请根据实际设计图确定颜色方案。
"""
            
            # 格式化字体样式
            formatted_analysis += "\n## 字体样式\n"
            if fonts and len(fonts) > 0:
                for font in fonts:
                    if isinstance(font, dict):
                        name = font.get("name", "未命名字体")
                        style = font.get("style", "Regular")
                        size = font.get("size", "未知大小")
                        
                        formatted_analysis += f"- **{name}** {style}, {size}\n"
            else:
                # 提供通用的字体样式信息作为回退
                formatted_analysis += """常见的应用字体样式包括：
- **标题字体**: Roboto Bold, 24sp
- **副标题字体**: Roboto Medium, 18sp
- **正文字体**: Roboto Regular, 16sp
- **小字体**: Roboto Light, 14sp
- **按钮文本**: Roboto Medium, 14sp

请根据实际设计图确定使用的字体和大小。
"""
            
            # 格式化布局结构
            formatted_analysis += "\n## 布局结构\n"
            if layout and isinstance(layout, dict):
                structure = layout.get("structure", "未知布局")
                alignment = layout.get("alignment", "未知对齐方式")
                spacing = layout.get("spacing", "未知间距")
                
                formatted_analysis += f"- **布局结构**: {structure}\n"
                formatted_analysis += f"- **对齐方式**: {alignment}\n"
                formatted_analysis += f"- **间距设置**: {spacing}\n"
            else:
                # 提供通用的布局结构信息作为回退
                formatted_analysis += """常见的应用布局结构包括：
- **布局结构**: 垂直线性布局（LinearLayout），带有嵌套的水平布局
- **对齐方式**: 左对齐内容，居中对齐标题和按钮
- **间距设置**: 16dp 外边距，8dp 元素间距

请根据实际设计图确定布局结构细节。
"""
            
            return formatted_analysis
                
        except Exception as e:
            logger.error(f"格式化设计分析结果时出错: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 返回基本的错误信息和通用提示
            return """## 设计分析（处理出错）

抱歉，处理设计图分析结果时出现错误。以下是一些通用的设计指南：

### UI组件
- 顶部应用栏：包含应用名称和主要操作
- 内容区域：展示主要信息
- 交互元素：按钮、输入框等
- 导航元素：底部导航栏或侧边导航

### 颜色方案
- 主色调：应用的主要品牌颜色
- 次要颜色：用于强调和区分元素
- 背景色：通常为白色或浅色
- 文本颜色：深色文本以保证可读性

### 字体样式
- 标题：较大且醒目
- 正文：清晰易读
- 强调文本：使用粗体或不同颜色

### 布局结构
- 清晰的视觉层次
- 一致的对齐和间距
- 适当的元素分组

请结合实际设计图补充具体细节。"""

    async def _get_design_image_path(self, design_image_id: str) -> str:
        """获取设计图路径
        
        Args:
            design_image_id: 设计图ID
            
        Returns:
            str: 设计图路径
        """
        try:
            # 使用config.config中的设置获取上传目录
            from config.config import settings
            
            # 构建设计图路径 - 使用pathlib.Path确保跨平台兼容性
            upload_dir = Path(settings.UPLOAD_DIR)
            
            # 检查目录是否存在，不存在则创建
            if not upload_dir.exists():
                try:
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"已创建上传目录: {upload_dir}")
                except Exception as e:
                    logger.error(f"创建上传目录失败: {str(e)}")
                    return None
            
            # 判断设计图ID是否为base64编码的图像数据
            # 通常base64编码的图像数据以"data:image/"开头
            if design_image_id and isinstance(design_image_id, str) and design_image_id.startswith('data:image/'):
                try:
                    logger.info("检测到base64编码的图像数据")
                    import base64
                    import tempfile
                    import uuid
                    
                    # 提取MIME类型和base64数据
                    parts = design_image_id.split(';base64,')
                    if len(parts) != 2:
                        logger.error("base64数据格式不正确")
                        return None
                        
                    mime_type = parts[0].replace('data:', '')
                    base64_data = parts[1]
                    
                    # 从MIME类型获取图像格式
                    img_format = mime_type.split('/')[1]
                    
                    # 解码base64数据
                    try:
                        img_data = base64.b64decode(base64_data)
                    except Exception as decode_err:
                        logger.error(f"base64解码失败: {str(decode_err)}")
                        return None
                    
                    # 创建临时文件
                    temp_file = upload_dir / f"design_image_{uuid.uuid4().hex}.{img_format}"
                    
                    # 保存图像到临时文件
                    with open(temp_file, "wb") as f:
                        f.write(img_data)
                    
                    logger.info(f"已将base64图像保存为文件: {temp_file}")
                    
                    # 将base64数据保存到state中，以便后续使用
                    self.state["image_base64"] = design_image_id
                    
                    return str(temp_file)
                except Exception as e:
                    logger.error(f"处理base64图像数据失败: {str(e)}")
                    logger.error(traceback.format_exc())
                    # 继续尝试其他方法
            
            # 尝试常见的图像扩展名
            for ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']:
                test_path = upload_dir / f"{design_image_id}{ext}"
                if test_path.exists():
                    logger.info(f"找到设计图文件: {test_path}")
                    return str(test_path)
            
            # 尝试在上传目录中模糊匹配
            for file_path in upload_dir.glob("*"):
                if file_path.is_file() and any(file_path.name.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']):
                    # 检查文件名是否包含设计ID
                    if design_image_id in file_path.stem:
                        logger.info(f"找到匹配的设计图: {file_path}")
                        return str(file_path)
            
            # 找不到设计图，记录错误
            logger.error(f"无法找到设计图: {design_image_id}")
            return None
            
        except Exception as e:
            logger.error(f"获取设计图路径失败: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def _create_fallback_llm(self):
        """创建一个回退LLM，在API调用失败时使用
        
        Returns:
            一个模拟的LLM对象，可以在API失败时提供基本功能
        """
        # 创建一个模拟LLM对象
        class FallbackLLM:
            async def apredict_messages(self, messages):
                """模拟异步预测，返回一个包含默认内容的响应"""
                system_content = ""
                user_content = ""
                
                # 提取system和user消息，用于生成更有针对性的回退响应
                for msg in messages:
                    if hasattr(msg, "content") and hasattr(msg, "type"):
                        if msg.type == "system":
                            system_content = msg.content
                        elif msg.type == "human":
                            user_content = msg.content
                
                # 创建一个AIMessage对象
                from langchain_core.messages import AIMessage
                
                # 使用回退生成方法
                fallback_content = self._generate_fallback_prompt(system_content, user_content)
                return AIMessage(content=fallback_content)
                
        # 返回实例
        fallback_llm = FallbackLLM()
        fallback_llm._generate_fallback_prompt = self._generate_fallback_prompt
        return fallback_llm
    
    def _generate_fallback_prompt(self, system_content: str, user_content: str) -> str:
        """生成回退提示词（当LLM调用失败时使用）
        
        Args:
            system_content: 系统提示内容
            user_content: 用户提示内容
            
        Returns:
            str: 生成的回退提示词
        """
        logger.warning("使用回退机制生成提示词")
        
        try:
            # 从用户内容中提取技术栈信息
            tech_stack_match = re.search(r'为(.+?)技术栈生成', user_content)
            tech_stack = tech_stack_match.group(1) if tech_stack_match else "未指定"
            
            # 构建通用模板提示词
            fallback_prompt = f"""# {tech_stack}开发提示词

## 界面概述
这是一个移动应用UI界面，包含了多个交互组件。界面采用简洁现代的设计风格，主要使用了轻量级的视觉元素。

## 关键UI组件
- **顶部导航栏**：包含标题文本和可能的返回按钮
- **内容区域**：主要展示区域，可能包含文本、图像或列表内容
- **底部操作区**：可能包含按钮、输入框或其他交互元素
- **状态指示器**：显示当前状态或进度的元素（如加载动画、进度条）

## 布局要求
- 采用垂直流式布局，组件从上到下排列
- 内容区域可能需要支持滚动
- 组件间距保持一致，建议使用8dp的基础间距单位
- 边距统一为16dp

## 颜色方案
- **主色调**：#4285F4（蓝色）
- **次要色**：#34A853（绿色）
- **强调色**：#EA4335（红色）
- **背景色**：#FFFFFF（白色）
- **文本色**：#202124（深灰色，主文本），#5F6368（中灰色，次要文本）

## 字体规范
- 主要文本：Roboto Regular，14sp
- 标题文本：Roboto Medium，16sp
- 强调文本：Roboto Bold，14sp
- 次要文本：Roboto Light，12sp

## 交互细节
- 按钮点击时应有轻微的视觉反馈（颜色变化或轻微缩放）
- 列表项支持滑动操作
- 输入框获得焦点时边框颜色应变为主色调
- 错误状态使用强调色（红色）提示

## 响应式考虑
- 界面元素应适应不同屏幕尺寸，保持相对比例
- 文本应设置最大行数，超出部分显示省略号
- 图片应维持宽高比，避免变形

## 动画效果
- 界面转场使用淡入淡出效果，时长300ms
- 按钮点击反馈使用波纹效果
- 列表加载时可使用骨架屏动画

## 实现建议
- 使用{tech_stack}标准组件库实现界面元素
- 考虑使用约束布局减少嵌套层级
- 文本使用资源文件管理，支持多语言
- 图标建议使用矢量图标（SVG或XML矢量）
"""
            
            logger.info("成功生成回退提示词模板")
            return fallback_prompt
            
        except Exception as e:
            logger.error(f"生成回退提示词时出错: {str(e)}")
            # 最简单的回退方案
            return f"""# {tech_stack}界面开发提示词

## 界面描述
这是一个包含常见UI组件的{tech_stack}界面。

## 实现要点
- 使用标准UI组件
- 遵循平台设计规范
- 实现基本的用户交互
- 保持界面简洁易用"""

    def _save_prompt_cache(self) -> None:
        """保存提示词缓存到文件"""
        try:
            if not self._prompt_cache_file:
                logger.warning("缓存文件路径未设置，无法保存缓存")
                return
                
            # 确保文件路径存在
            cache_dir = Path(self._prompt_cache_file).parent
            cache_dir.mkdir(parents=True, exist_ok=True)
                
            # 保存到文件
            with open(self._prompt_cache_file, "w", encoding="utf-8") as f:
                json.dump(self._prompt_cache, f, ensure_ascii=False, indent=2)
                
            logger.info(f"已将{len(self._prompt_cache)}条提示词缓存保存到文件: {self._prompt_cache_file}")
            
            # 裁剪缓存大小
            if len(self._prompt_cache) > self._max_cache_size:
                # 按时间戳排序
                sorted_items = sorted(
                    [(k, v.get('timestamp', 0) if isinstance(v, dict) else 0) 
                     for k, v in self._prompt_cache.items()],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # 保留最新的项目
                keys_to_keep = [k for k, _ in sorted_items[:self._max_cache_size]]
                self._prompt_cache = {k: self._prompt_cache[k] for k in keys_to_keep}
                logger.info(f"已裁剪缓存大小至{len(self._prompt_cache)}条")
                
                # 重新保存
                with open(self._prompt_cache_file, "w", encoding="utf-8") as f:
                    json.dump(self._prompt_cache, f, ensure_ascii=False, indent=2)
                    
        except Exception as e:
            logger.error(f"保存提示词缓存失败: {str(e)}")
            logger.error(traceback.format_exc())

    def _post_process_prompt(self, prompt: str) -> str:
        """对生成的提示词进行后处理
        
        Args:
            prompt: 原始生成的提示词
            
        Returns:
            str: 处理后的提示词
        """
        try:
            # 移除可能的Markdown代码块标记
            prompt = re.sub(r'```[a-zA-Z]*\n', '', prompt)
            prompt = prompt.replace('```', '')
            
            # 移除多余的空行
            prompt = re.sub(r'\n{3,}', '\n\n', prompt)
            
            # 确保每个主要部分之间有空行
            section_headers = [
                '# ', '## ', '### ', '#### ', 
                '界面概述', '关键UI组件', '布局要求', '颜色方案', 
                '字体规范', '交互细节', '响应式考虑', '动画效果', 
                '实现建议'
            ]
            
            for header in section_headers:
                prompt = re.sub(f'([^\n])\n{re.escape(header)}', f'\\1\n\n{header}', prompt)
            
            # 确保标题前有足够的空间
            for i in range(1, 5):
                hashes = '#' * i
                prompt = re.sub(f'([^\n])\n{hashes} ', f'\\1\n\n{hashes} ', prompt)
                
            # 移除可能的前导指令，如"下面是..."
            prompt = re.sub(r'^(以下是|下面是|这是)[^#\n]+\n+', '', prompt)
            
            # 添加技术栈信息（如果没有）
            tech_stack = self.state.get("tech_stack", "")
            if tech_stack and "技术栈" not in prompt[:200]:
                if prompt.startswith("#"):
                    # 已经有标题，添加技术栈信息
                    title_end = prompt.find("\n")
                    if title_end > 0:
                        title = prompt[:title_end].strip()
                        rest = prompt[title_end:].strip()
                        # 如果标题中没有技术栈，添加技术栈
                        if tech_stack not in title:
                            prompt = f"{title} - {tech_stack}技术栈\n\n{rest}"
                else:
                    # 没有标题，添加一个标题
                    prompt = f"# {tech_stack}技术栈开发提示词\n\n{prompt}"
            
            return prompt.strip()
        except Exception as e:
            logger.error(f"提示词后处理失败: {str(e)}")
            logger.error(traceback.format_exc())
            # 返回原始提示词
            return prompt 