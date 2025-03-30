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
from src.templates.tech_stack_templates import TECH_STACK_SYSTEM_PROMPTS, TECH_STACK_OPTIMIZATION_TEMPLATES
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
    tech_stack_components: Optional[List[Dict[str, Any]]]  # 技术栈特定组件
    evaluation_result: Optional[Dict[str, Any]]  # 评估结果
    project_analysis: Optional[Dict[str, Any]]  # 项目分析结果

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

## 设计图分析结果
{design_analysis}

## 项目分析结果
{project_analysis}

## 技术栈组件信息
{tech_stack_components}

历史相似设计图的提示词可以作为参考，但请确保生成的提示词针对当前设计图的特点。

## 相似设计图提示词参考
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
            # 根据任务类型选择合适的模型和参数
            if task_type == "design_analysis":
                return ChatOpenAI(
                    temperature=temperature or settings.VISION_MODEL_CONFIG["temperature"],
                    model=settings.VISION_MODEL,
                    max_tokens=settings.VISION_MODEL_CONFIG["max_tokens"]
                )
            elif task_type == "prompt_generation":
                return ChatOpenAI(
                    temperature=temperature or settings.DESIGN_PROMPT_CONFIG["temperature"],
                    model=settings.DESIGN_PROMPT_CONFIG["model_name"],
                    max_tokens=settings.DESIGN_PROMPT_CONFIG["max_tokens"]
                )
            elif task_type == "evaluation":
                return ChatOpenAI(
                    temperature=temperature or 0.3,
                    model=settings.OPENAI_MODEL,
                    max_tokens=2000
                )
            else:
                # 默认配置
                return ChatOpenAI(
                    temperature=temperature or 0.7,
                    model=settings.OPENAI_MODEL
                )
        except Exception as e:
            logger.error(f"初始化LLM失败: {str(e)}")
            # 回退到基本配置
            return ChatOpenAI(
                temperature=0.7,
                model="gpt-3.5-turbo"
            )
            
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
        self.langgraph_version = self._get_langgraph_version_info()
        logger.info(f"LangGraph 版本: {self.langgraph_version}")
        
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
    
    async def _analyze_design(self, state: DesignPromptState) -> DesignPromptState:
        """分析设计图
        
        Args:
            state: 当前状态
            
        Returns:
            DesignPromptState: 更新后的状态
        """
        design_image_id = state.get("design_image_id")
        tech_stack = state.get("tech_stack")
        
        try:
            logger.info(f"开始分析设计图: {design_image_id}")
            
            if not self.design_processor:
                logger.error("设计图处理器未初始化")
                state["design_analysis"] = "设计图处理器未初始化，无法分析设计图"
                state["next_step"] = "retrieve_history_prompts"
                return state
                
            # 添加文件位置搜索并确保能找到设计图
            try:
                # 获取设计图路径
                image_path = self.design_processor._get_design_image_path(design_image_id)
                logger.info(f"设计图路径: {image_path}")
                state["design_image_path"] = str(image_path)
                
                # 检查图片文件是否存在
                if not os.path.exists(image_path):
                    logger.error(f"设计图文件不存在: {image_path}")
                    
                    # 尝试在uploads目录中找到任何匹配的文件
                    upload_dir = Path(self.design_processor.upload_dir)
                    alt_files = list(upload_dir.glob(f"{design_image_id}*"))
                    if alt_files:
                        image_path = str(alt_files[0])
                        logger.info(f"找到替代文件: {image_path}")
                        state["design_image_path"] = image_path
                    else:
                        state["design_analysis"] = f"设计图文件不存在: {image_path}。请重新上传设计图"
                        state["next_step"] = "retrieve_history_prompts"
                        return state
            except Exception as e:
                logger.error(f"获取设计图路径失败: {str(e)}")
                state["design_analysis"] = f"获取设计图路径失败: {str(e)}"
                state["next_step"] = "retrieve_history_prompts"
                return state
                
            # 确保能读取图片文件
            try:
                with open(image_path, "rb") as f:
                    image_data = f.read()
                logger.info(f"成功读取设计图文件: {image_path}, 大小: {len(image_data)} 字节")
            except Exception as e:
                logger.error(f"读取设计图文件失败: {str(e)}")
                state["design_analysis"] = f"读取设计图文件失败: {str(e)}"
                state["next_step"] = "retrieve_history_prompts"
                return state
                
            # 尝试验证图片格式
            try:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(image_data))
                logger.info(f"图片格式: {img.format}, 尺寸: {img.size}, 模式: {img.mode}")
                
                # 保存验证后的图片，确保它可以被正确打开
                validated_path = os.path.join(os.path.dirname(image_path), f"validated_{os.path.basename(image_path)}")
                img.save(validated_path)
                logger.info(f"验证后的图片已保存到: {validated_path}")
                
                # 使用验证后的图片路径
                image_path = validated_path
                state["design_image_path"] = str(image_path)
            except Exception as e:
                logger.warning(f"验证图片格式失败: {str(e)}")
                # 继续处理，不中断流程
                
            # 分析设计图
            logger.info(f"调用analyze_design_image分析设计图: {image_path}, 技术栈: {tech_stack}")
            analysis_result = await self.design_processor.analyze_design_image(image_path, tech_stack)
            
            if not analysis_result:
                logger.warning(f"analyze_design_image返回空结果: {design_image_id}")
                state["design_analysis"] = "分析设计图失败，请检查设计图是否有效"
                state["next_step"] = "retrieve_history_prompts"
                return state
                
            # 检查分析结果状态
            status = analysis_result.get("status")
            if status == "error":
                logger.warning(f"设计图分析返回错误: {analysis_result.get('message')}")
                state["design_analysis"] = f"分析设计图出错: {analysis_result.get('message')}"
                state["next_step"] = "retrieve_history_prompts"
                return state
                
            # 获取分析内容
            analysis_content = analysis_result.get("analysis")
            if not analysis_content or len(analysis_content) < 50:
                logger.warning(f"分析设计图返回内容过短 ({len(analysis_content) if analysis_content else 0} 字符)")
                state["design_analysis"] = "分析设计图返回内容不足，请检查设计图是否有效"
                state["next_step"] = "retrieve_history_prompts"
                return state
                
            # 格式化分析结果
            formatted_analysis = self._format_design_analysis(analysis_result)
            state["design_analysis"] = formatted_analysis
            logger.info(f"设计图分析完成: {design_image_id}, 分析结果长度: {len(formatted_analysis)} 字符")
            
            # 设置下一步为检索历史提示词
            state["next_step"] = "retrieve_history_prompts"
            return state
            
        except Exception as e:
            logger.error(f"分析设计图失败: {str(e)}", exc_info=True)
            state["design_analysis"] = f"分析设计图时出错: {str(e)}"
            state["next_step"] = "retrieve_history_prompts"
            return state
    
    def _format_design_analysis(self, analysis: Dict[str, Any]) -> str:
        """格式化设计分析结果
        
        Args:
            analysis: 设计分析结果
            
        Returns:
            str: 格式化后的设计分析结果
        """
        try:
            # 提取关键信息
            elements = analysis.get("elements", [])
            colors = analysis.get("colors", [])
            fonts = analysis.get("fonts", [])
            layout = analysis.get("layout", {})
            summary = analysis.get("summary", "无摘要")
            
            # 格式化元素
            elements_text = ""
            for i, element in enumerate(elements[:10]):  # 限制显示前10个元素
                element_type = element.get("type", "未知类型")
                element_name = element.get("name", f"元素{i+1}")
                element_desc = element.get("description", "无描述")
                elements_text += f"- {element_name} ({element_type}): {element_desc}\n"
            
            # 格式化颜色
            colors_text = ""
            for i, color in enumerate(colors[:5]):  # 限制显示前5个颜色
                hex_code = color.get("hex", "#000000")
                color_name = color.get("name", f"颜色{i+1}")
                colors_text += f"- {color_name}: {hex_code}\n"
            
            # 格式化字体
            fonts_text = ""
            for i, font in enumerate(fonts[:3]):  # 限制显示前3个字体
                font_name = font.get("name", f"字体{i+1}")
                font_style = font.get("style", "常规")
                fonts_text += f"- {font_name} ({font_style})\n"
            
            # 组合结果
            result = f"""## 设计分析结果
### 摘要
{summary}

### 元素 (前10个)
{elements_text if elements_text else "未检测到元素"}

### 颜色 (前5个)
{colors_text if colors_text else "未检测到颜色"}

### 字体 (前3个)
{fonts_text if fonts_text else "未检测到字体"}

### 布局信息
- 结构: {layout.get("structure", "未检测到布局结构")}
- 对齐方式: {layout.get("alignment", "未检测到对齐方式")}
- 间距: {layout.get("spacing", "未检测到间距信息")}
"""
            return result
            
        except Exception as e:
            logger.error(f"格式化设计分析结果失败: {str(e)}")
            return f"设计分析结果格式化失败: {str(e)}"
    
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
    
    async def _generate_prompt(self, state: DesignPromptState) -> DesignPromptState:
        """生成提示词
        
        Args:
            state: 当前状态
            
        Returns:
            DesignPromptState: 更新后的状态
        """
        # 生成缓存键
        design_id = state["design_image_id"]
        tech_stack = state["tech_stack"]
        cache_key = f"{design_id}_{tech_stack}"
        
        # 如果不跳过缓存，尝试从缓存中获取
        if not state.get("skip_cache", False) and cache_key in DesignPromptAgent._prompt_cache:
            cached_data = DesignPromptAgent._prompt_cache[cache_key]
            # 检查缓存是否过期
            if time.time() - cached_data.get("timestamp", 0) < DesignPromptAgent._cache_expiry:
                logger.info(f"使用缓存的提示词: {cache_key}")
                state["generated_prompt"] = cached_data["prompt"]
                state["next_step"] = "end"
                return state
        
        logger.info(f"开始生成提示词: 设计图ID={design_id}, 技术栈={tech_stack}")
        
        try:
            # 准备基础模板
            if tech_stack in TECH_STACK_SYSTEM_PROMPTS:
                system_template = TECH_STACK_SYSTEM_PROMPTS[tech_stack]
                logger.info(f"使用特定技术栈模板: {tech_stack}")
            else:
                system_template = SYSTEM_PROMPT
                logger.info("使用默认系统提示模板")
            
            # 准备上下文信息
            similar_designs_text = ""
            for i, design in enumerate(state.get("similar_designs", [])[:3]):
                prompt = design.get("prompt", "")
                if len(prompt) > 300:
                    prompt = prompt[:300] + "..."
                similar_designs_text += f"示例{i+1}: {prompt}\n\n"
                
            if not similar_designs_text:
                similar_designs_text = "没有相似设计图提示词"
                
            # 技术栈组件信息
            tech_stack_components = state.get("tech_stack_components", [])
            tech_stack_components_text = ""
            if tech_stack_components:
                tech_stack_components_text = "技术栈组件:\n"
                for component in tech_stack_components[:5]:
                    tech_stack_components_text += f"- {component.get('name', '')}: {component.get('description', '')}\n"
            
            # 获取设计分析结果
            design_analysis = state.get("design_analysis", "")
            if not design_analysis:
                design_analysis = "设计分析未完成或失败"
                
            # 获取项目分析结果
            project_analysis = state.get("project_analysis")
            project_analysis_text = ""
            if project_analysis and isinstance(project_analysis, dict):
                if project_analysis.get("status") == "success":
                    project_analysis_text = "## 项目分析信息:\n"
                    # 添加技术栈信息
                    tech_stack_info = project_analysis.get("tech_stack", {})
                    if tech_stack_info:
                        project_analysis_text += "- 技术栈:\n"
                        for tech_type, techs in tech_stack_info.items():
                            project_analysis_text += f"  - {tech_type}: {', '.join(techs)}\n"
                        
                        # 添加组件信息
                        components = project_analysis.get("components", {})
                        if components:
                            project_analysis_text += "- 组件:\n"
                            for comp_type, comps in components.items():
                                project_analysis_text += f"  - {comp_type}: {', '.join(comps[:5])}\n"
                        
                        # 添加摘要
                        summary = project_analysis.get("summary")
                        if summary:
                            project_analysis_text += f"- 摘要: {summary}\n"
            
            # 准备提示词生成输入
            context = {
                "tech_stack": tech_stack,
                "similar_designs": similar_designs_text,
                "design_analysis": design_analysis,
                "tech_stack_components": tech_stack_components_text,
                "project_analysis": project_analysis_text
            }
            
            # 使用LLM生成提示词
            llm = self._initialize_llm("prompt_generation", state.get("temperature"))
            
            # 准备系统提示
            formatted_system_prompt = system_template.format(**context)
            
            messages = [
                SystemMessage(content=formatted_system_prompt),
                HumanMessage(content=f"请为这个{tech_stack}技术栈的设计图生成一个详细的开发提示词。确保提示词考虑了设计图的特点和技术栈的要求。")
            ]
            
            # 替换ainvoke调用，改用当前支持的API方法
            try:
                # 使用invoke方法替代ainvoke
                logger.info("使用invoke方法调用LLM")
                response = await asyncio.to_thread(llm.invoke, messages)
            except AttributeError:
                # 旧版本langchain可能使用__call__方法
                logger.info("尝试使用__call__方法调用LLM")
                response = await asyncio.to_thread(llm, messages)
                
            generated_prompt = response.content
            
            # 检查是否成功生成
            if not generated_prompt or len(generated_prompt) < 50:
                logger.warning(f"生成的提示词过短或为空: {generated_prompt}")
                generated_prompt = f"提示词生成失败。请检查设计图是否有效，或者尝试不同的技术栈。\n\n设计分析结果: {design_analysis}"
            
            # 更新状态
            state["generated_prompt"] = generated_prompt
            state["next_step"] = "end"
            
            # 存入缓存
            if not state.get("skip_cache", False):
                DesignPromptAgent._prompt_cache[cache_key] = {
                    "prompt": generated_prompt,
                    "timestamp": time.time()
                }
                # 保存缓存
                self._save_prompt_cache()
                
            logger.info(f"提示词生成完成: 字符数={len(generated_prompt)}")
            return state
            
        except Exception as e:
            logger.error(f"生成提示词失败: {str(e)}", exc_info=True)
            state["generated_prompt"] = f"生成提示词时出错: {str(e)}"
            state["next_step"] = "end"
            return state
    
    async def _evaluate_prompt(self, state: DesignPromptState) -> DesignPromptState:
        """评估生成的提示词
        
        Args:
            state: Agent状态
            
        Returns:
            DesignPromptState: 更新后的状态
        """
        try:
            # 获取评估配置
            evaluation_dimensions = settings.DESIGN_PROMPT_CONFIG.get("evaluation_dimensions", [])
            if not evaluation_dimensions:
                logger.warning("未配置评估维度，跳过评估")
                state["evaluation_result"] = {
                    "status": "skipped",
                    "message": "未配置评估维度"
                }
                state["next_step"] = "completed"
                return state
            
            # 准备评估系统提示
            tech_stack = state.get("tech_stack", "")
            evaluation_system_prompt = f"""
            你是一位专业的提示词评估专家，负责评估设计图提示词的质量。
            请根据以下维度评估提示词，为每个维度打分（0-10分）并给出改进建议：
            
            {', '.join(evaluation_dimensions)}
            
            请考虑以下因素：
            1. 提示词是否清晰描述了设计图的UI元素和布局
            2. 是否符合{tech_stack}技术栈的特点和最佳实践
            3. 是否包含了必要的技术实现细节
            4. 是否考虑了性能、可维护性和用户体验
            5. 是否充分利用了项目中已有的组件
            
            请以JSON格式返回评估结果，包含以下字段：
            1. scores: 各维度的评分
            2. average_score: 平均评分
            3. feedback: 总体反馈
            4. improvement_suggestions: 改进建议（如果平均分低于阈值）
            """
            
            # 准备评估提示
            evaluation_prompt = f"""
            请评估以下{tech_stack}技术栈的设计图提示词：
            
            设计图分析：
            {state.get('design_analysis', '未提供设计图分析')}
            
            生成的提示词：
            {state.get('generated_prompt', '未生成提示词')}
            
            请在评估中考虑以下技术栈特点：
            {json.dumps(settings.DESIGN_PROMPT_CONFIG.get('tech_stack_config', {}).get(tech_stack, {}), ensure_ascii=False, indent=2)}
            
            请以JSON格式返回评估结果。
            """
            
            # 初始化评估LLM
            evaluation_llm = self._initialize_llm(task_type="evaluation", temperature=0.3)
            
            # 创建评估提示模板
            evaluation_template = ChatPromptTemplate.from_messages([
                ("system", evaluation_system_prompt),
                ("human", evaluation_prompt)
            ])
            
            # 格式化提示
            formatted_evaluation_prompt = evaluation_template.format_messages()
            
            # 调用LLM
            logger.info(f"调用LLM评估提示词，使用模型: {evaluation_llm.model_name}")
            result = await evaluation_llm.apredict_messages(formatted_evaluation_prompt)
            
            # 解析评估结果 - 尝试提取JSON部分
            evaluation_text = result.content
            try:
                # 查找JSON部分
                json_start = evaluation_text.find('{')
                json_end = evaluation_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_text = evaluation_text[json_start:json_end]
                    evaluation_result = json.loads(json_text)
                else:
                    # 无法找到JSON，使用整个文本
                    evaluation_result = {
                        "scores": {},
                        "average_score": 0,
                        "feedback": "无法解析评估结果",
                        "raw_result": evaluation_text
                    }
            except json.JSONDecodeError:
                evaluation_result = {
                    "scores": {},
                    "average_score": 0,
                    "feedback": "无法解析评估结果JSON",
                    "raw_result": evaluation_text
                }
            
            # 获取评估阈值
            evaluation_threshold = settings.DESIGN_PROMPT_CONFIG.get("evaluation_threshold", 7.0)
            
            # 检查是否需要改进
            average_score = evaluation_result.get("average_score", 0)
            if average_score < evaluation_threshold and "improvement_suggestions" not in evaluation_result:
                # 生成改进建议
                improvement_prompt = f"""
                刚刚评估的提示词平均分为{average_score}，低于阈值{evaluation_threshold}。
                请提供具体的改进建议，使提示词达到更高质量。
                """
                
                improvement_messages = [
                    SystemMessage(content=evaluation_system_prompt),
                    HumanMessage(content=evaluation_prompt),
                    AIMessage(content=evaluation_text),
                    HumanMessage(content=improvement_prompt)
                ]
                
                improvement_result = await evaluation_llm.apredict_messages(improvement_messages)
                evaluation_result["improvement_suggestions"] = improvement_result.content
            
            # 更新状态
            state["evaluation_result"] = evaluation_result
            state["next_step"] = "completed"
            return state
        except Exception as e:
            logger.error(f"评估提示词失败: {str(e)}")
            state["evaluation_result"] = {
                "status": "error",
                "message": f"评估失败: {str(e)}"
            }
            state["next_step"] = "completed"
            return state
    
    async def generate_design_prompt(
        self,
        design_image_id: str,
        tech_stack: str,
        agent_type: str = None,
        rag_method: str = None,
        retriever_top_k: int = None,
        temperature: float = None,
        context_window_size: int = None,
        skip_cache: bool = False,
        project_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """生成设计提示词
        
        Args:
            design_image_id: 设计图ID
            tech_stack: 技术栈
            agent_type: 代理类型
            rag_method: RAG方法
            retriever_top_k: 检索结果数量
            temperature: 温度
            context_window_size: 上下文窗口大小
            skip_cache: 是否跳过缓存
            project_analysis: 项目分析结果
            
        Returns:
            Dict[str, Any]: 生成结果
        """
        start_time = time.time()
        
        # 设置默认值
        agent_type = agent_type or "standard"
        rag_method = rag_method or "similarity"
        retriever_top_k = retriever_top_k or 3
        temperature = temperature or self.temperature
        context_window_size = context_window_size or 4000
        
        # 初始化状态
        initial_state: DesignPromptState = {
            "messages": [],
            "tech_stack": tech_stack,
            "design_image_id": design_image_id,
            "design_image_path": "",
            "similar_designs": [],
            "history_prompts": [],
            "rag_method": rag_method,
            "retriever_top_k": retriever_top_k,
            "agent_type": agent_type,
            "temperature": temperature,
            "context_window_size": context_window_size,
            "generated_prompt": "",
            "next_step": "start",
            "design_analysis": "",
            "project_analysis": project_analysis,
            "skip_cache": skip_cache,
            "tech_stack_components": [],
            "evaluation_result": None
        }
        
        try:
            logger.info(f"开始生成设计提示词: 设计图ID={design_image_id}, 技术栈={tech_stack}")
            
            # 检查设计图ID
            if not design_image_id:
                return {"status": "error", "message": "设计图ID不能为空"}
                
            # 检查技术栈
            if not tech_stack:
                return {"status": "error", "message": "技术栈不能为空"}
            
            # 执行工作流
            logger.info("执行设计提示词生成工作流")
            
            # 由于工作流是同步调用而方法是异步的，我们暂时跳过工作流
            # 直接执行流程以生成结果
            try:
                # 简化流程：直接执行必要步骤
                state = initial_state.copy()
                
                # 分析设计图
                design_result = await self._analyze_design(state)
                design_analysis = design_result.get("design_analysis", "")
                
                # 获取相似设计
                similar_result = await self._retrieve_similar_designs(state)
                similar_designs = similar_result.get("similar_designs", [])
                
                # 获取设计提示词
                state["design_analysis"] = design_analysis
                state["similar_designs"] = similar_designs
                prompt_result = await self._generate_prompt(state)
                
                final_state = {
                    "generated_prompt": prompt_result.get("generated_prompt", ""),
                    "design_analysis": design_analysis,
                    "similar_designs": similar_designs,
                    "tech_stack_components": prompt_result.get("tech_stack_components", [])
                }
                
            except Exception as e:
                logger.error(f"执行设计提示词生成流程失败: {str(e)}")
                final_state = {
                    "generated_prompt": f"生成设计提示词失败: {str(e)}",
                    "design_analysis": "",
                    "similar_designs": []
                }
            
            # 检查生成结果
            if not final_state.get("generated_prompt"):
                return {
                    "status": "error",
                    "message": "生成提示词失败",
                    "details": final_state
                }
                
            # 构建返回结果
            processing_time = time.time() - start_time
            result = {
                "status": "success",
                "prompt": final_state["generated_prompt"],
                "tech_stack": tech_stack,
                "design_image_id": design_image_id,
                "processing_time": processing_time,
                "design_analysis": final_state.get("design_analysis", ""),
                "similar_designs_count": len(final_state.get("similar_designs", [])),
                "agent_type": agent_type,
                "temperature": temperature
            }
            
            logger.info(f"设计提示词生成完成: 耗时={processing_time:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"生成设计提示词失败: {str(e)}", exc_info=True)
            
            # 返回错误信息
            return {
                "status": "error",
                "message": f"生成设计提示词失败: {str(e)}",
                "details": {
                    "design_image_id": design_image_id,
                    "tech_stack": tech_stack,
                    "error": str(e)
                }
            } 