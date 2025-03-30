import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Button, Card, Input, Upload, message, Select, Slider, Tabs, 
  Space, Alert, Spin, Typography, Radio, Image, Modal, Form, Divider, Collapse, Checkbox, Tooltip, Empty,
  Switch, Row, Col, Badge, Progress, InputNumber, List, Tag, Descriptions, Statistic
} from 'antd';
import { 
  UploadOutlined, LoadingOutlined, SaveOutlined, EditOutlined, QuestionCircleOutlined,
  CopyOutlined, CloseOutlined, GithubOutlined, BranchesOutlined, DatabaseOutlined,
  SettingOutlined, InfoCircleOutlined, CheckCircleOutlined, CloseCircleOutlined, SyncOutlined,
  ReloadOutlined, LinkOutlined, PlusOutlined, CloudUploadOutlined
} from '@ant-design/icons';
import type { UploadFile, UploadProps } from 'antd/es/upload/interface';
import { API_BASE_URL } from '../config';
import ReactMarkdown from 'react-markdown';
import copy from 'copy-to-clipboard';
import { Light as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomOneDark } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import './DesignPromptGenerator.css';

const { TextArea } = Input;
const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { Panel } = Collapse;
const { TabPane } = Tabs;

// 支持的图片类型
const SUPPORTED_IMAGE_FORMATS = [
  '.jpg', '.jpeg', '.png', '.webp'
];

// 技术栈选项
const TECH_STACKS = ['Android', 'iOS', 'Flutter'];

// RAG方法选项
const RAG_METHODS = ['similarity', 'mmr', 'hybrid'];

// Agent类型选项
const AGENT_TYPES = ['ReActAgent', 'ConversationalRetrievalAgent'];

interface ProcessingStatus {
  status: 'idle' | 'processing' | 'completed' | 'error';
  message?: string;
  canRetry?: boolean;
}

// 自定义全屏加载图标
const antIcon = <LoadingOutlined style={{ fontSize: 40 }} spin />;

// 全屏加载组件样式
const fullScreenLoadingStyle: React.CSSProperties = {
  position: 'fixed',
  top: 0,
  left: 0,
  width: '100vw',
  height: '100vh',
  display: 'flex',
  flexDirection: 'column',
  justifyContent: 'center',
  alignItems: 'center',
  backgroundColor: 'rgba(0, 0, 0, 0.7)',
  zIndex: 9999,
  color: 'white',
};

// 添加Git仓库状态接口
interface GitRepoStatus {
  status: 'idle' | 'processing' | 'completed' | 'error';
  message?: string;
  summary?: string;
  processingTime?: number;
  error?: string;
  canRetry?: boolean;
}

const DesignPromptGenerator: React.FC = () => {
  // 基础状态
  const [techStack, setTechStack] = useState<string>(TECH_STACKS[0]);
  const [designImage, setDesignImage] = useState<UploadFile | null>(null);
  const [generatedPrompt, setGeneratedPrompt] = useState<string>('');
  const [editedPrompt, setEditedPrompt] = useState<string>('');
  const [isEditing, setIsEditing] = useState<boolean>(false);
  
  // RAG和Agent参数
  const [ragMethod, setRagMethod] = useState<string>(RAG_METHODS[0]);
  const [retrieverTopK, setRetrieverTopK] = useState<number>(3);
  const [agentType, setAgentType] = useState<string>(AGENT_TYPES[0]);
  const [temperature, setTemperature] = useState<number>(0.5);
  const [contextWindowSize, setContextWindowSize] = useState<number>(4000);
  
  // 处理状态
  const [uploadStatus, setUploadStatus] = useState<ProcessingStatus>({ status: 'idle' });
  const [generationStatus, setGenerationStatus] = useState<ProcessingStatus>({ status: 'idle' });
  const [saveStatus, setSaveStatus] = useState<ProcessingStatus>({ status: 'idle' });
  
  // 上传结果
  const [uploadResult, setUploadResult] = useState<any>(null);
  
  // 全屏加载状态
  const [fullScreenLoading, setFullScreenLoading] = useState<boolean>(false);
  const [loadingMessage, setLoadingMessage] = useState<string>('');
  
  // 确认对话框状态
  const [showSaveConfirm, setShowSaveConfirm] = useState<boolean>(false);
  
  // 历史Prompt状态
  const [hasHistoryContext, setHasHistoryContext] = useState<boolean>(false);
  const [historyPrompts, setHistoryPrompts] = useState<any[]>([]);
  
  // 添加设计分析状态
  const [designAnalysis, setDesignAnalysis] = useState<any>({});
  
  // 上传配置
  const uploadProps: UploadProps = {
    name: 'file',
    multiple: false,
    maxCount: 1,
    accept: SUPPORTED_IMAGE_FORMATS.join(','),
    fileList: designImage ? [designImage] : [],
    beforeUpload: (file) => {
      // 检查文件类型
      const isValidFormat = SUPPORTED_IMAGE_FORMATS.some(format => 
        file.name.toLowerCase().endsWith(format)
      );
      if (!isValidFormat) {
        message.error(`不支持的文件格式，请上传 ${SUPPORTED_IMAGE_FORMATS.join(', ')} 格式的图片`);
        return Upload.LIST_IGNORE;
      }
      
      // 检查文件大小 (5MB)
      const isLessThan5M = file.size / 1024 / 1024 < 5;
      if (!isLessThan5M) {
        message.error('图片大小不能超过5MB');
        return Upload.LIST_IGNORE;
      }
      
      // 设置文件
      setDesignImage({
        uid: file.uid,
        name: file.name,
        status: 'done',
        url: URL.createObjectURL(file),
        originFileObj: file
      });
      
      // 阻止自动上传
      return false;
    },
    onRemove: () => {
      setDesignImage(null);
      setUploadResult(null);
      return true;
    },
    onChange: (info) => {
      if (info.fileList.length === 0) {
        setDesignImage(null);
        setUploadResult(null);
      }
    }
  };
  
  // 在状态定义部分添加跳过缓存选项
  const [skipCache, setSkipCache] = useState<boolean>(false);
  
  // 在状态定义部分添加Git仓库相关状态
  const [useGitRepo, setUseGitRepo] = useState<boolean>(false);
  const [gitRepoUrl, setGitRepoUrl] = useState<string>('');
  const [gitRepoBranch, setGitRepoBranch] = useState<string>('main');
  const [gitRepoStatus, setGitRepoStatus] = useState<GitRepoStatus>({ 
    status: 'idle' 
  });
  
  // 在RAG和Agent参数状态下方添加错误信息状态
  const [errorDetails, setErrorDetails] = useState<any>(null);
  
  // 向量数据库状态
  const [vectorDbStatus, setVectorDbStatus] = useState<'ready' | 'initializing' | 'error' | 'partial'>('initializing');
  const [vectorDbError, setVectorDbError] = useState<string | null>(null);
  
  // 顶部import部分添加state定义
  const [gitUsername, setGitUsername] = useState<string>('');
  const [gitPassword, setGitPassword] = useState<string>('');
  const [showGitCredentials, setShowGitCredentials] = useState<boolean>(false);
  
  // 添加新的提示说明
  const [authTips, setAuthTips] = useState<boolean>(false);
  
  // 检查向量数据库状态
  const checkVectorDbStatus = async () => {
    try {
      console.log('正在检查向量数据库状态...');
      const response = await fetch(`${API_BASE_URL}/api/vector-db-status`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        mode: 'cors',
        cache: 'no-cache'
      });
      
      if (!response.ok) {
        console.error('向量数据库状态检查失败:', response.status, response.statusText);
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Vector DB Status:', data);
      
      if (data.error) {
        console.error('向量数据库错误:', data.error);
        setVectorDbStatus('error');
        setVectorDbError(data.error);
        return false;
      }
      
      // 使用新的status字段
      setVectorDbStatus(data.status);
      
      if (data.status === 'error') {
        setVectorDbError(data.error || '向量数据库初始化失败');
        console.warn('向量数据库状态: error', data.error || '无错误信息');
        return false;
      } else if (data.status === 'initializing') {
        setVectorDbError('向量数据库正在初始化中，请稍后再试');
        console.info('向量数据库状态: initializing');
        return false;
      } else if (data.status === 'partial') {
        // 部分就绪状态，记录消息但允许使用
        setVectorDbError(data.message || '向量数据库部分初始化，某些功能可能受限');
        console.info('向量数据库状态: partial', data.message);
        return true;
      } else if (data.status === 'ready') {
        setVectorDbError(null);
        console.info('向量数据库状态: ready');
        return true;
      }
      
      // 兼容旧版API，如果没有status字段则使用is_ready
      if (data.is_ready === true) {
        console.info('向量数据库就绪 (兼容模式)');
        return true;
      }
      
      console.warn('向量数据库状态未知:', data);
      return false;
      
    } catch (error) {
      console.error('检查向量数据库状态失败:', error);
      setVectorDbStatus('error');
      setVectorDbError(error instanceof Error ? error.message : '检查向量数据库状态失败');
      return false;
    }
  };

  // 定期检查向量数据库状态
  useEffect(() => {
    let intervalId: NodeJS.Timeout;
    let retryCount = 0;
    const maxRetries = 5;  // 最大重试次数
    
    const startStatusCheck = async () => {
      // 立即检查一次
      const isReady = await checkVectorDbStatus();
      
      if (!isReady && retryCount < maxRetries) {
        // 如果数据库未就绪，每5秒检查一次
        intervalId = setInterval(async () => {
          retryCount++;
          const ready = await checkVectorDbStatus();
          
          if (ready || retryCount >= maxRetries) {
            clearInterval(intervalId);
            if (!ready && retryCount >= maxRetries) {
              console.error('向量数据库初始化超时');
              setVectorDbError('向量数据库初始化超时，请刷新页面重试');
            }
          }
        }, 5000);
      }
    };
    
    startStatusCheck();
    
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, []);

  // 上传设计图
  const handleUploadDesign = async () => {
    if (vectorDbStatus !== 'ready' && vectorDbStatus !== 'partial') {
      message.warning('请等待向量数据库初始化完成');
      return;
    }

    if (!designImage?.originFileObj) {
      message.error('请先选择设计图');
      return;
    }
    
    try {
      setUploadStatus({ status: 'processing' });
      setFullScreenLoading(true);
      setLoadingMessage('正在上传设计图...');
      
      const formData = new FormData();
      formData.append('file', designImage.originFileObj);
      formData.append('tech_stack', techStack);
      
      const response = await fetch(`${API_BASE_URL}/api/design/upload`, {
        method: 'POST',
        body: formData,
        mode: 'cors',
        cache: 'no-cache'
      });
      
      if (!response.ok) {
        throw new Error(`上传失败: ${response.status} ${response.statusText}`);
      }
      
      const result = await response.json();
      setUploadResult(result);
      setUploadStatus({ status: 'completed' });
      
      // 检查向量存储状态
      if (result.vector_store_success === false) {
        message.warning('设计图已保存，但未成功添加到向量数据库，某些功能可能受限');
        console.warn('设计图向量存储失败，可能会影响检索和生成质量');
      } else {
        message.success('设计图上传成功');
      }
      
    } catch (error) {
      console.error('上传设计图失败:', error);
      setUploadStatus({ 
        status: 'error', 
        message: error instanceof Error ? error.message : '上传失败' 
      });
      message.error(`上传设计图失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setFullScreenLoading(false);
    }
  };
  
  // 单独分析Git仓库
  const handleAnalyzeGitRepo = async () => {
    try {
      if (!gitRepoUrl.trim()) {
        message.error('请输入项目本地路径');
        return;
      }
      
      setGitRepoStatus({
        status: 'processing',
        message: '正在分析项目代码...'
      });
      
      // 准备请求数据
      const formData = new FormData();
      formData.append('local_project_path', gitRepoUrl.trim());
      formData.append('git_repo_branch', gitRepoBranch); // 保留分支参数以支持本地Git项目
      
      // 发送请求到新的本地项目分析接口
      const response = await fetch(`${API_BASE_URL}/api/analyze-local-project`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || '分析项目失败');
      }
      
      const data = await response.json();
      
      if (data.status === 'error') {
        setGitRepoStatus({
          status: 'error',
          message: '项目分析失败',
          error: data.error,
          canRetry: true
        });
        message.error(`项目分析失败: ${data.error}`);
        return;
      }
      
      // 设置成功状态
      setGitRepoStatus({
        status: 'completed',
        message: '项目分析完成',
        summary: data.summary,
        processingTime: data.processing_time
      });
      
      message.success('项目分析完成');
      
    } catch (error) {
      console.error('项目分析错误:', error);
      
      setGitRepoStatus({
        status: 'error',
        message: '项目分析失败',
        error: error instanceof Error ? error.message : String(error),
        canRetry: true
      });
      
      message.error(`项目分析失败: ${error instanceof Error ? error.message : String(error)}`);
    }
  };
  
  // 生成Prompt
  const handleGeneratePrompt = async () => {
    if (vectorDbStatus !== 'ready' && vectorDbStatus !== 'partial') {
      message.warning('请等待向量数据库初始化完成');
      return;
    }

    try {
      if (!uploadResult) {
        message.error('请先上传设计图');
        return;
      }
      
      // 检查Git仓库分析状态
      if (useGitRepo && gitRepoUrl) {
        if (gitRepoStatus.status === 'idle') {
          const shouldProceed = await new Promise<boolean>(resolve => {
            Modal.confirm({
              title: 'Git仓库尚未分析',
              content: '您已启用Git仓库上下文，但尚未分析仓库。是否先分析Git仓库再生成提示词？',
              okText: '先分析Git仓库',
              cancelText: '跳过分析',
              onOk: () => {
                resolve(false);
                handleAnalyzeGitRepo();
              },
              onCancel: () => resolve(true)
            });
          });
          
          if (!shouldProceed) return;
        } else if (gitRepoStatus.status === 'error') {
          const shouldProceed = await new Promise<boolean>(resolve => {
            Modal.confirm({
              title: 'Git仓库分析失败',
              content: '之前的Git仓库分析失败。是否继续生成提示词？（这将不包含Git仓库上下文）',
              okText: '继续',
              cancelText: '取消',
              onOk: () => resolve(true),
              onCancel: () => resolve(false)
            });
          });
          
          if (!shouldProceed) return;
        }
      }
      
      setGenerationStatus({ status: 'processing' });
      setFullScreenLoading(true);
      setLoadingMessage('正在生成Prompt...');
      setErrorDetails(null); // 清除之前的错误详情
      setGeneratedPrompt(''); // 清除之前生成的Prompt
      
      const startTime = Date.now();
      
      // 准备请求数据
      const requestData: any = {
        tech_stack: techStack,
        design_image_id: uploadResult.image_id,
        design_image_path: uploadResult.file_path,
        rag_method: ragMethod,
        retriever_top_k: retrieverTopK,
        agent_type: agentType,
        temperature: temperature,
        context_window_size: contextWindowSize,
        skip_cache: skipCache // 添加跳过缓存参数
      };
      
      // 添加Git仓库信息（如果启用）
      if (useGitRepo && gitRepoUrl) {
        requestData.git_repo_url = gitRepoUrl;
        requestData.git_repo_branch = gitRepoBranch;
      }
      
      // 添加前端重试逻辑
      const maxRetries = 2; // 前端最多重试2次
      let retryCount = 0;
      let response;
      let errorOccurred = false;
      
      while (retryCount <= maxRetries) {
        try {
          if (retryCount > 0) {
            setLoadingMessage(`正在生成Prompt...（第${retryCount}次重试）`);
            console.log(`第${retryCount}次重试生成Prompt`);
            // 重试前等待一段时间（指数退避）
            await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, retryCount - 1)));
          }
          
          response = await fetch(`${API_BASE_URL}/api/design/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(requestData),
        mode: 'cors',
        cache: 'no-cache'
      });
      
          if (response.ok) {
            // 成功响应，跳出重试循环
            errorOccurred = false;
            break;
          }
          
          // 判断是否是500内部错误或422错误（可能是临时性的）
          if (response.status === 500 || response.status === 422) {
            const errorData = await response.json().catch(() => ({}));
            // 如果是OpenAI的内部错误，尝试重试
            if ((errorData?.error?.message?.toLowerCase().includes('internal error') ||
                errorData?.detail?.message?.toLowerCase().includes('internal error') ||
                errorData?.detail?.error_type === 'OPENAI_INTERNAL_ERROR')) {
              
              // 保存错误详情以供显示
              setErrorDetails(errorData?.detail || errorData);
              
              errorOccurred = true;
              retryCount++;
              console.log('检测到OpenAI API内部错误，准备重试', errorData);
              
              if (retryCount <= maxRetries) {
                continue; // 继续下一次重试
              }
            }
          }
          
          // 其他错误或重试次数已用完，跳出循环
          errorOccurred = true;
          break;
          
        } catch (error) {
          console.error(`Prompt生成请求失败 (尝试 ${retryCount + 1}/${maxRetries + 1})`, error);
          errorOccurred = true;
          retryCount++;
          
          if (retryCount <= maxRetries) {
            continue; // 继续下一次重试
          }
          break;
        }
      }
      
      // 处理所有重试后的结果
      if (errorOccurred || !response || !response.ok) {
        // 所有重试都失败了
        if (!response) {
          throw new Error('无法连接到服务器，请检查网络连接');
        }
        
        const errorData = await response.json().catch(() => null);
        let errorMessage = "";
        
        if (errorData) {
          // 保存详细错误信息供显示
          setErrorDetails(errorData?.detail || errorData);
          
          if (errorData.detail && typeof errorData.detail === 'object') {
            // 处理结构化错误信息
            errorMessage = errorData.detail.message || JSON.stringify(errorData.detail);
          } else if (errorData.detail) {
            // 字符串形式的错误信息
            errorMessage = errorData.detail;
          } else if (errorData.message) {
            // 一般的消息字段
            errorMessage = errorData.message;
          } else if (errorData.error && errorData.error.message) {
            // 嵌套的错误对象
            errorMessage = errorData.error.message;
          } else {
            // 尝试将整个对象转为字符串
            errorMessage = JSON.stringify(errorData);
          }
        }
        
        if (!errorMessage) {
          errorMessage = `生成失败: ${response.status} ${response.statusText}`;
        }
        
        // 对于OpenAI的内部错误，提供更友好的提示
        if (errorMessage.toLowerCase().includes('internal error') || 
            response.status === 500 || 
            (errorData?.detail?.error_type === 'OPENAI_INTERNAL_ERROR')) {
          errorMessage = `OpenAI API服务器内部错误。这通常是暂时性问题，请尝试以下解决方案：
1. 勾选"跳过缓存"选项重试
2. 等待几分钟后再试
3. 联系管理员检查API密钥配置

详细信息: ${errorMessage}`;
        }
        
        throw new Error(errorMessage);
      }
      
      // 成功获取结果
      const result = await response.json();
      
      // 添加日志，输出整个响应对象以便调试
      console.log('API响应数据:', result);
      
      // 从result.prompt中获取生成的提示词（API返回的字段是prompt而不是generated_prompt）
      const promptContent = result.prompt || result.generated_prompt || '';
      
      if (!promptContent) {
        console.warn('API响应中没有找到prompt或generated_prompt字段:', result);
      }
      
      setGeneratedPrompt(promptContent);
      setEditedPrompt(promptContent);
      setHasHistoryContext(result.has_history_context);
      setHistoryPrompts(result.history_prompts || []);
      setGenerationStatus({ status: 'completed' });
      
      // 显示缓存状态
      const cacheHit = result.cache_hit || false;
      if (cacheHit) {
        message.success('Prompt生成成功 (使用缓存)');
      } else {
      message.success('Prompt生成成功');
      }
      
      // 记录环境检查信息
      if (result.env_check) {
        console.log('环境变量检查结果:', result.env_check);
      }
      
      // 如果是兜底prompt，设置状态为警告
      if (result.is_fallback_prompt) {
        setGenerationStatus({ 
          status: 'completed',
          message: '没有找到足够的相似设计图或历史Prompt，已使用默认提示词生成'
        });
      } else {
        setGenerationStatus({ status: 'completed' });
      }
    } catch (error: any) {
      console.error('生成Prompt失败:', error);
      let errorMessage = error instanceof Error ? error.message : '未知错误';
      
      // 增加更多的错误信息详情输出到控制台，帮助调试
      console.error('错误详情:', error);
      
      setGenerationStatus({ 
        status: 'error', 
        message: errorMessage
      });
      
      // 使用Alert组件展示更详细的错误信息
      message.error({
        content: '生成Prompt失败，查看详细错误信息',
        duration: 5
      });
      
      // 即使出错，也设置一个默认提示词
      if (!generatedPrompt) {
        const defaultPrompt = `# ${techStack}应用开发提示词\n\n## 生成过程发生错误\n\n无法生成提示词，请尝试以下解决方案：\n1. 勾选"跳过缓存"选项重试\n2. 刷新页面后重试\n3. 使用不同的技术栈\n4. 上传不同的设计图\n5. 稍后再试\n\n错误信息: ${errorMessage}`;
        setGeneratedPrompt(defaultPrompt);
        setEditedPrompt(defaultPrompt);
      }
    } finally {
      setFullScreenLoading(false);
    }
  };
  
  // 保存用户修改后的Prompt
  const handleSavePrompt = async () => {
    if (!uploadResult) {
      message.error('请先上传设计图');
      return;
    }
    
    if (!editedPrompt.trim()) {
      message.error('Prompt不能为空');
      return;
    }
    
    try {
      setSaveStatus({ status: 'processing' });
      setFullScreenLoading(true);
      setLoadingMessage('正在保存Prompt...');
      
      const requestData = {
        prompt: editedPrompt,
        tech_stack: techStack,
        design_image_id: uploadResult.image_id
      };
      
      const response = await fetch(`${API_BASE_URL}/api/design/save`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(requestData),
        mode: 'cors',
        cache: 'no-cache'
      });
      
      const data = await response.json();
      
      if (response.ok) {
        if (data.status === 'success') {
          setSaveStatus({ status: 'completed' });
          setShowSaveConfirm(false);
          message.success('Prompt保存成功');
        } else if (data.status === 'partial_success') {
          setSaveStatus({ status: 'completed' });
          setShowSaveConfirm(false);
          message.warning('Prompt已保存，但使用了备选方法：' + data.message);
        } else if (data.status === 'warning') {
          setSaveStatus({ status: 'completed' });
          setShowSaveConfirm(false);
          message.warning('Prompt已保存，但可能存在问题：' + data.message);
        } else {
          throw new Error(data.message || '保存失败，状态码: ' + data.status);
        }
      } else {
        throw new Error(`保存失败: ${response.status} ${response.statusText}`);
      }
      
    } catch (error) {
      console.error('保存Prompt失败:', error);
      setSaveStatus({ 
        status: 'error', 
        message: error instanceof Error ? error.message : '保存失败' 
      });
      message.error(`保存Prompt失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setFullScreenLoading(false);
    }
  };
  
  // 开始编辑
  const handleStartEditing = () => {
    setIsEditing(true);
  };
  
  // 取消编辑
  const handleCancelEditing = () => {
    setEditedPrompt(generatedPrompt);
    setIsEditing(false);
  };
  
  // 确认保存
  const handleConfirmSave = () => {
    setShowSaveConfirm(true);
  };
  
  // 取消保存
  const handleCancelSave = () => {
    setShowSaveConfirm(false);
  };
  
  // 设计分析结果展示
  const renderDesignAnalysisSimple = () => {
    if (!uploadResult || !uploadResult.design_analysis) {
      return null;
    }

    return (
      <Card title="设计分析摘要" style={{ marginTop: 16 }}>
        <Typography.Paragraph>
          {uploadResult.design_analysis.summary || '无设计分析摘要'}
        </Typography.Paragraph>
      </Card>
    );
  };
  
  // 渲染设置面板
  const renderSettingsPanel = () => {
    return (
      <div style={{ marginBottom: 16 }}>
        <Collapse defaultActiveKey={['settings']}>
          <Panel header="高级设置" key="settings">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div>
                <Typography.Text strong>技术栈</Typography.Text>
                <Select
                  style={{ width: '100%', marginTop: 8 }}
                  value={techStack}
                  onChange={setTechStack}
                >
                  {TECH_STACKS.map(tech => (
                    <Option key={tech} value={tech}>{tech}</Option>
                  ))}
                </Select>
              </div>
              
              <Divider style={{ margin: '8px 0' }} />
              
              {/* Git仓库设置 */}
              <div>
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
                  <Checkbox 
                    checked={useGitRepo} 
                    onChange={(e) => setUseGitRepo(e.target.checked)}
                    style={{ marginRight: 8 }}
                  />
                  <Typography.Text strong>使用代码仓库作为上下文</Typography.Text>
                  <Tooltip title="通过分析代码仓库，提高生成Prompt的准确性">
                    <QuestionCircleOutlined style={{ marginLeft: 8 }} />
                  </Tooltip>
                </div>
                
                {useGitRepo && (
                  <>
                    <Row gutter={16}>
                      <Col span={16}>
                        <Form.Item
                          label={
                            <span>
                              <LinkOutlined /> 项目本地路径
                              <Tooltip title="输入项目在本地电脑上的完整路径">
                                <InfoCircleOutlined style={{ marginLeft: 4 }} />
                              </Tooltip>
                            </span>
                          }
                          name="gitRepoUrl"
                        >
                          <Input 
                            placeholder="例如: C:\Projects\MyProject 或 /Users/username/Projects/MyProject" 
                            value={gitRepoUrl}
                            onChange={e => setGitRepoUrl(e.target.value)}
                            disabled={gitRepoStatus.status === 'processing' || generationStatus.status === 'processing'}
                          />
                        </Form.Item>
                      </Col>
                      <Col span={8}>
                        <Button 
                          type="primary"
                          onClick={handleAnalyzeGitRepo}
                          loading={gitRepoStatus.status === 'processing'}
                          disabled={!gitRepoUrl || generationStatus.status === 'processing'}
                          icon={<DatabaseOutlined />}
                          style={{ width: '100%' }}
                        >
                          分析本地项目
                        </Button>
                      </Col>
                    </Row>
                    
                    {renderGitRepoStatus()}
                  </>
                )}
              </div>
              
              <Divider style={{ margin: '8px 0' }} />
              
              <div>
                <Typography.Text strong>检索方法</Typography.Text>
                <Select
                  style={{ width: '100%', marginTop: 8 }}
                  value={ragMethod}
                  onChange={setRagMethod}
                >
                  <Option value="similarity">相似度检索</Option>
                  <Option value="mmr">最大边际相关性 (MMR)</Option>
                  <Option value="hybrid">混合检索</Option>
                </Select>
              </div>
              
              <div>
                <Typography.Text strong>温度 ({temperature})</Typography.Text>
                <Slider
                  min={0}
                  max={1}
                  step={0.1}
                  value={temperature}
                  onChange={setTemperature}
                />
              </div>
              
              <div>
                <Typography.Text strong>检索结果数量 ({retrieverTopK})</Typography.Text>
                <Slider
                  min={1}
                  max={10}
                  step={1}
                  value={retrieverTopK}
                  onChange={setRetrieverTopK}
                />
              </div>
              
              <div>
                <Typography.Text strong>上下文窗口大小 ({contextWindowSize})</Typography.Text>
                <Slider
                  min={1000}
                  max={8000}
                  step={1000}
                  value={contextWindowSize}
                  onChange={setContextWindowSize}
                />
              </div>
              
              <div>
                <Checkbox
                  checked={skipCache}
                  onChange={(e) => setSkipCache(e.target.checked)}
                >
                  跳过缓存（强制重新生成）
                </Checkbox>
              </div>
            </div>
          </Panel>
        </Collapse>
      </div>
    );
  };
  
  // 添加Git仓库状态显示组件
  const renderGitRepoStatus = () => {
    if (!useGitRepo || !gitRepoUrl) {
      return null;
    }
    
    if (gitRepoStatus.status === 'idle') {
      return null;
    }
    
    if (gitRepoStatus.status === 'processing') {
      return (
        <Alert
          type="info"
          showIcon
          icon={<SyncOutlined spin />}
          message="项目分析中"
          description={
            <div>
              <p>正在分析本地项目: {gitRepoUrl}</p>
              <p>请稍候，这可能需要几秒钟...</p>
            </div>
          }
        />
      );
    }
    
    if (gitRepoStatus.status === 'error') {
      return (
        <Alert
          type="error"
          showIcon
          message="项目分析失败"
          description={
            <div>
              <p>分析本地项目时出错: {gitRepoStatus.error}</p>
              <p>请检查项目路径是否正确，并确保有访问权限。</p>
              <Button 
                type="primary" 
                size="small" 
                onClick={handleAnalyzeGitRepo}
                style={{ marginTop: 8 }}
              >
                重试
              </Button>
            </div>
          }
        />
      );
    }
    
    if (gitRepoStatus.status === 'completed') {
      return (
        <Alert
          type="success"
          showIcon
          message="项目分析完成"
          description={
            <div>
              <p>本地项目路径: {gitRepoUrl}</p>
              {gitRepoStatus.processingTime && (
                <p>处理耗时: {gitRepoStatus.processingTime.toFixed(2)}秒</p>
              )}
              {gitRepoStatus.summary && (
                <div>
                  <p><strong>项目概述:</strong></p>
                  <div style={{ 
                    maxHeight: '100px', 
                    overflowY: 'auto', 
                    padding: '8px', 
                    background: '#f5f5f5', 
                    borderRadius: '4px',
                    marginTop: '8px'
                  }}>
                    {gitRepoStatus.summary}
                  </div>
                </div>
              )}
            </div>
          }
        />
      );
    }
    
    return null;
  };
  
  // 渲染生成状态
  const renderGenerationStatus = () => {
    if (generationStatus.status === 'idle') {
      return null;
    }
    
    if (generationStatus.status === 'processing') {
      return (
        <div className="generation-status processing">
          <Spin />
          <span>正在生成Prompt...</span>
        </div>
      );
    }
    
    if (generationStatus.status === 'error') {
      return (
        <div className="generation-status error">
          <Alert 
            type="error" 
            message="生成失败" 
            description={generationStatus.message || '未知错误'} 
            showIcon 
          />
          {generationStatus.canRetry && (
            <Button 
              type="primary" 
              onClick={handleGeneratePrompt} 
              style={{ marginTop: 16 }}
            >
              重试
            </Button>
          )}
        </div>
      );
    }
    
    if (generationStatus.status === 'completed' && generationStatus.message) {
      return (
        <div className="generation-status warning">
          <Alert 
            type="warning" 
            message="使用兜底方案" 
            description={generationStatus.message} 
            showIcon 
          />
          {generationStatus.canRetry && (
            <Button 
              type="primary" 
              onClick={handleGeneratePrompt} 
              style={{ marginTop: 16 }}
            >
              重新生成
            </Button>
          )}
        </div>
      );
    }
    
    return null;
  };
  
  // 格式化生成的prompt
  const formatGeneratedPrompt = (prompt: string): React.ReactNode => {
    if (!prompt) return null;
    
    // 检查是否是Markdown格式
    const isMdFormat = prompt.includes('##') || prompt.includes('#');
    
    if (isMdFormat) {
      // 使用ReactMarkdown渲染Markdown内容
      return (
        <div className="markdown-content" style={{ padding: '10px' }}>
          <ReactMarkdown>{prompt}</ReactMarkdown>
        </div>
      );
    }
    
    // 以下是原有的格式化逻辑，用于处理非Markdown格式
    // 分割文本为段落
    const sections = prompt.split(/\d+\.\s+/).filter(section => section.trim());
    
    // 如果没有找到数字编号的段落，则直接返回原文本
    if (sections.length === 0) {
      return <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit' }}>{prompt}</pre>;
    }
    
    // 提取标题（如果有的话）
    let title = '';
    let content = prompt;
    
    if (prompt.startsWith('1.')) {
      // 已经是格式化的内容，直接处理
    } else {
      // 尝试提取标题，使用[\s\S]代替/s标志
      const titleMatch = prompt.match(/^([\s\S]*?)(?=\d+\.\s+)/);
      if (titleMatch && titleMatch[0]) {
        title = titleMatch[0].trim();
        content = prompt.substring(title.length).trim();
      }
    }
    
    // 分割内容为带编号的段落
    const numberedSections = content.split(/(\d+\.\s+[^\d]+)(?=\d+\.\s+|$)/g)
      .filter(section => section.trim());
    
    return (
      <div className="formatted-prompt">
        {title && <div className="prompt-title">{title}</div>}
        {numberedSections.map((section, index) => {
          // 提取段落标题和内容，使用[\s\S]代替/s标志
          const sectionMatch = section.match(/(\d+\.\s+)([^:：]+)[:：]?\s*([\s\S]*)/);
          
          if (sectionMatch) {
            const [, number, sectionTitle, sectionContent] = sectionMatch;
            
            // 进一步分割内容为子项
            const subItems = sectionContent.split(/[-•]\s+/).filter(item => item.trim());
            
            return (
              <div key={index} className="prompt-section" style={{ marginBottom: '16px' }}>
                <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>
                  {number}{sectionTitle.trim()}
                </div>
                {subItems.length > 1 ? (
                  <ul style={{ paddingLeft: '20px', margin: '8px 0' }}>
                    {subItems.map((item, i) => (
                      <li key={i} style={{ marginBottom: '4px' }}>{item.trim()}</li>
                    ))}
                  </ul>
                ) : (
                  <div style={{ paddingLeft: '20px' }}>{sectionContent.trim()}</div>
                )}
              </div>
            );
          }
          
          // 如果不匹配预期格式，直接显示原文
          return <div key={index} style={{ marginBottom: '8px' }}>{section}</div>;
        })}
      </div>
    );
  };
  
  // 复制文本到剪贴板的函数
  const copyToClipboard = (text: string) => {
    try {
      // 优先使用navigator.clipboard API
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text)
          .then(() => message.success('已复制到剪贴板'))
          .catch(() => {
            // 如果navigator.clipboard失败，使用copy-to-clipboard库
            const success = copy(text);
            if (success) {
              message.success('已复制到剪贴板');
            } else {
              message.error('复制失败，请手动复制');
            }
          });
      } else {
        // 直接使用copy-to-clipboard库
        const success = copy(text);
        if (success) {
          message.success('已复制到剪贴板');
        } else {
          message.error('复制失败，请手动复制');
        }
      }
    } catch (error) {
      console.error('复制失败:', error);
      message.error('复制失败，请手动复制');
    }
  };
  
  // 添加新的显示技术栈特定组件的面板
  const renderTechStackSpecificComponents = (repoAnalysis: any) => {
    if (!repoAnalysis || !repoAnalysis.ui_components || repoAnalysis.ui_components.length === 0) {
      return <Empty description="无可用组件" />;
    }
    
    return (
      <Collapse defaultActiveKey={['1']}>
        <Panel header="项目组件库" key="1">
          <List
            dataSource={repoAnalysis.ui_components}
            renderItem={(component: any) => (
              <List.Item>
                <Card
                  size="small"
                  title={component.name}
                  extra={
                    <Tooltip title="使用此组件">
                      <Button 
                        type="text" 
                        icon={<PlusOutlined />}
                        onClick={() => addComponentToPrompt(component)}
                      />
                    </Tooltip>
                  }
                >
                  <p>{component.description}</p>
                  {component.usage_example && (
                    <SyntaxHighlighter language="typescript" style={atomOneDark}>
                      {component.usage_example}
                    </SyntaxHighlighter>
                  )}
                </Card>
              </List.Item>
            )}
          />
        </Panel>
      </Collapse>
    );
  };

  // 添加项目UI规范面板
  const renderUIStandards = (repoAnalysis: any) => {
    if (!repoAnalysis || !repoAnalysis.ui_standards) {
      return <Empty description="无可用UI规范" />;
    }
    
    return (
      <Collapse defaultActiveKey={['1']}>
        <Panel header="项目UI规范" key="1">
          <List
            dataSource={Object.entries(repoAnalysis.ui_standards || {})}
            renderItem={([key, value]: [string, any]) => (
              <List.Item>
                <Text strong>{key}:</Text> {value}
              </List.Item>
            )}
          />
        </Panel>
      </Collapse>
    );
  };

  // 添加生成提示词评估结果展示
  const renderEvaluationResults = (promptEvaluation: any) => {
    if (!promptEvaluation) return null;
    
    // 评估维度中文标签
    const dimensionLabels: {[key: string]: string} = {
      ui_completeness: "UI完整度",
      layout_accuracy: "布局准确性",
      tech_stack_relevance: "技术栈相关性",
      component_reuse: "组件复用",
      style_consistency: "样式一致性"
    };
    
    return (
      <Card title="提示词评估" className="evaluation-card">
        <Row gutter={[16, 16]}>
          {Object.entries(promptEvaluation.scores || {}).map(([dimension, score]) => (
            <Col span={12} key={dimension}>
              <Statistic
                title={dimensionLabels[dimension] || dimension}
                value={score as number}
                suffix="/10"
                valueStyle={{ color: (score as number) >= 7 ? '#3f8600' : (score as number) >= 5 ? '#cf9f00' : '#cf1322' }}
              />
            </Col>
          ))}
        </Row>
        
        {promptEvaluation.improvement_suggestions && (
          <div className="improvement-suggestions">
            <Divider />
            <Title level={5}>改进建议</Title>
            <ReactMarkdown>{promptEvaluation.improvement_suggestions}</ReactMarkdown>
          </div>
        )}
      </Card>
    );
  };

  // 添加设计分析结果可视化
  const renderDesignAnalysis = (designAnalysis: any) => {
    if (!designAnalysis || !designAnalysis.tech_stack_analysis) {
      return null;
    }
    
    return (
      <Card title="设计分析结果" className="design-analysis-card">
        <Tabs defaultActiveKey="1">
          <TabPane tab="UI组件" key="1">
            <List
              dataSource={designAnalysis.ui_components || []}
              renderItem={(component: any) => (
                <List.Item>
                  <Card
                    size="small"
                    title={component.name}
                    extra={
                      <Tag color="blue">{component.tech_stack}</Tag>
                    }
                  >
                    <p>{component.description}</p>
                  </Card>
                </List.Item>
              )}
            />
          </TabPane>
          <TabPane tab="配色方案" key="2">
            <Row gutter={[16, 16]}>
              {(designAnalysis.color_scheme?.primary_colors || []).map((color: any, index: number) => (
                <Col span={8} key={index}>
                  <Card bodyStyle={{ padding: 10 }}>
                    <div style={{ 
                      backgroundColor: color.color, 
                      height: 80, 
                      borderRadius: 4,
                      marginBottom: 8
                    }} />
                    <div>
                      <Text strong>{color.role.toUpperCase()}</Text>
                      <br />
                      <Text>{color.color}</Text>
                      <br />
                      <Text type="secondary">{Math.round(color.percentage * 100)}%</Text>
                    </div>
                  </Card>
                </Col>
              ))}
            </Row>
            <Divider orientation="left">调色板</Divider>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {(designAnalysis.color_scheme?.color_palette || []).map((color: string, index: number) => (
                <Tooltip title={color} key={index}>
                  <div style={{ 
                    backgroundColor: color, 
                    width: 40, 
                    height: 40, 
                    borderRadius: 4,
                    cursor: 'pointer',
                    border: '1px solid #f0f0f0'
                  }} onClick={() => copy(color)} />
                </Tooltip>
              ))}
            </div>
          </TabPane>
          <TabPane tab="排版" key="3">
            <Descriptions bordered column={1}>
              <Descriptions.Item label="字体家族">{designAnalysis.typography?.font_family || '未识别'}</Descriptions.Item>
            </Descriptions>
            <Divider orientation="left">文本样式</Divider>
            <List
              dataSource={designAnalysis.typography?.text_styles || []}
              renderItem={(style: any) => (
                <List.Item>
                  <Text strong>{style.type}:</Text> {style.size}
                </List.Item>
              )}
            />
          </TabPane>
          <TabPane tab="间距" key="4">
            <Descriptions bordered column={1}>
              <Descriptions.Item label="水平间距">{designAnalysis.spacing?.horizontal_spacing || '未识别'}</Descriptions.Item>
              <Descriptions.Item label="垂直间距">{designAnalysis.spacing?.vertical_spacing || '未识别'}</Descriptions.Item>
              <Descriptions.Item label="内边距">{designAnalysis.spacing?.padding || '未识别'}</Descriptions.Item>
              <Descriptions.Item label="外边距">{designAnalysis.spacing?.margin || '未识别'}</Descriptions.Item>
              <Descriptions.Item label="对齐方式">{designAnalysis.spacing?.alignment || '未识别'}</Descriptions.Item>
            </Descriptions>
          </TabPane>
        </Tabs>
      </Card>
    );
  };

  // 添加组件到提示词的功能
  const addComponentToPrompt = (component: any) => {
    // 获取当前编辑状态的提示词
    const currentPrompt = isEditing ? editedPrompt : generatedPrompt;
    
    // 构建组件描述
    const componentDesc = `\n\n## 使用组件: ${component.name}\n${component.description}`;
    
    // 添加到提示词
    if (isEditing) {
      setEditedPrompt(currentPrompt + componentDesc);
    } else {
      setGeneratedPrompt(currentPrompt + componentDesc);
      setEditedPrompt(currentPrompt + componentDesc);
      setIsEditing(true);
    }
    
    message.success(`已添加组件: ${component.name}`);
  };

  return (
    <div style={{ padding: '20px' }}>
      <Title level={2}>设计图Prompt生成</Title>
      <Paragraph>
        上传设计图并选择技术栈，生成对应的开发提示词。
      </Paragraph>
      
      {/* 设置面板 */}
      {renderSettingsPanel()}
      
      <Card title="步骤1: 上传设计图" style={{ marginBottom: '20px' }}>
        <Form layout="vertical">
          <Form.Item label="选择技术栈">
            <Select 
              value={techStack} 
              onChange={setTechStack}
              style={{ width: '100%' }}
            >
              {TECH_STACKS.map(stack => (
                <Option key={stack} value={stack}>{stack}</Option>
              ))}
            </Select>
          </Form.Item>
          
          <Form.Item label="上传设计图">
            <Upload {...uploadProps} listType="picture">
              <Button icon={<UploadOutlined />}>选择设计图</Button>
            </Upload>
            <div style={{ marginTop: '10px' }}>
              <Text type="secondary">支持的格式: {SUPPORTED_IMAGE_FORMATS.join(', ')}, 最大5MB</Text>
            </div>
          </Form.Item>
          
          <Form.Item>
            <Button 
              type="primary" 
              onClick={handleUploadDesign}
              loading={uploadStatus.status === 'processing'}
              disabled={!designImage}
            >
              上传设计图
            </Button>
          </Form.Item>
        </Form>
        
        {uploadStatus.status === 'error' && (
          <Alert 
            message="上传失败" 
            description={uploadStatus.message} 
            type="error" 
            showIcon 
            style={{ marginTop: '10px' }}
          />
        )}
        
        {uploadResult && (
          <Alert 
            message="上传成功" 
            description={`设计图ID: ${uploadResult.image_id}`} 
            type="success" 
            showIcon 
            style={{ marginTop: '10px' }}
          />
        )}
      </Card>
      
      <Card title="步骤2: 配置生成参数" style={{ marginBottom: '20px' }}>
        <Form layout="vertical">
          <Form.Item label="RAG方法">
            <Select 
              value={ragMethod} 
              onChange={setRagMethod}
              style={{ width: '100%' }}
            >
              {RAG_METHODS.map(method => (
                <Option key={method} value={method}>{method}</Option>
              ))}
            </Select>
          </Form.Item>
          
          <Form.Item label={`检索结果数量: ${retrieverTopK}`}>
            <Slider 
              min={1} 
              max={10} 
              value={retrieverTopK} 
              onChange={setRetrieverTopK} 
            />
          </Form.Item>
          
          <Form.Item label="Agent类型">
            <Select 
              value={agentType} 
              onChange={setAgentType}
              style={{ width: '100%' }}
            >
              {AGENT_TYPES.map(type => (
                <Option key={type} value={type}>{type}</Option>
              ))}
            </Select>
          </Form.Item>
          
          <Form.Item label={`温度: ${temperature}`}>
            <Slider 
              min={0} 
              max={1} 
              step={0.1} 
              value={temperature} 
              onChange={setTemperature} 
            />
          </Form.Item>
          
          <Form.Item label={`上下文窗口大小: ${contextWindowSize}`}>
            <Slider 
              min={1000} 
              max={8000} 
              step={1000} 
              value={contextWindowSize} 
              onChange={setContextWindowSize} 
            />
          </Form.Item>
          
          <Form.Item>
            <div style={{ marginBottom: '10px' }}>
            <Button 
              type="primary" 
              onClick={handleGeneratePrompt}
              loading={generationStatus.status === 'processing'}
              disabled={!uploadResult}
                style={{ marginRight: '10px' }}
            >
              生成Prompt
            </Button>
              
              <Checkbox 
                checked={skipCache} 
                onChange={(e) => setSkipCache(e.target.checked)}
                style={{ marginRight: '8px' }}
              >
                跳过缓存
              </Checkbox>
              
              <Tooltip title="选中此项将不使用缓存结果，重新生成Prompt。如遇到生成错误，请尝试勾选此选项。">
                <QuestionCircleOutlined style={{ color: '#1890ff' }} />
              </Tooltip>
            </div>
          </Form.Item>
        </Form>
        
        {renderGenerationStatus()}
      </Card>
      
      <Card 
        title="步骤3: 查看和编辑生成的Prompt" 
        style={{ marginBottom: '20px' }}
      >
        {generationStatus.status === 'completed' ? (
          <Row gutter={[16, 16]}>
            <Col span={16}>
              <Card 
                title="生成的提示词" 
                extra={
                  <Space>
                    <Button
                      icon={<EditOutlined />}
                      onClick={() => {
                        setIsEditing(true);
                        setEditedPrompt(generatedPrompt);
                      }}
                    >
                      编辑
                    </Button>
                    <Button
                      icon={<CopyOutlined />}
                      onClick={() => {
                        copy(generatedPrompt);
                        message.success('已复制到剪贴板');
                      }}
                    >
                      复制
                    </Button>
                  </Space>
                }
              >
                {isEditing ? (
                  <>
                    <TextArea
                      value={editedPrompt}
                      onChange={(e) => setEditedPrompt(e.target.value)}
                      style={{ height: 400 }}
                    />
                    <div style={{ marginTop: 16, textAlign: 'right' }}>
                      <Space>
                        <Button onClick={() => setIsEditing(false)}>取消</Button>
                        <Button 
                          type="primary"
                          onClick={() => {
                            setGeneratedPrompt(editedPrompt);
                            setIsEditing(false);
                            message.success('提示词已更新');
                          }}
                        >
                          保存
                        </Button>
                      </Space>
                    </div>
                  </>
                ) : (
                  <div className="generated-prompt-content">
                    {formatGeneratedPrompt(generatedPrompt)}
                  </div>
                )}
              </Card>
            </Col>
            <Col span={8}>
              <Space direction="vertical" style={{ width: '100%' }} size="large">
                {renderEvaluationResults(uploadResult?.evaluation_result)}
                {renderDesignAnalysis(uploadResult?.design_analysis)}
                {renderTechStackSpecificComponents(uploadResult?.repo_analysis)}
                {renderUIStandards(uploadResult?.repo_analysis)}
              </Space>
            </Col>
          </Row>
        ) : (
          <CustomEmpty description="请先生成Prompt" />
        )}
        
        {hasHistoryContext && historyPrompts.length > 0 && !isEditing && (
          <div style={{ marginTop: '20px' }}>
            <Divider orientation="left">参考的历史Prompt</Divider>
            <Collapse 
              items={historyPrompts.map((prompt, index) => {
                const metadata = prompt.metadata || {};
                return {
                  key: index,
                  label: `历史Prompt ${index + 1} (${metadata.tech_stack || '未知技术栈'})`,
                  children: (
                    <>
                    <div style={{ whiteSpace: 'pre-wrap' }}>
                      {prompt.text || '无内容'}
                    </div>
                    <div style={{ marginTop: '10px', color: '#888' }}>
                      <Text type="secondary">
                        创建时间: {metadata.created_at || '未知'}
                        {metadata.user_modified && ' | 用户修改: 是'}
                      </Text>
                    </div>
                    </>
                  )
                };
              })}
            />
          </div>
        )}
      </Card>
      
      {/* 全屏加载 */}
      {fullScreenLoading && (
        <div style={fullScreenLoadingStyle}>
          <Spin indicator={antIcon} />
          <div style={{ marginTop: '20px' }}>{loadingMessage}</div>
        </div>
      )}
      
      {/* 保存确认对话框 */}
      <Modal
        title="确认保存"
        open={showSaveConfirm}
        onOk={handleSavePrompt}
        onCancel={handleCancelSave}
        confirmLoading={saveStatus.status === 'processing'}
      >
        <p>确定要保存修改后的Prompt吗？保存后将用于未来的Prompt生成。</p>
      </Modal>
      
      {/* 显示错误详情 */}
      {errorDetails && generationStatus.status === 'error' && (
        <div style={{ marginBottom: '16px' }}>
          <Collapse>
            <Collapse.Panel header="错误详情" key="1">
              <div style={{ maxHeight: '300px', overflow: 'auto' }}>
                <pre>{JSON.stringify(errorDetails, null, 2)}</pre>
              </div>
              
              {errorDetails.env_check && (
                <div style={{ marginTop: '10px' }}>
                  <h4>环境变量检查结果：</h4>
                  <ul>
                    {Object.entries(errorDetails.env_check).map(([key, value]) => (
                      <li key={key}><strong>{key}:</strong> {String(value)}</li>
                    ))}
                  </ul>
                </div>
              )}
            </Collapse.Panel>
          </Collapse>
        </div>
      )}
    </div>
  );
};

// Empty组件定义
const CustomEmpty: React.FC<{ description: string }> = ({ description }) => (
  <Empty
    description={<Text type="secondary">{description}</Text>}
    style={{ padding: '40px 0' }}
  />
);

export default DesignPromptGenerator; 