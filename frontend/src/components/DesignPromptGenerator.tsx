import React, { useState } from 'react';
import { 
<<<<<<< HEAD
  Button, Card, Input, Upload, message, Select, Slider, Tabs, 
  Space, Alert, Spin, Typography, Radio, Image, Modal, Form, Divider, Collapse, Checkbox, Tooltip 
} from 'antd';
import { UploadOutlined, LoadingOutlined, SaveOutlined, EditOutlined, QuestionCircleOutlined } from '@ant-design/icons';
=======
  Button, Card, Input, Upload, message, Select, Slider, 
  Space, Alert, Spin, Typography, Modal, Form, Divider, Collapse, Empty 
} from 'antd';
import { UploadOutlined, LoadingOutlined, SaveOutlined, EditOutlined, CopyOutlined, CloseOutlined } from '@ant-design/icons';
>>>>>>> 82c1bcc0ead144b5abb7ab2621735f4f0e5a6b88
import type { UploadFile, UploadProps } from 'antd/es/upload/interface';
import { API_BASE_URL } from '../config';
import ReactMarkdown from 'react-markdown';
import copy from 'copy-to-clipboard';

const { TextArea } = Input;
const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { Panel } = Collapse;

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
  
  // 在RAG和Agent参数状态下方添加错误信息状态
  const [errorDetails, setErrorDetails] = useState<any>(null);
  
  // 上传设计图
  const handleUploadDesign = async () => {
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
      message.success('设计图上传成功');
      
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
  
  // 生成Prompt
  const handleGeneratePrompt = async () => {
    try {
      if (!uploadResult) {
        message.error('请先上传设计图');
        return;
      }

      setGenerationStatus({ status: 'processing' });
      setFullScreenLoading(true);
      setLoadingMessage('正在生成Prompt...');
<<<<<<< HEAD
      setErrorDetails(null); // 清除之前的错误详情
=======
      setGeneratedPrompt('');
>>>>>>> 82c1bcc0ead144b5abb7ab2621735f4f0e5a6b88
      
      const startTime = Date.now();
      
      // 准备请求数据
      const requestData = {
        tech_stack: techStack,
        design_image_id: uploadResult.image_id,
<<<<<<< HEAD
        design_image_path: uploadResult.image_path,
=======
        design_image_path: uploadResult.file_path,
>>>>>>> 82c1bcc0ead144b5abb7ab2621735f4f0e5a6b88
        rag_method: ragMethod,
        retriever_top_k: retrieverTopK,
        agent_type: agentType,
        temperature: temperature,
        context_window_size: contextWindowSize,
<<<<<<< HEAD
        skip_cache: skipCache // 添加跳过缓存参数
      };
      
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
      setGeneratedPrompt(result.generated_prompt);
      setEditedPrompt(result.generated_prompt);
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
=======
        prompt: "设计图Prompt生成请求"
      };

      // 发送请求
      const response = await fetch(`${API_BASE_URL}/api/design/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(requestData),
        mode: 'cors',
        cache: 'no-cache'
      });
      
      // 计算耗时
      const endTime = Date.now();
      const timeUsed = (endTime - startTime) / 1000;

      if (!response.ok) {
        let errorMessage = `生成失败: ${response.status} ${response.statusText}`;
        
        try {
          const errorData = await response.json();
          
          // 检查是否是OpenAI API错误
          if (response.status === 503 && errorData.detail && errorData.detail.error_type === 'openai_api_error') {
            errorMessage = errorData.detail.message || 'OpenAI API服务器暂时不可用，请稍后重试';
            message.error(errorMessage);
            
            // 设置重试按钮
            setGenerationStatus({ 
              status: 'error', 
              message: errorMessage,
              canRetry: true
            });
          } else if (errorData.detail) {
            errorMessage = typeof errorData.detail === 'string' 
              ? errorData.detail 
              : JSON.stringify(errorData.detail);
            
            setGenerationStatus({ 
              status: 'error', 
              message: errorMessage
            });
          }
        } catch (jsonError) {
          // 如果无法解析JSON，则使用文本响应
          const errorText = await response.text();
          errorMessage = `${errorMessage}\n${errorText}`;
          setGenerationStatus({ 
            status: 'error', 
            message: errorMessage
          });
        }
        
        throw new Error(errorMessage);
      }

      const data = await response.json();
      
      // 检查是否来自缓存
      const fromCache = data.from_cache || false;
      
      // 检查是否是兜底prompt
      const isFallbackPrompt = data.is_fallback_prompt || false;
      
      // 显示成功消息，包含耗时信息
      if (fromCache) {
        message.success(`Prompt生成成功 (从缓存获取)，耗时: ${timeUsed.toFixed(2)}秒`);
      } else if (isFallbackPrompt) {
        message.warning(`生成Prompt失败，已使用设计图分析结果作为兜底方案，耗时: ${timeUsed.toFixed(2)}秒`);
      } else {
        message.success(`Prompt生成成功，耗时: ${timeUsed.toFixed(2)}秒`);
      }
      
      setGeneratedPrompt(data.generated_prompt || '');
      setEditedPrompt(data.generated_prompt || '');
      setHasHistoryContext(data.has_history_context || false);
      setHistoryPrompts(data.history_prompts || []);
      setDesignAnalysis(data.design_analysis || {});
>>>>>>> 82c1bcc0ead144b5abb7ab2621735f4f0e5a6b88
      
      // 如果是兜底prompt，设置状态为警告
      if (isFallbackPrompt) {
        setGenerationStatus({ 
          status: 'completed',
          message: '由于生成失败，使用了设计图分析结果作为兜底方案。您可以编辑此Prompt或重试。',
          canRetry: true
        });
      } else {
        setGenerationStatus({ status: 'completed' });
      }
    } catch (error: any) {
      console.error('生成Prompt失败:', error);
<<<<<<< HEAD
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
=======
      
      // 如果状态已经设置为错误，则不再更新
      if (generationStatus.status !== 'error') {
        setGenerationStatus({ 
          status: 'error', 
          message: error instanceof Error ? error.message : '生成失败' 
        });
        message.error(`生成Prompt失败: ${error instanceof Error ? error.message : '未知错误'}`);
>>>>>>> 82c1bcc0ead144b5abb7ab2621735f4f0e5a6b88
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
      
      if (!response.ok) {
        throw new Error(`保存失败: ${response.status} ${response.statusText}`);
      }
      
      // 不需要使用结果变量
      await response.json();
      setSaveStatus({ status: 'completed' });
      setShowSaveConfirm(false);
      message.success('Prompt保存成功');
      
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
  
  // 添加设计分析展示组件
  const renderDesignAnalysis = () => {
    if (!designAnalysis || Object.keys(designAnalysis).length === 0) {
      return null;
    }

    // 准备Collapse的items
    const collapseItems = [];
    
    if (designAnalysis.design_style) {
      collapseItems.push({
        key: 'design_style',
        label: '设计风格',
        children: <Paragraph>{designAnalysis.design_style}</Paragraph>
      });
    }
    
    if (designAnalysis.layout) {
      collapseItems.push({
        key: 'layout',
        label: '布局结构',
        children: <Paragraph>{designAnalysis.layout}</Paragraph>
      });
    }
    
    if (designAnalysis.color_scheme) {
      collapseItems.push({
        key: 'color_scheme',
        label: '颜色方案',
        children: <Paragraph>{designAnalysis.color_scheme}</Paragraph>
      });
    }
    
    if (designAnalysis.ui_components) {
      collapseItems.push({
        key: 'ui_components',
        label: 'UI组件',
        children: Array.isArray(designAnalysis.ui_components) ? (
          <ul>
            {designAnalysis.ui_components.map((component: any, index: number) => (
              <li key={index}>
                {typeof component === 'object' ? 
                  `${component.name || '组件'}: ${component.description || ''}` : 
                  String(component)
                }
              </li>
            ))}
          </ul>
        ) : (
          <Paragraph>{designAnalysis.ui_components}</Paragraph>
        )
      });
    }
    
    if (designAnalysis.interaction_elements) {
      collapseItems.push({
        key: 'interaction_elements',
        label: '交互元素',
        children: <Paragraph>{designAnalysis.interaction_elements}</Paragraph>
      });
    }
    
    if (designAnalysis.usability) {
      collapseItems.push({
        key: 'usability',
        label: '可用性评估',
        children: <Paragraph>{designAnalysis.usability}</Paragraph>
      });
    }
    
    if (designAnalysis.tech_implementation) {
      collapseItems.push({
        key: 'tech_implementation',
        label: '技术实现建议',
        children: <Paragraph>{designAnalysis.tech_implementation}</Paragraph>
      });
    }

    return (
      <Card 
        title="设计分析结果" 
        style={{ marginTop: 16 }}
        extra={<Text type="secondary">AI分析结果</Text>}
      >
        <Collapse defaultActiveKey={['design_style', 'ui_components']} items={collapseItems} />
      </Card>
    );
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
  
  return (
    <div style={{ padding: '20px' }}>
      <Title level={2}>设计图Prompt生成</Title>
      <Paragraph>
        上传设计图并选择技术栈，生成对应的开发提示词。
      </Paragraph>
      
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
        
<<<<<<< HEAD
        {generationStatus.status === 'error' && (
          <Alert
            message="生成Prompt时发生错误"
            description={generationStatus.message}
            type="error"
            showIcon
            style={{ marginBottom: '16px' }}
          />
        )}
        
        {/* 显示错误详情 */}
        {errorDetails && generationStatus.status === 'error' && (
          <div style={{ marginBottom: '16px' }}>
            <Collapse>
              <Panel header="错误详情" key="1">
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
              </Panel>
            </Collapse>
          </div>
        )}
=======
        {renderGenerationStatus()}
>>>>>>> 82c1bcc0ead144b5abb7ab2621735f4f0e5a6b88
      </Card>
      
      <Card 
        title="步骤3: 查看和编辑生成的Prompt" 
        style={{ marginBottom: '20px' }}
      >
        {generationStatus.status === 'completed' ? (
          <div style={{ display: 'flex', flexDirection: 'row', gap: '16px' }}>
            <div style={{ flex: 1 }}>
              {/* 生成结果展示 */}
              <Card 
                title="生成的Prompt" 
                style={{ marginTop: 16 }}
                extra={
                  <Space>
                    {!isEditing ? (
                      <Button 
                        type="primary" 
                        icon={<EditOutlined />} 
                        onClick={handleStartEditing}
                        disabled={!generatedPrompt}
                      >
                        编辑
                      </Button>
                    ) : (
                      <>
                        <Button onClick={handleCancelEditing}>取消</Button>
                        <Button 
                          type="primary" 
                          icon={<SaveOutlined />} 
                          onClick={handleConfirmSave}
                        >
                          保存
                        </Button>
                      </>
                    )}
                  </Space>
                }
              >
                {generatedPrompt ? (
                  isEditing ? (
                    <TextArea 
                      value={editedPrompt} 
                      onChange={e => setEditedPrompt(e.target.value)} 
                      autoSize={{ minRows: 10, maxRows: 20 }}
                    />
                  ) : (
                    <div className="generated-prompt-container">
                      <div className="prompt-actions" style={{ marginBottom: '16px' }}>
                        <Button 
                          type="text" 
                          icon={<CopyOutlined />} 
                          onClick={() => copyToClipboard(generatedPrompt)}
                        >
                          复制全文
                        </Button>
                      </div>
                      <div className="prompt-content" style={{ maxHeight: '600px', overflow: 'auto' }}>
                        {formatGeneratedPrompt(generatedPrompt)}
                      </div>
                    </div>
                  )
                ) : (
                  <CustomEmpty description="请先生成Prompt" />
                )}
              </Card>
              
              {/* 设计分析结果展示 */}
              {renderDesignAnalysis()}
            </div>
          </div>
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