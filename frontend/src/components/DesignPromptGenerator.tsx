import React, { useState } from 'react';
import { 
  Button, Card, Input, Upload, message, Select, Slider, 
  Space, Alert, Spin, Typography, Modal, Form, Divider, Collapse, Empty 
} from 'antd';
import { UploadOutlined, LoadingOutlined, SaveOutlined, EditOutlined, CopyOutlined, CloseOutlined } from '@ant-design/icons';
import type { UploadFile, UploadProps } from 'antd/es/upload/interface';
import { API_BASE_URL } from '../config';
import ReactMarkdown from 'react-markdown';

const { TextArea } = Input;
const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

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
      setGeneratedPrompt('');
      
      const startTime = Date.now();
      
      // 准备请求数据
      const requestData = {
        tech_stack: techStack,
        design_image_id: uploadResult.image_id,
        design_image_path: uploadResult.file_path,
        rag_method: ragMethod,
        retriever_top_k: retrieverTopK,
        agent_type: agentType,
        temperature: temperature,
        context_window_size: contextWindowSize,
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
      
      // 如果状态已经设置为错误，则不再更新
      if (generationStatus.status !== 'error') {
        setGenerationStatus({ 
          status: 'error', 
          message: error instanceof Error ? error.message : '生成失败' 
        });
        message.error(`生成Prompt失败: ${error instanceof Error ? error.message : '未知错误'}`);
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
            <Button 
              type="primary" 
              onClick={handleGeneratePrompt}
              loading={generationStatus.status === 'processing'}
              disabled={!uploadResult}
            >
              生成Prompt
            </Button>
          </Form.Item>
        </Form>
        
        {renderGenerationStatus()}
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
                          onClick={() => {
                            navigator.clipboard.writeText(generatedPrompt);
                            message.success('已复制到剪贴板');
                          }}
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