import React, { useState, useEffect } from 'react';
import { 
  Button, Card, Input, Upload, message, Select, Slider, Tabs, 
  Space, Alert, Spin, Typography, Radio, Image, Modal, Form, Divider, Collapse 
} from 'antd';
import { UploadOutlined, LoadingOutlined, SaveOutlined, EditOutlined } from '@ant-design/icons';
import type { UploadFile, UploadProps } from 'antd/es/upload/interface';
import { API_BASE_URL } from '../config';

const { TextArea } = Input;
const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
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
    if (!uploadResult) {
      message.error('请先上传设计图');
      return;
    }
    
    try {
      setGenerationStatus({ status: 'processing' });
      setFullScreenLoading(true);
      setLoadingMessage('正在生成Prompt...');
      
      const requestData = {
        tech_stack: techStack,
        design_image_id: uploadResult.image_id,
        design_image_path: uploadResult.image_path,
        rag_method: ragMethod,
        retriever_top_k: retrieverTopK,
        agent_type: agentType,
        temperature: temperature,
        context_window_size: contextWindowSize,
        prompt: "设计图Prompt生成请求"
      };
      
      console.log('发送请求数据:', JSON.stringify(requestData, null, 2));
      
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
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('请求失败:', response.status, response.statusText, errorText);
        throw new Error(`生成失败: ${response.status} ${response.statusText}`);
      }
      
      const result = await response.json();
      console.log('接收到响应:', JSON.stringify(result, null, 2));
      setGeneratedPrompt(result.generated_prompt || result.prompt);
      setEditedPrompt(result.generated_prompt || result.prompt);
      setHasHistoryContext(result.has_history_context);
      setHistoryPrompts(result.history_prompts || []);
      setGenerationStatus({ status: 'completed' });
      message.success('Prompt生成成功');
      
    } catch (error) {
      console.error('生成Prompt失败:', error);
      setGenerationStatus({ 
        status: 'error', 
        message: error instanceof Error ? error.message : '生成失败' 
      });
      message.error(`生成Prompt失败: ${error instanceof Error ? error.message : '未知错误'}`);
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
      
      const result = await response.json();
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
        
        {generationStatus.status === 'error' && (
          <Alert 
            message="生成失败" 
            description={generationStatus.message} 
            type="error" 
            showIcon 
            style={{ marginTop: '10px' }}
          />
        )}
      </Card>
      
      <Card 
        title="步骤3: 查看和编辑生成的Prompt" 
        style={{ marginBottom: '20px' }}
        extra={
          generatedPrompt ? (
            <Space>
              {isEditing ? (
                <>
                  <Button onClick={handleCancelEditing}>取消</Button>
                  <Button type="primary" onClick={handleConfirmSave}>保存</Button>
                </>
              ) : (
                <Button 
                  type="primary" 
                  icon={<EditOutlined />} 
                  onClick={handleStartEditing}
                >
                  编辑
                </Button>
              )}
            </Space>
          ) : null
        }
      >
        {generationStatus.status === 'completed' && (
          <Alert
            message={
              hasHistoryContext 
                ? `已使用 ${historyPrompts.length} 个历史Prompt作为上下文` 
                : "没有找到相关的历史Prompt，使用默认上下文生成"
            }
            type={hasHistoryContext ? "success" : "info"}
            showIcon
            style={{ marginBottom: '16px' }}
          />
        )}
        
        {generatedPrompt ? (
          isEditing ? (
            <TextArea 
              value={editedPrompt} 
              onChange={e => setEditedPrompt(e.target.value)} 
              autoSize={{ minRows: 10, maxRows: 20 }}
            />
          ) : (
            <div style={{ whiteSpace: 'pre-wrap' }}>
              {generatedPrompt}
            </div>
          )
        ) : (
          <Empty description="暂无生成的Prompt" />
        )}
        
        {hasHistoryContext && historyPrompts.length > 0 && !isEditing && (
          <div style={{ marginTop: '20px' }}>
            <Divider orientation="left">参考的历史Prompt</Divider>
            <Collapse>
              {historyPrompts.map((prompt, index) => {
                const metadata = prompt.metadata || {};
                return (
                  <Collapse.Panel 
                    key={index} 
                    header={`历史Prompt ${index + 1} (${metadata.tech_stack || '未知技术栈'})`}
                  >
                    <div style={{ whiteSpace: 'pre-wrap' }}>
                      {prompt.text || '无内容'}
                    </div>
                    <div style={{ marginTop: '10px', color: '#888' }}>
                      <Text type="secondary">
                        创建时间: {metadata.created_at || '未知'}
                        {metadata.user_modified && ' | 用户修改: 是'}
                      </Text>
                    </div>
                  </Collapse.Panel>
                );
              })}
            </Collapse>
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
const Empty: React.FC<{ description: string }> = ({ description }) => (
  <div style={{ 
    display: 'flex', 
    flexDirection: 'column', 
    alignItems: 'center', 
    justifyContent: 'center',
    padding: '40px 0'
  }}>
    <div style={{ fontSize: '72px', color: '#ccc', marginBottom: '20px' }}>?</div>
    <Text type="secondary">{description}</Text>
  </div>
);

export default DesignPromptGenerator; 