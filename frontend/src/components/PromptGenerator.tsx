import React, { useState, useEffect } from 'react';
import { Button, Card, Input, Upload, message, Switch, Modal, Progress, Space, Alert, Spin } from 'antd';
import { UploadOutlined, LoadingOutlined } from '@ant-design/icons';
import type { UploadFile } from 'antd/es/upload/interface';
import { API_BASE_URL } from '../config';

const { TextArea } = Input;

// 支持的文件类型
const SUPPORTED_EXTENSIONS = [
    '.txt', '.md', '.markdown',
    '.py', '.js', '.jsx', '.ts', '.tsx',
    '.json', '.yaml', '.yml',
    '.zip', '.rar', '.7z'
];

interface ProcessingStatus {
    total: number;
    processed: number;
    status: 'idle' | 'processing' | 'completed' | 'error';
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

const PromptGenerator: React.FC = () => {
    // 基础状态
    const [prompt, setPrompt] = useState('');
    const [template, setTemplate] = useState('');
    const [optimizedPrompt, setOptimizedPrompt] = useState('');
    const [fileList, setFileList] = useState<UploadFile[]>([]);
    const [isDirectory, setIsDirectory] = useState(false);
    
    // 加载和处理状态
    const [isUploading, setIsUploading] = useState(false);
    const [processingStatus, setProcessingStatus] = useState<ProcessingStatus>({
        total: 0,
        processed: 0,
        status: 'idle'
    });
    
    // 向量数据库状态
    const [vectorDbStatus, setVectorDbStatus] = useState<'ready' | 'initializing' | 'error'>('initializing');
    const [vectorDbError, setVectorDbError] = useState<string | null>(null);
    
    // 确认对话框状态
    const [showTemplateConfirm, setShowTemplateConfirm] = useState(false);
    const [showPromptConfirm, setShowPromptConfirm] = useState(false);
    const [tempTemplate, setTempTemplate] = useState('');
    const [tempOptimizedPrompt, setTempOptimizedPrompt] = useState('');
    
    // 保存状态
    const [savingTemplate, setSavingTemplate] = useState(false);
    const [savingPrompt, setSavingPrompt] = useState(false);
    
    // 生成状态
    const [generatingTemplate, setGeneratingTemplate] = useState(false);
    const [optimizingPrompt, setOptimizingPrompt] = useState(false);
    
    // 全屏加载状态
    const [fullScreenLoading, setFullScreenLoading] = useState(false);
    const [loadingMessage, setLoadingMessage] = useState('');

    // 检查向量数据库状态
    const checkVectorDbStatus = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/vector-db-status`);
            const data = await response.json();
            
            if (response.ok) {
                setVectorDbStatus(data.status);
                setVectorDbError(data.error);
                return data.status === 'ready';
            } else {
                setVectorDbStatus('error');
                setVectorDbError(data.detail || '检查向量数据库状态失败');
                return false;
            }
        } catch (error) {
            setVectorDbStatus('error');
            setVectorDbError('检查向量数据库状态失败');
            return false;
        }
    };

    // 定期检查向量数据库状态
    useEffect(() => {
        let intervalId: NodeJS.Timeout;
        
        const startStatusCheck = () => {
            // 立即检查一次
            checkVectorDbStatus();
            
            // 如果数据库未就绪，每5秒检查一次
            intervalId = setInterval(async () => {
                const isReady = await checkVectorDbStatus();
                if (isReady) {
                    clearInterval(intervalId);
                }
            }, 5000);
        };
        
        startStatusCheck();
        
        return () => {
            if (intervalId) {
                clearInterval(intervalId);
            }
        };
    }, []);

    const handleGenerateTemplate = async () => {
        if (vectorDbStatus !== 'ready') {
            message.warning('请等待向量数据库初始化完成');
            return;
        }

        try {
            setGeneratingTemplate(true);
            const response = await fetch(`${API_BASE_URL}/api/generate-template`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    context_files: fileList.map(file => file.name),
                }),
            });
            const data = await response.json();
            if (data.status === 'success') {
                setTempTemplate(data.template);
                setShowTemplateConfirm(true);
                message.success('模板生成成功！请确认内容');
            } else {
                message.error('生成失败：' + data.detail);
            }
        } catch (error) {
            message.error('生成失败：' + error);
        } finally {
            setGeneratingTemplate(false);
        }
    };

    const handleOptimizePrompt = async () => {
        if (vectorDbStatus !== 'ready') {
            message.warning('请等待向量数据库初始化完成');
            return;
        }

        if (!prompt) {
            message.warning('请输入需要优化的prompt');
            return;
        }

        try {
            setOptimizingPrompt(true);
            const response = await fetch(`${API_BASE_URL}/api/optimize-prompt`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt,
                    context_files: fileList.map(file => file.name),
                }),
            });
            const data = await response.json();
            if (data.status === 'success') {
                setTempOptimizedPrompt(data.optimized_prompt);
                setShowPromptConfirm(true);
                message.success('优化成功！请确认内容');
            } else {
                message.error('优化失败：' + data.detail);
            }
        } catch (error) {
            message.error('优化失败：' + error);
        } finally {
            setOptimizingPrompt(false);
        }
    };

    const handleUpload = async (file: UploadFile) => {
        if (vectorDbStatus !== 'ready') {
            message.warning('请等待向量数据库初始化完成');
            return;
        }
        
        if (processingStatus.status === 'processing') {
            message.warning('请等待当前文件处理完成');
            return;
        }

        setIsUploading(true);
        setProcessingStatus({
            total: 100,  // 设置初始总进度为100
            processed: 0,
            status: 'processing'
        });

        const formData = new FormData();
        formData.append('file', file as any);
        formData.append('is_directory', isDirectory.toString());

        try {
            const response = await fetch(`${API_BASE_URL}/api/upload-context`, {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            if (data.status === 'success') {
                // 更新文件列表
                setFileList(prev => [...prev, {...file, status: 'done'}]);
                
                // 更新处理状态为完成
                setProcessingStatus({
                    total: 100,
                    processed: 100,
                    status: 'completed'
                });
                
                message.success(`${file.name} 上传成功！`);
                if (data.processing_result) {
                    message.info(
                        `文件处理完成：${data.processing_result.total_chunks || 0} 个文本块，` +
                        `类型：${data.processing_result.file_type}`
                    );
                }
            } else {
                setProcessingStatus(prev => ({ ...prev, status: 'error' }));
                message.error(`${file.name} 上传失败！${data.detail}`);
            }
        } catch (error) {
            setProcessingStatus(prev => ({ ...prev, status: 'error' }));
            message.error(`${file.name} 上传失败：${error}`);
        } finally {
            // 重要：立即重置上传状态
            setIsUploading(false);
            
            // 3秒后清除完成状态
            setTimeout(() => {
                setProcessingStatus({
                    total: 0,
                    processed: 0,
                    status: 'idle'
                });
            }, 3000);
        }
    };

    const handleFileListChange = ({ fileList }: { fileList: UploadFile[] }) => {
        // 过滤掉上传中的文件，只保留上传成功的文件
        const successFiles = fileList.filter(file => file.status === 'done');
        setFileList(successFiles);
    };

    const beforeUpload = (file: UploadFile) => {
        const extension = `.${file.name.split('.').pop()?.toLowerCase()}`;
        if (!SUPPORTED_EXTENSIONS.includes(extension)) {
            message.error(`不支持的文件类型：${extension}`);
            return false;
        }
        return true;
    };

    const handleConfirmTemplate = async () => {
        try {
            // 显示全屏加载
            setFullScreenLoading(true);
            setLoadingMessage('正在保存模板，请稍候...');
            
            // 添加保存状态，解决问题3
            setSavingTemplate(true);
            
            // 关闭确认对话框
            setShowTemplateConfirm(false);
            
            const response = await fetch(`${API_BASE_URL}/api/confirm-template`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    template: tempTemplate,
                }),
            });
            const data = await response.json();
            if (data.status === 'success') {
                setTemplate(tempTemplate);
                message.success('模板已确认并保存到向量数据库！');
            } else {
                message.error('保存失败：' + data.detail);
            }
        } catch (error) {
            message.error('保存失败：' + error);
        } finally {
            // 重置保存状态
            setSavingTemplate(false);
            // 隐藏全屏加载
            setFullScreenLoading(false);
        }
    };

    const handleConfirmPrompt = async () => {
        try {
            // 显示全屏加载
            setFullScreenLoading(true);
            setLoadingMessage('正在保存优化后的Prompt，请稍候...');
            
            // 添加保存状态，解决问题3
            setSavingPrompt(true);
            
            // 关闭确认对话框
            setShowPromptConfirm(false);
            
            const response = await fetch(`${API_BASE_URL}/api/confirm-prompt`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    optimized_prompt: tempOptimizedPrompt,
                }),
            });
            const data = await response.json();
            if (data.status === 'success') {
                setOptimizedPrompt(tempOptimizedPrompt);
                message.success('优化后的Prompt已确认并保存到向量数据库！');
            } else {
                message.error('保存失败：' + data.detail);
            }
        } catch (error) {
            message.error('保存失败：' + error);
        } finally {
            // 重置保存状态
            setSavingPrompt(false);
            // 隐藏全屏加载
            setFullScreenLoading(false);
        }
    };

    const renderProcessingStatus = () => {
        // 修改为：只在处理中或完成时显示进度
        if (processingStatus.status !== 'processing' && processingStatus.status !== 'completed') {
            return null;
        }

        return (
            <Card style={{ marginBottom: 16 }}>
                <Space direction="vertical" style={{ width: '100%' }}>
                    <div>文件处理状态</div>
                    <Progress
                        percent={processingStatus.status === 'completed' ? 100 : Math.round((processingStatus.processed / processingStatus.total) * 100)}
                        status={processingStatus.status === 'completed' ? 'success' : 'active'}
                        format={percent => `${percent}%`}
                    />
                    <div>向量数据库状态: {vectorDbStatus === 'ready' ? '就绪' : vectorDbStatus === 'initializing' ? '初始化中' : '错误'}</div>
                </Space>
            </Card>
        );
    };

    // 全屏加载组件
    const renderFullScreenLoading = () => {
        if (!fullScreenLoading) return null;
        
        return (
            <div style={fullScreenLoadingStyle}>
                <Spin indicator={antIcon} size="large" />
                <div style={{ marginTop: 20, fontSize: 18 }}>{loadingMessage}</div>
            </div>
        );
    };

    return (
        <div style={{ padding: 24 }}>
            {/* 全屏加载组件 */}
            {renderFullScreenLoading()}
            
            {vectorDbStatus !== 'ready' && (
                <Alert
                    message={vectorDbStatus === 'initializing' ? '向量数据库初始化中...' : '向量数据库错误'}
                    description={vectorDbError || '请等待向量数据库初始化完成'}
                    type={vectorDbStatus === 'initializing' ? 'info' : 'error'}
                    showIcon
                    style={{ marginBottom: 16 }}
                />
            )}

            <Card title="上下文文件上传" style={{ marginBottom: 16 }}>
                <Space direction="vertical" style={{ width: '100%' }}>
                    <Switch
                        checked={isDirectory}
                        onChange={setIsDirectory}
                        disabled={isUploading || processingStatus.status === 'processing' || fullScreenLoading}
                        checkedChildren="目录模式"
                        unCheckedChildren="文件模式"
                    />
                    <Upload
                        beforeUpload={beforeUpload}
                        customRequest={({ file }) => handleUpload(file as UploadFile)}
                        fileList={fileList}
                        onChange={handleFileListChange}
                        disabled={isUploading || processingStatus.status === 'processing' || fullScreenLoading}
                    >
                        <Button 
                            icon={<UploadOutlined />}
                            disabled={isUploading || processingStatus.status === 'processing' || fullScreenLoading}
                            loading={isUploading}
                        >
                            选择{isDirectory ? '目录' : '文件'}
                        </Button>
                    </Upload>
                </Space>
            </Card>

            {renderProcessingStatus()}

            <Card title="Prompt模板生成" style={{ marginBottom: 16 }}>
                <Space direction="vertical" style={{ width: '100%' }}>
                    <Button
                        type="primary"
                        onClick={handleGenerateTemplate}
                        loading={generatingTemplate}
                        disabled={
                            vectorDbStatus !== 'ready' || 
                            processingStatus.status === 'processing' || 
                            isUploading || 
                            fullScreenLoading
                        }
                    >
                        生成模板
                    </Button>
                    {template && (
                        <TextArea
                            value={template}
                            autoSize={{ minRows: 4, maxRows: 8 }}
                            readOnly
                        />
                    )}
                </Space>
            </Card>

            <Card title="Prompt优化">
                <Space direction="vertical" style={{ width: '100%' }}>
                    <TextArea
                        placeholder="请输入需要优化的Prompt"
                        value={prompt}
                        onChange={e => setPrompt(e.target.value)}
                        autoSize={{ minRows: 4, maxRows: 6 }}
                        disabled={
                            vectorDbStatus !== 'ready' || 
                            processingStatus.status === 'processing' || 
                            isUploading || 
                            fullScreenLoading
                        }
                    />
                    <Button
                        type="primary"
                        onClick={handleOptimizePrompt}
                        loading={optimizingPrompt}
                        disabled={
                            vectorDbStatus !== 'ready' || 
                            !prompt || 
                            processingStatus.status === 'processing' || 
                            isUploading || 
                            generatingTemplate || // 解决问题2：生成模板时禁用优化按钮
                            fullScreenLoading
                        }
                    >
                        优化Prompt
                    </Button>
                    {optimizedPrompt && (
                        <TextArea
                            value={optimizedPrompt}
                            autoSize={{ minRows: 4, maxRows: 8 }}
                            readOnly
                        />
                    )}
                </Space>
            </Card>

            <Modal
                title="确认模板"
                open={showTemplateConfirm}
                onOk={handleConfirmTemplate}
                onCancel={() => setShowTemplateConfirm(false)}
                width={800}
                confirmLoading={savingTemplate}
                okButtonProps={{ disabled: savingTemplate }}
                cancelButtonProps={{ disabled: savingTemplate }}
                maskClosable={false}
                closable={!savingTemplate}
                keyboard={!savingTemplate}
            >
                <TextArea
                    value={tempTemplate}
                    onChange={e => setTempTemplate(e.target.value)}
                    autoSize={{ minRows: 6, maxRows: 12 }}
                    disabled={savingTemplate}
                />
            </Modal>

            <Modal
                title="确认优化后的Prompt"
                open={showPromptConfirm}
                onOk={handleConfirmPrompt}
                onCancel={() => setShowPromptConfirm(false)}
                width={800}
                confirmLoading={savingPrompt}
                okButtonProps={{ disabled: savingPrompt }}
                cancelButtonProps={{ disabled: savingPrompt }}
                maskClosable={false}
                closable={!savingPrompt}
                keyboard={!savingPrompt}
            >
                <TextArea
                    value={tempOptimizedPrompt}
                    onChange={e => setTempOptimizedPrompt(e.target.value)}
                    autoSize={{ minRows: 6, maxRows: 12 }}
                    disabled={savingPrompt}
                />
            </Modal>
        </div>
    );
};

export default PromptGenerator; 