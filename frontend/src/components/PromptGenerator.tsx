import React, { useState, useEffect } from 'react';
import { Button, Card, Input, Upload, message, Switch, Modal, Progress, Space, Alert } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
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


const PromptGenerator: React.FC = () => {
    // 基础状态
    const [prompt, setPrompt] = useState('');
    const [template, setTemplate] = useState('');
    const [optimizedPrompt, setOptimizedPrompt] = useState('');
    const [fileList, setFileList] = useState<UploadFile[]>([]);
    const [isDirectory, setIsDirectory] = useState(false);
    
    // 加载和处理状态
    const [loading, setLoading] = useState(false);
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
            setLoading(true);
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
            setLoading(false);
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
            setLoading(true);
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
            setLoading(false);
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
                setFileList(prev => [...prev, file]);
                
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
            setIsUploading(false);
            // 3秒后清除完成状态
            if (processingStatus.status === 'completed') {
                setTimeout(() => {
                    setProcessingStatus({
                        total: 0,
                        processed: 0,
                        status: 'idle'
                    });
                }, 3000);
            }
        }
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
                setShowTemplateConfirm(false);
                message.success('模板已确认并保存到向量数据库！');
            } else {
                message.error('保存失败：' + data.detail);
            }
        } catch (error) {
            message.error('保存失败：' + error);
        }
    };

    const handleConfirmPrompt = async () => {
        try {
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
                setShowPromptConfirm(false);
                message.success('优化后的Prompt已确认并保存到向量数据库！');
            } else {
                message.error('保存失败：' + data.detail);
            }
        } catch (error) {
            message.error('保存失败：' + error);
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

    return (
        <div style={{ padding: 24 }}>
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
                        disabled={isUploading || processingStatus.status === 'processing'}
                        checkedChildren="目录模式"
                        unCheckedChildren="文件模式"
                    />
                    <Upload
                        beforeUpload={beforeUpload}
                        customRequest={({ file }) => handleUpload(file as UploadFile)}
                        fileList={fileList}
                        onChange={({ fileList }) => setFileList(fileList)}
                        disabled={isUploading || processingStatus.status === 'processing'}
                    >
                        <Button 
                            icon={<UploadOutlined />}
                            disabled={isUploading || processingStatus.status === 'processing'}
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
                        loading={loading}
                        disabled={vectorDbStatus !== 'ready' || processingStatus.status === 'processing'}
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
                        disabled={vectorDbStatus !== 'ready' || processingStatus.status === 'processing'}
                    />
                    <Button
                        type="primary"
                        onClick={handleOptimizePrompt}
                        loading={loading}
                        disabled={vectorDbStatus !== 'ready' || !prompt || processingStatus.status === 'processing'}
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
            >
                <TextArea
                    value={tempTemplate}
                    onChange={e => setTempTemplate(e.target.value)}
                    autoSize={{ minRows: 6, maxRows: 12 }}
                />
            </Modal>

            <Modal
                title="确认优化后的Prompt"
                open={showPromptConfirm}
                onOk={handleConfirmPrompt}
                onCancel={() => setShowPromptConfirm(false)}
                width={800}
            >
                <TextArea
                    value={tempOptimizedPrompt}
                    onChange={e => setTempOptimizedPrompt(e.target.value)}
                    autoSize={{ minRows: 6, maxRows: 12 }}
                />
            </Modal>
        </div>
    );
};

export default PromptGenerator; 