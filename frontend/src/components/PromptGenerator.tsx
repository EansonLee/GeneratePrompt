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
    const [vectorDbStatus, setVectorDbStatus] = useState<'ready' | 'initializing' | 'error' | 'partial'>('initializing');
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

    // 优化结果状态
    const [optimizationResult, setOptimizationResult] = useState<any>(null);

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

    const handleGenerateTemplate = async () => {
        if (vectorDbStatus !== 'ready' && vectorDbStatus !== 'partial') {
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
                    project_type: null,
                    project_description: null,
                    context_files: []
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || '生成模板失败');
            }

            const data = await response.json();
            // 设置临时模板并打开确认对话框
            setTempTemplate(data.template);
            setShowTemplateConfirm(true);
            message.success('模板生成成功，请确认或编辑后保存');
        } catch (error) {
            console.error('生成模板失败:', error);
            message.error(error instanceof Error ? error.message : '生成模板失败');
        } finally {
            setGeneratingTemplate(false);
        }
    };

    const handleOptimizePrompt = async () => {
        if (vectorDbStatus !== 'ready' && vectorDbStatus !== 'partial') {
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
                    prompt: prompt,
                    context_files: fileList.map(file => file.name)
                })
            });

            const data = await response.json();
            
            if (!response.ok) {
                const errorMessage = data.detail?.message || '优化prompt失败';
                throw new Error(errorMessage);
            }

            if (data.status === 'success') {
                // 保存优化结果
                setOptimizationResult(data);
                // 设置临时优化后的prompt并打开确认对话框
                setTempOptimizedPrompt(data.optimized_prompt);
                setShowPromptConfirm(true);
                message.success('Prompt优化成功，请确认或编辑后保存');
                
                // 如果有评估结果，显示评估信息
                if (data.evaluation) {
                    const { scores } = data.evaluation;
                    message.info(
                        `优化评分：清晰度 ${scores.clarity}, 完整性 ${scores.completeness}, ` +
                        `相关性 ${scores.relevance}, 一致性 ${scores.consistency}, ` +
                        `结构性 ${scores.structure}`
                    );
                }
            } else {
                throw new Error(data.message || '优化失败');
            }
        } catch (error) {
            console.error('优化prompt失败:', error);
            if (error instanceof Error) {
                message.error(error.message);
            } else {
                message.error('优化prompt失败，请重试');
            }
        } finally {
            setOptimizingPrompt(false);
        }
    };

    const handleUpload = async (file: UploadFile) => {
        if (vectorDbStatus !== 'ready' && vectorDbStatus !== 'partial') {
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
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || '上传失败');
            }
            
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
                message.error(`${file.name} 上传失败！${data.detail || '未知错误'}`);
            }
        } catch (error) {
            console.error('上传文件失败:', error);
            setProcessingStatus(prev => ({ 
                ...prev, 
                status: 'error',
                message: error instanceof Error ? error.message : '上传失败'
            }));
            message.error(error instanceof Error ? error.message : '上传失败，请稍后重试');
        } finally {
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
            message.error(`不支持的文件类型：${extension}，支持的类型：${SUPPORTED_EXTENSIONS.join(', ')}`);
            return false;
        }
        
        // 检查文件大小 (10MB)
        const isLessThan10M = (file as any).size / 1024 / 1024 < 10;
        if (!isLessThan10M) {
            message.error('文件大小不能超过10MB');
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
            setFullScreenLoading(true);
            setLoadingMessage('正在保存优化后的Prompt，请稍候...');
            setSavingPrompt(true);
            setShowPromptConfirm(false);
            
            const response = await fetch(`${API_BASE_URL}/api/confirm-prompt`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    optimized_prompt: tempOptimizedPrompt,
                    optimization_result: optimizationResult
                }),
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                setOptimizedPrompt(tempOptimizedPrompt);
                message.success('优化后的Prompt已确认并保存到向量数据库！');
            } else if (data.status === 'partial_success') {
                setOptimizedPrompt(tempOptimizedPrompt);
                message.warning('Prompt已保存，但使用了备选方法：' + data.message);
            } else if (data.status === 'warning') {
                setOptimizedPrompt(tempOptimizedPrompt);
                message.warning('Prompt已保存，但可能存在问题：' + data.message);
            } else {
                throw new Error(data.message || '保存失败');
            }
        } catch (error) {
            console.error('保存失败:', error);
            message.error(error instanceof Error ? error.message : '保存失败，请重试');
        } finally {
            setSavingPrompt(false);
            setFullScreenLoading(false);
            setOptimizationResult(null);
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
                    <div>向量数据库状态: {vectorDbStatus === 'ready' ? '就绪' : vectorDbStatus === 'initializing' ? '初始化中' : vectorDbStatus === 'partial' ? '部分初始化' : '错误'}</div>
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
            
            {vectorDbStatus !== 'ready' && vectorDbStatus !== 'partial' && (
                <Alert
                    message={vectorDbStatus === 'initializing' ? '向量数据库初始化中...' : vectorDbStatus === 'error' ? '向量数据库错误' : '向量数据库部分初始化'}
                    description={vectorDbError || '请等待向量数据库初始化完成'}
                    type={vectorDbStatus === 'initializing' ? 'info' : vectorDbStatus === 'error' ? 'error' : 'warning'}
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
                            vectorDbStatus !== 'ready' && vectorDbStatus !== 'partial' || 
                            processingStatus.status === 'processing' || 
                            isUploading || 
                            fullScreenLoading
                        }
                    >
                        生成模板
                    </Button>
                    {template && (
                        <>
                            <TextArea
                                value={template}
                                onChange={e => setTemplate(e.target.value)}
                                autoSize={{ minRows: 4, maxRows: 8 }}
                            />
                            <Button 
                                type="primary"
                                onClick={() => {
                                    setTempTemplate(template);
                                    setShowTemplateConfirm(true);
                                }}
                            >
                                保存模板
                            </Button>
                        </>
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
                            vectorDbStatus !== 'ready' && vectorDbStatus !== 'partial' || 
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
                            vectorDbStatus !== 'ready' && vectorDbStatus !== 'partial' || 
                            !prompt || 
                            processingStatus.status === 'processing' || 
                            isUploading || 
                            generatingTemplate || 
                            fullScreenLoading
                        }
                    >
                        优化Prompt
                    </Button>
                    {optimizedPrompt && (
                        <>
                            <TextArea
                                value={optimizedPrompt}
                                onChange={e => setOptimizedPrompt(e.target.value)}
                                autoSize={{ minRows: 4, maxRows: 8 }}
                            />
                            <Button 
                                type="primary"
                                onClick={() => {
                                    setTempOptimizedPrompt(optimizedPrompt);
                                    setShowPromptConfirm(true);
                                }}
                            >
                                保存优化后的Prompt
                            </Button>
                        </>
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