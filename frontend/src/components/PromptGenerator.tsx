import React, { useState } from 'react';
import { Button, Card, Input, Upload, message, Tabs, Switch, Tooltip } from 'antd';
import { UploadOutlined, InfoCircleOutlined } from '@ant-design/icons';
import type { UploadFile } from 'antd/es/upload/interface';
import { API_BASE_URL } from '../config';

const { TextArea } = Input;
const { TabPane } = Tabs;

// 支持的文件类型
const SUPPORTED_EXTENSIONS = [
    '.txt', '.md', '.markdown',
    '.py', '.js', '.jsx', '.ts', '.tsx',
    '.json', '.yaml', '.yml',
    '.zip', '.rar', '.7z'
];

const PromptGenerator: React.FC = () => {
    const [prompt, setPrompt] = useState('');
    const [template, setTemplate] = useState('');
    const [optimizedPrompt, setOptimizedPrompt] = useState('');
    const [fileList, setFileList] = useState<UploadFile[]>([]);
    const [loading, setLoading] = useState(false);
    const [isDirectory, setIsDirectory] = useState(false);

    const handleGenerateTemplate = async () => {
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
                setTemplate(data.template);
                message.success('模板生成成功！');
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
                setOptimizedPrompt(data.optimized_prompt);
                message.success('优化成功！');
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
                message.success(`${file.name} 上传成功！`);
                if (data.processing_result) {
                    message.info(
                        `处理结果：${data.processing_result.chunks || 0} 个文本块，` +
                        `类型：${data.processing_result.file_type}`
                    );
                }
            } else {
                message.error(`${file.name} 上传失败！${data.detail}`);
            }
        } catch (error) {
            message.error(`${file.name} 上传失败：${error}`);
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

    return (
        <div style={{ padding: '20px' }}>
            <Card title="Prompt生成优化系统" style={{ width: '100%' }}>
                <Tabs defaultActiveKey="1">
                    <TabPane tab="生成模板" key="1">
                        <div style={{ marginBottom: 16 }}>
                            <div style={{ marginBottom: 8 }}>
                                <Switch
                                    checked={isDirectory}
                                    onChange={setIsDirectory}
                                    style={{ marginRight: 8 }}
                                />
                                <span>作为目录处理</span>
                                <Tooltip title="开启后将上传的压缩包作为目录处理">
                                    <InfoCircleOutlined style={{ marginLeft: 8 }} />
                                </Tooltip>
                            </div>
                            <Upload
                                fileList={fileList}
                                onChange={({ fileList }) => setFileList(fileList)}
                                customRequest={({ file, onSuccess }) => {
                                    handleUpload(file as UploadFile);
                                    onSuccess?.('ok');
                                }}
                                beforeUpload={beforeUpload}
                            >
                                <Button icon={<UploadOutlined />}>上传上下文文件</Button>
                            </Upload>
                            <div style={{ marginTop: 8, color: '#666' }}>
                                支持的文件类型：{SUPPORTED_EXTENSIONS.join(', ')}
                            </div>
                        </div>
                        <Button
                            type="primary"
                            onClick={handleGenerateTemplate}
                            loading={loading}
                            style={{ marginBottom: 16 }}
                        >
                            生成模板
                        </Button>
                        <TextArea
                            value={template}
                            onChange={e => setTemplate(e.target.value)}
                            placeholder="生成的模板将显示在这里"
                            autoSize={{ minRows: 6 }}
                            readOnly
                        />
                    </TabPane>
                    <TabPane tab="优化Prompt" key="2">
                        <div style={{ marginBottom: 16 }}>
                            <div style={{ marginBottom: 8 }}>
                                <Switch
                                    checked={isDirectory}
                                    onChange={setIsDirectory}
                                    style={{ marginRight: 8 }}
                                />
                                <span>作为目录处理</span>
                                <Tooltip title="开启后将上传的压缩包作为目录处理">
                                    <InfoCircleOutlined style={{ marginLeft: 8 }} />
                                </Tooltip>
                            </div>
                            <Upload
                                fileList={fileList}
                                onChange={({ fileList }) => setFileList(fileList)}
                                customRequest={({ file, onSuccess }) => {
                                    handleUpload(file as UploadFile);
                                    onSuccess?.('ok');
                                }}
                                beforeUpload={beforeUpload}
                            >
                                <Button icon={<UploadOutlined />}>上传上下文文件</Button>
                            </Upload>
                            <div style={{ marginTop: 8, color: '#666' }}>
                                支持的文件类型：{SUPPORTED_EXTENSIONS.join(', ')}
                            </div>
                        </div>
                        <TextArea
                            value={prompt}
                            onChange={e => setPrompt(e.target.value)}
                            placeholder="请输入需要优化的prompt"
                            autoSize={{ minRows: 4 }}
                            style={{ marginBottom: 16 }}
                        />
                        <Button
                            type="primary"
                            onClick={handleOptimizePrompt}
                            loading={loading}
                            style={{ marginBottom: 16 }}
                        >
                            优化Prompt
                        </Button>
                        <TextArea
                            value={optimizedPrompt}
                            onChange={e => setOptimizedPrompt(e.target.value)}
                            placeholder="优化后的prompt将显示在这里"
                            autoSize={{ minRows: 4 }}
                            readOnly
                        />
                    </TabPane>
                </Tabs>
            </Card>
        </div>
    );
};

export default PromptGenerator; 