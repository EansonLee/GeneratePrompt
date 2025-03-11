import React, { useState } from 'react';
import { Layout, Menu, Typography, Divider } from 'antd';
import { 
  FileTextOutlined, 
  AppstoreOutlined, 
  SettingOutlined,
  MobileOutlined
} from '@ant-design/icons';
import PromptGenerator from './PromptGenerator';
import DesignPromptGenerator from './DesignPromptGenerator';

const { Header, Content, Sider } = Layout;
const { Title } = Typography;

// 功能页面枚举
enum FunctionPage {
  PROMPT_GENERATOR = 'prompt_generator',
  DESIGN_PROMPT_GENERATOR = 'design_prompt_generator'
}

const MainPage: React.FC = () => {
  // 当前选中的功能页面
  const [currentPage, setCurrentPage] = useState<FunctionPage>(FunctionPage.PROMPT_GENERATOR);
  
  // 菜单项点击处理
  const handleMenuClick = (key: string) => {
    setCurrentPage(key as FunctionPage);
  };
  
  // 渲染当前功能页面
  const renderCurrentPage = () => {
    switch (currentPage) {
      case FunctionPage.PROMPT_GENERATOR:
        return <PromptGenerator />;
      case FunctionPage.DESIGN_PROMPT_GENERATOR:
        return <DesignPromptGenerator />;
      default:
        return <PromptGenerator />;
    }
  };
  
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ 
        display: 'flex', 
        alignItems: 'center', 
        padding: '0 20px',
        background: '#fff',
        borderBottom: '1px solid #f0f0f0'
      }}>
        <Title level={3} style={{ margin: 0 }}>Prompt生成器</Title>
      </Header>
      
      <Layout>
        <Sider width={200} theme="light" breakpoint="lg" collapsedWidth={0}>
          <Menu
            mode="inline"
            selectedKeys={[currentPage]}
            style={{ height: '100%', borderRight: 0 }}
            onClick={({ key }) => handleMenuClick(key)}
          >
            <Menu.Item key={FunctionPage.PROMPT_GENERATOR} icon={<FileTextOutlined />}>
              提示词生成
            </Menu.Item>
            <Menu.Item key={FunctionPage.DESIGN_PROMPT_GENERATOR} icon={<MobileOutlined />}>
              设计图Prompt生成
            </Menu.Item>
          </Menu>
        </Sider>
        
        <Content style={{ 
          background: '#fff', 
          padding: 0, 
          margin: 0, 
          minHeight: 280 
        }}>
          {renderCurrentPage()}
        </Content>
      </Layout>
    </Layout>
  );
};

export default MainPage; 