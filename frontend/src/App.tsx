import React from 'react';
import PromptGenerator from './components/PromptGenerator';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';

const App: React.FC = () => {
  return (
    <ConfigProvider locale={zhCN}>
      <div className="App">
        <PromptGenerator />
      </div>
    </ConfigProvider>
  );
};

export default App;
