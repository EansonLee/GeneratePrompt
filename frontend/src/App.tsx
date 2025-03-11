import React from 'react';
import MainPage from './components/MainPage';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';

const App: React.FC = () => {
  return (
    <ConfigProvider locale={zhCN}>
      <div className="App">
        <MainPage />
      </div>
    </ConfigProvider>
  );
};

export default App;
