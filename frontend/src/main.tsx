import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Overview } from './pages/Overview';
import { MarketData } from './pages/MarketData';
import HistoricalData from './pages/HistoricalData';
import { Strategies } from './pages/Strategies';
import { Backtests } from './pages/Backtests';
import { Optimizer } from './pages/Optimizer';
import './index.css';

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/market" element={<MarketData />} />
          <Route path="/historical" element={<HistoricalData />} />
          <Route path="/strategies" element={<Strategies />} />
          <Route path="/backtests" element={<Backtests />} />
          <Route path="/optimizer" element={<Optimizer />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
