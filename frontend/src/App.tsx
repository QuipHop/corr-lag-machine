import { useState } from 'react';
import SeriesPanel from './ui/SeriesPanel';
import SeriesBrowser from './ui/SeriesBrowser';
import GraphPanel from './ui/GraphPanel';
import CorrLagPanel from './ui/CorrLagPanel';
import './App.css';

type Tab = 'upload' | 'browse' | 'graph' | 'lag';

export default function App() {
  const [tab, setTab] = useState<Tab>('upload');
  console.log("SSS");
  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 12 }}>
      <h2>Macro Correlations v1</h2>
      <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
        <button onClick={() => setTab('upload')} style={{ fontWeight: tab === 'upload' ? 700 : 400 }}>Upload</button>
        <button onClick={() => setTab('browse')} style={{ fontWeight: tab === 'browse' ? 700 : 400 }}>Browse</button>
        <button onClick={() => setTab('graph')} style={{ fontWeight: tab === 'graph' ? 700 : 400 }}>Graph</button>
        <button onClick={() => setTab('lag')} style={{ fontWeight: tab === 'lag' ? 700 : 400 }}>Lag</button>
      </div>

      {tab === 'upload' && <SeriesPanel />}
      {tab === 'browse' && <SeriesBrowser />}
      {tab === 'graph' && <GraphPanel />}
      {tab === 'lag' && <CorrLagPanel />}
    </div>
  );
}
