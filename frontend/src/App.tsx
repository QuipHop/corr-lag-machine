import React, { useState } from 'react';
import SeriesPanel from './ui/SeriesPanel';
import CorrLagPanel from './ui/CorrLagPanel';


export default function App() {
  const [knownIds, setKnownIds] = useState<number[]>([0, 1]); // your current two
  return (
    <div className="container">
      <h1 style={{ marginBottom: 8 }}>Cascade Correlations — MVP</h1>
      <div className="muted" style={{ marginBottom: 16 }}>API base: {import.meta.env.VITE_API_BASE ?? 'http://localhost:3000'}</div>


      <div className="row">
        <div className="card" style={{ flex: 1, minWidth: 320 }}>
          <h2>Series quick access</h2>
          <p className="muted">Enter an ID and view the time series. Presets: {knownIds.join(', ')}.</p>
          <SeriesPanel presets={knownIds} onUpdatePresets={setKnownIds} />
        </div>


        <div className="card" style={{ flex: 1, minWidth: 420 }}>
          <h2>Correlation (lagged)</h2>
          <CorrLagPanel defaultIds={knownIds} />
        </div>
      </div>


      <div style={{ marginTop: 24 }} className="muted">Tip: you can change presets any time — useful once you add real datasets.</div>
    </div>
  );
}