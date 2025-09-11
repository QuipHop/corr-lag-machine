import React, { useEffect, useMemo, useState } from 'react';
import { fetchSeriesData, fetchSeriesMeta } from '../api';
import type { Point, SeriesMeta } from '../types';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts';

function formatDate(iso: string) {
  return new Date(iso).toISOString().slice(0, 10);
}

export default function SeriesPanel({ presets, onUpdatePresets }: { presets: number[]; onUpdatePresets: (ids: number[]) => void }) {
  const [idInput, setIdInput] = useState<string>(String(presets[0] ?? ''));
  const [meta, setMeta] = useState<SeriesMeta | null>(null);
  const [data, setData] = useState<Point[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);


  async function load(id: number) {
    setError(null); setLoading(true);
    try {
      const m = await fetchSeriesMeta(id);
      setMeta(m);
      const d = await fetchSeriesData(id);
      const cooked = d.points.map(p => ({ ...p, date: formatDate(p.date) }));
      setData(cooked);
      if (!presets.includes(id)) onUpdatePresets([...presets, id]);
    } catch (e: any) {
      setMeta(null); setData([]);
      setError(e?.message ?? 'Failed to load');
    } finally {
      setLoading(false);
    }
  }


  useEffect(() => {
    if (presets.length) load(presets[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);


  const unit = meta?.indicator?.unit ?? '';
  const title = meta ? `${meta.indicator.code}${meta.region ? ' — ' + meta.region : ''}` : 'Series';

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <input value={idInput} onChange={e => setIdInput(e.target.value)} placeholder="Series ID" />
        <button onClick={() => { const n = Number(idInput); if (!Number.isNaN(n)) load(n); }}>Load</button>
      </div>
      {loading && <div className="muted">Loading…</div>}
      {error && <div style={{ color: '#ffa7a7' }}>{error}</div>}
      {meta && (
        <div style={{ marginBottom: 12 }}>
          <div className="pill">{meta.frequency}</div> <strong>{title}</strong> <span className="muted">[{unit}]</span>
        </div>
      )}
      <div style={{ height: 300 }} className="card">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 12 }} interval={'preserveStartEnd'} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip />
            <Line type="monotone" dataKey="value" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}