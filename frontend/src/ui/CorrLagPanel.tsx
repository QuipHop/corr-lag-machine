import React, { useState } from 'react';
import { type CorrLagEdge } from '../types';
import GraphPanel from './GraphPanel';
import { corrLag } from '../api';

export default function CorrLagPanel({ defaultIds }: { defaultIds: number[] }) {
  const [ids, setIds] = useState<string>(defaultIds.join(', '));
  const [method, setMethod] = useState<'pearson' | 'spearman'>('spearman');
  const [maxLag, setMaxLag] = useState<number>(12);
  const [minOverlap, setMinOverlap] = useState<number>(5);
  const [edgeMin, setEdgeMin] = useState<number>(0.3);
  const [edges, setEdges] = useState<CorrLagEdge[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);


  async function run() {
    const parsed = ids.split(',').map(s => Number(s.trim())).filter(n => Number.isFinite(n));
    if (!parsed.length) { setError('Please enter at least one series id'); return; }
    setError(null); setLoading(true);
    try {
      const res = await corrLag(parsed, { method, maxLag, minOverlap, edgeMin });
      setEdges(res.edges.sort((a: any, b: any) => Math.abs(b.weight) - Math.abs(a.weight)));
    } catch (e: any) {
      setError(e?.message ?? 'Failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        <div>
          <label className="muted">Series IDs (comma-separated)</label>
          <input value={ids} onChange={e => setIds(e.target.value)} />
        </div>
        <div>
          <label className="muted">Method</label><br />
          <select value={method} onChange={e => setMethod(e.target.value as any)}>
            <option value="pearson">pearson</option>
            <option value="spearman">spearman</option>
          </select>
        </div>
        <div>
          <label className="muted">Max lag</label>
          <input type="number" value={maxLag} onChange={e => setMaxLag(Number(e.target.value))} />
        </div>
        <div>
          <label className="muted">Min overlap</label>
          <input type="number" value={minOverlap} onChange={e => setMinOverlap(Number(e.target.value))} />
        </div>
        <div>
          <label className="muted">Edge min |corr|</label>
          <input type="number" step="0.1" value={edgeMin} onChange={e => setEdgeMin(Number(e.target.value))} />
        </div>
        <div style={{ alignSelf: 'end' }}>
          <button onClick={run} disabled={loading}>{loading ? 'Running…' : 'Run corr-lag'}</button>
        </div>
      </div>


      <div style={{ marginTop: 12 }}>
        {error && <div style={{ color: '#ffa7a7' }}>{error}</div>}
        {!error && edges.length === 0 && <div className="muted">No edges yet. Try lowering Edge min or increasing Min overlap / adding more months.</div>}
        {edges.length > 0 && (
          <div className="card" style={{ marginTop: 12 }}>
            <h3 style={{ marginTop: 0 }}>Edges ({edges.length})</h3>
            <table>
              <thead><tr><th>Source</th><th>Target</th><th>Lag</th><th>Weight</th></tr></thead>
              <tbody>
                {edges.map((e, idx) => (
                  <tr key={idx}>
                    <td>{e.source}</td>
                    <td>{e.target}</td>
                    <td>{e.lag}</td>
                    <td>{e.weight.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
      {edges.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <h3 style={{ marginTop: 0 }}>Graph</h3>
          <GraphPanel edges={edges} />
        </div>
      )}
    </div>
  );
}