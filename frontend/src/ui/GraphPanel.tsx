import { useEffect, useMemo, useState } from 'react';
import { listDatasets, listSeriesForDataset, correlate, persistGraph } from './api.series';
import type { DatasetLite, SeriesLite, CorrelateResp } from '../types';

export default function GraphPanel() {
  const [datasets, setDatasets] = useState<DatasetLite[]>([]);
  const [datasetId, setDatasetId] = useState<string>('');
  const [allSeries, setAllSeries] = useState<SeriesLite[]>([]);
  const [selected, setSelected] = useState<string[]>([]); // by key
  const [method, setMethod] = useState<'spearman' | 'pearson'>('spearman');
  const [pearsonAlso, setPearsonAlso] = useState(true);
  const [minOverlap, setMinOverlap] = useState(6);
  const [edgeMin, setEdgeMin] = useState(0.5);
  const [res, setRes] = useState<CorrelateResp | null>(null);
  const [persistMsg, setPersistMsg] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');

  useEffect(() => {
    listDatasets().then(ds => {
      setDatasets(ds);
      if (ds[0]) setDatasetId(ds[0].id);
    }).catch(e => setErr(e.message ?? String(e)));
  }, []);

  useEffect(() => {
    if (!datasetId) return;
    listSeriesForDataset(datasetId).then(ss => {
      setAllSeries(ss);
      setSelected(ss.slice(0, Math.min(4, ss.length)).map(s => s.key));
    }).catch(e => setErr(e.message ?? String(e)));
  }, [datasetId]);

  const canRun = useMemo(() => selected.length >= 2, [selected]);

  const run = async () => {
    if (!canRun) return;
    setBusy(true); setErr(''); setRes(null); setPersistMsg('');
    try {
      const r = await correlate({ datasetId, series: selected, method, pearsonAlso });
      setRes(r);
    } catch (e: any) { setErr(e.message ?? String(e)); }
    finally { setBusy(false); }
  };

  const persist = async () => {
    if (!canRun) return;
    setBusy(true); setErr(''); setPersistMsg('');
    try {
      const r = await persistGraph({ datasetId, series: selected, method, pearsonAlso, minOverlap, edgeMin });
      setPersistMsg(`Saved run #${r.runId} — edges: ${r.edgesInserted}`);
    } catch (e: any) { setErr(e.message ?? String(e)); }
    finally { setBusy(false); }
  };

  return (
    <div style={{ padding: 12 }}>
      <h3>Correlation & Graph</h3>
      {err && <div style={{ color: '#c33' }}>{err}</div>}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        <div>
          <div>Dataset:&nbsp;
            <select value={datasetId} onChange={e => setDatasetId(e.target.value)}>
              {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          </div>

          <div style={{ marginTop: 8 }}>
            <div style={{ fontWeight: 600 }}>Select series (by key):</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
              {allSeries.map(s => {
                const on = selected.includes(s.key);
                return (
                  <button key={s.id} onClick={() => {
                    setSelected(prev => on ? prev.filter(k => k !== s.key) : [...prev, s.key]);
                  }}
                    style={{ padding: '4px 8px', borderRadius: 8, border: '1px solid #ccc', background: on ? '#eef' : '#fff' }}>
                    {s.key}{on ? ' ✓' : ''}
                  </button>
                );
              })}
            </div>
          </div>

          <div style={{ marginTop: 8 }}>
            Method:&nbsp;
            <select value={method} onChange={e => setMethod(e.target.value as any)}>
              <option value="spearman">Spearman (default)</option>
              <option value="pearson">Pearson</option>
            </select>
            &nbsp; <label><input type="checkbox" checked={pearsonAlso} onChange={e => setPearsonAlso(e.target.checked)} /> also compute Pearson</label>
          </div>

          <div style={{ marginTop: 8 }}>
            Min overlap: <input type="number" value={minOverlap} min={2} onChange={e => setMinOverlap(Number(e.target.value))} style={{ width: 70 }} />
            &nbsp; Edge min (|r|): <input type="number" value={edgeMin} step="0.05" onChange={e => setEdgeMin(Number(e.target.value))} style={{ width: 70 }} />
          </div>

          <div style={{ marginTop: 10 }}>
            <button onClick={run} disabled={!canRun || busy}>Run correlation</button>
            &nbsp;
            <button onClick={persist} disabled={!canRun || busy}>Persist graph</button>
          </div>

          {persistMsg && <div style={{ color: '#2a6', marginTop: 8 }}>{persistMsg}</div>}
        </div>

        <div>
          <div style={{ fontWeight: 600 }}>Pairs</div>
          {!res ? <div>Run to see results…</div> : (
            <table style={{ marginTop: 6, borderCollapse: 'collapse', width: '100%' }}>
              <thead>
                <tr>
                  <th style={{ textAlign: 'left', borderBottom: '1px solid #ddd' }}>Pair</th>
                  <th>n</th>
                  <th>Spearman</th>
                  <th>Pearson</th>
                  <th>Overlap</th>
                </tr>
              </thead>
              <tbody>
                {res.pairs.map(p => (
                  <tr key={`${p.x}-${p.y}`}>
                    <td style={{ borderBottom: '1px solid #eee' }}>{p.x} ↔ {p.y}</td>
                    <td style={{ textAlign: 'center' }}>{p.n}</td>
                    <td style={{ textAlign: 'center' }}>{p.spearman == null ? '—' : p.spearman.toFixed(3)}</td>
                    <td style={{ textAlign: 'center' }}>{p.pearson == null ? '—' : p.pearson.toFixed(3)}</td>
                    <td style={{ textAlign: 'center' }}>{p.overlapFrom} → {p.overlapTo}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
