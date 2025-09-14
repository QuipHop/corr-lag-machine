import { useEffect, useMemo, useState } from 'react';
import { listDatasets, listSeriesForDataset, correlate, persistGraph, listRuns, getRun } from './api.series';
import type { DatasetLite, SeriesLite, CorrelateResp, RunLite, RunDetail } from '../types';

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

  // NEW: history state
  const [runs, setRuns] = useState<RunLite[]>([]);
  const [runDetail, setRunDetail] = useState<RunDetail | null>(null);

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

    // NEW: load history for dataset
    listRuns(datasetId).then(rs => setRuns(rs)).catch(e => setErr(e.message ?? String(e)));
    setRunDetail(null);
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
      // refresh history
      const rs = await listRuns(datasetId);
      setRuns(rs);
    } catch (e: any) { setErr(e.message ?? String(e)); }
    finally { setBusy(false); }
  };

  // NEW: load a run detail
  const openRun = async (id: number) => {
    setBusy(true); setErr(''); setRunDetail(null);
    try {
      const d = await getRun(id);
      setRunDetail(d);
    } catch (e: any) { setErr(e.message ?? String(e)); }
    finally { setBusy(false); }
  };

  return (
    <div style={{ padding: 12 }}>
      <h3>Correlation & Graph</h3>
      {err && <div style={{ color: '#c33' }}>{err}</div>}

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 12 }}>
        {/* Left column: controls + live result */}
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
                      style={{ color: "black", padding: '4px 8px', borderRadius: 8, border: '1px solid #ccc', background: on ? '#eef' : '#fff' }}>
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

        {/* Right column: history */}
        <div>
          <div style={{ fontWeight: 600 }}>History (dataset)</div>
          {!runs.length ? <div>No saved runs yet.</div> : (
            <ul style={{ listStyle: 'none', padding: 0, marginTop: 6 }}>
              {runs.map(r => (
                <li key={r.id} style={{ marginBottom: 6 }}>
                  <button onClick={() => openRun(r.id)} style={{ width: '100%', textAlign: 'left', padding: '6px 8px', border: '1px solid #ddd', borderRadius: 8 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Run #{r.id} • {new Date(r.createdAt).toLocaleString()}</span>
                      <span>{r.method} • edges {r.edgeCount}</span>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}

          {runDetail && (
            <div style={{ marginTop: 10 }}>
              <div style={{ fontWeight: 600 }}>Run #{runDetail.id} details</div>
              <div style={{ fontSize: 12, color: '#666' }}>
                {runDetail.method} • minOverlap {runDetail.minOverlap} • |r|≥{runDetail.edgeMin}
              </div>
              <table style={{ marginTop: 6, borderCollapse: 'collapse', width: '100%' }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: 'left', borderBottom: '1px solid #ddd' }}>Edge</th>
                    <th>Lag</th>
                    <th>Weight</th>
                  </tr>
                </thead>
                <tbody>
                  {runDetail.edges.map(e => (
                    <tr key={`${e.sourceId}-${e.targetId}-${e.lag}`}>
                      <td style={{ borderBottom: '1px solid #eee' }}>{e.sourceKey} → {e.targetKey}</td>
                      <td style={{ textAlign: 'center' }}>{e.lag}</td>
                      <td style={{ textAlign: 'center' }}>{e.weight.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
