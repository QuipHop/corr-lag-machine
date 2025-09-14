import { useEffect, useMemo, useState } from 'react';
import {
  listDatasets, createDataset, uploadFile, previewUpload, commitUpload, getSavedMapping
} from './api.series';
import type { DatasetLite, PreviewReq, PreviewResp, CommitResp } from '../types';

export default function SeriesPanel() {
  const [datasets, setDatasets] = useState<DatasetLite[]>([]);
  const [datasetId, setDatasetId] = useState<string>('');
  const [uploadId, setUploadId] = useState<string>('');
  const [columns, setColumns] = useState<{ name: string; typeGuess: 'date' | 'number'; examples: string[] }[]>([]);
  const [mapping, setMapping] = useState<PreviewReq>({
    dateColumn: '',
    valueColumns: [],
    decimal: 'auto',
    dateFormat: 'YYYY-MM',
    dropBlanks: true,
  });
  const [preview, setPreview] = useState<PreviewResp | null>(null);
  const [commit, setCommit] = useState<CommitResp | null>(null);
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
    // prefill from saved mapping if any
    getSavedMapping(datasetId).then(m => {
      if (!m || !Object.keys(m).length) return;
      const as = m as any;
      setMapping({
        dateColumn: as.dateColumn ?? '',
        valueColumns: Array.isArray(as.valueColumns) ? as.valueColumns : [],
        decimal: as.decimal ?? 'auto',
        dateFormat: as.dateFormat ?? 'YYYY-MM',
        dropBlanks: as.dropBlanks ?? true,
      });
    }).catch(() => { });
  }, [datasetId]);

  const numberColumns = useMemo(() => columns.filter(c => c.typeGuess === 'number'), [columns]);

  const onCreateDataset = async () => {
    setBusy(true); setErr('');
    try {
      const ds = await createDataset({ name: `Dataset ${new Date().toISOString().slice(0, 10)}`, freq: 'monthly' });
      setDatasets(d => [ds, ...d]); setDatasetId(ds.id);
    } catch (e: any) { setErr(e.message ?? String(e)); }
    finally { setBusy(false); }
  };

  const onPickFile = async (f: File) => {
    if (!datasetId) return;
    setBusy(true); setErr(''); setPreview(null); setCommit(null);
    try {
      const up = await uploadFile(datasetId, f);
      setUploadId(up.uploadId);
      setColumns(up.columns);
      const dateGuess = up.columns.find(c => c.typeGuess === 'date')?.name ?? up.columns[0]?.name ?? '';
      const vals = up.columns.filter(c => c.typeGuess === 'number').slice(0, 3);
      setMapping(m => ({ ...m, dateColumn: dateGuess, valueColumns: vals.map(v => ({ name: v.name, key: v.name.toUpperCase() })) }));
    } catch (e: any) { setErr(e.message ?? String(e)); }
    finally { setBusy(false); }
  };

  const onPreview = async () => {
    setBusy(true); setErr('');
    try {
      const p = await previewUpload(uploadId, mapping);
      setPreview(p);
    } catch (e: any) { setErr(e.message ?? String(e)); }
    finally { setBusy(false); }
  };

  const onCommit = async () => {
    setBusy(true); setErr('');
    try {
      const c = await commitUpload(uploadId, { saveMappingToDataset: true, createSeries: true, upsertMode: 'merge' });
      setCommit(c);
    } catch (e: any) { setErr(e.message ?? String(e)); }
    finally { setBusy(false); }
  };

  return (
    <div style={{ padding: 12 }}>
      <h3>Upload → Preview → Commit</h3>
      {err && <div style={{ color: '#c33' }}>{err}</div>}

      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <label>Dataset:&nbsp;</label>
        <select value={datasetId} onChange={e => setDatasetId(e.target.value)}>
          {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
        </select>
        <button onClick={onCreateDataset} disabled={busy}>+ New dataset</button>
      </div>

      <div style={{ marginTop: 8 }}>
        <input type="file"
          accept=".csv,.xls,.xlsx,text/csv,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
          onChange={e => e.target.files?.[0] && onPickFile(e.target.files[0])} />
      </div>

      {!!columns.length && (
        <div style={{ marginTop: 10 }}>
          <div>Date column:&nbsp;
            <select value={mapping.dateColumn} onChange={e => setMapping(m => ({ ...m, dateColumn: e.target.value }))}>
              {columns.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
            </select>
          </div>

          <div style={{ marginTop: 6 }}>
            Decimal:&nbsp;
            <select value={mapping.decimal} onChange={e => setMapping(m => ({ ...m, decimal: e.target.value as any }))}>
              <option value="auto">auto</option>
              <option value="dot">dot</option>
              <option value="comma">comma</option>
            </select>
          </div>

          <div style={{ marginTop: 8 }}>
            Value columns:
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 6 }}>
              {numberColumns.map(c => {
                const inc = mapping.valueColumns.some(v => v.name === c.name);
                return (
                  <button key={c.name}
                    onClick={() => setMapping(m => {
                      const exists = m.valueColumns.some(v => v.name === c.name);
                      return exists ? { ...m, valueColumns: m.valueColumns.filter(v => v.name !== c.name) }
                        : { ...m, valueColumns: [...m.valueColumns, { name: c.name, key: c.name.toUpperCase() }] };
                    })}
                    style={{ color: 'black', padding: '4px 8px', borderRadius: 8, border: '1px solid #ccc', background: inc ? '#eef' : '#fff' }}>
                    {c.name}{inc ? ' ✓' : ''}
                  </button>
                );
              })}
            </div>
          </div>

          <button style={{ marginTop: 10 }} onClick={onPreview} disabled={busy || !mapping.dateColumn || mapping.valueColumns.length === 0}>Preview</button>
        </div>
      )}

      {preview && (
        <div style={{ marginTop: 12 }}>
          <div style={{ color: '#666' }}>Warnings: {preview.warnings.length ? preview.warnings.join('; ') : 'none'}</div>
          {preview.normalized.series.map(s => (
            <div key={s.key} style={{ marginTop: 8 }}>
              <b>{s.key}</b> — {s.rowCount} rows
              <pre style={{ fontFamily: 'monospace', fontSize: 12 }}>
                {s.rows.slice(0, 8).map(r => `${r.date}  ${r.value}`).join('\n')}
                {s.rowCount > 8 ? '\n…' : ''}
              </pre>
            </div>
          ))}
          <button onClick={onCommit} disabled={busy} style={{ marginTop: 6 }}>Commit</button>
          {commit && <div style={{ color: '#2a6', marginTop: 6 }}>Upserted {commit.pointsUpserted} points; created: {commit.seriesCreated.join(', ') || '—'}</div>}
        </div>
      )}
    </div>
  );
}
