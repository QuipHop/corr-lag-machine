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
    shape: 'long',
    dateColumn: '',
    valueColumns: [],
    decimal: 'auto',
    dateFormat: 'YYYY-MM',
    dropBlanks: true,
    seriesKeyColumn: '',
    year: undefined,
  });

  const [preview, setPreview] = useState<PreviewResp | null>(null);
  const [commit, setCommit] = useState<CommitResp | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');

  // ---- bootstrap datasets
  useEffect(() => {
    listDatasets()
      .then(ds => {
        setDatasets(ds);
        if (ds[0]) setDatasetId(ds[0].id);
      })
      .catch(e => setErr(e.message ?? String(e)));
  }, []);

  // ---- prefill mapping from saved dataset mapping (if any)
  useEffect(() => {
    if (!datasetId) return;
    getSavedMapping(datasetId)
      .then(m => {
        if (!m || !Object.keys(m).length) return;
        const as = m as any;
        setMapping({
          shape: as.shape ?? 'long',
          dateColumn: as.dateColumn ?? '',
          valueColumns: Array.isArray(as.valueColumns) ? as.valueColumns : [],
          decimal: as.decimal ?? 'auto',
          dateFormat: as.dateFormat ?? 'YYYY-MM',
          dropBlanks: as.dropBlanks ?? true,
          seriesKeyColumn: as.seriesKeyColumn ?? '',
          monthColumns: as.monthColumns,        // optional; BE will auto-detect if missing
          year: as.year,
        });
      })
      .catch(() => { /* ignore */ });
  }, [datasetId]);

  // ---- column helpers
  const numberColumns = useMemo(
    () => columns.filter(c => c.typeGuess === 'number'),
    [columns]
  );

  const valueColumnOptions = useMemo(
    () => numberColumns.filter(c => c.name !== mapping.dateColumn),
    [numberColumns, mapping.dateColumn]
  );

  const canPreview = useMemo(() => {
    if (!uploadId) return false;
    if (mapping.shape === 'wide') {
      return !!mapping.seriesKeyColumn; // month columns auto-detected; year optional
    }
    return !!mapping.dateColumn && (mapping.valueColumns?.length ?? 0) > 0;
  }, [uploadId, mapping]);

  // ---- actions
  const onCreateDataset = async () => {
    setBusy(true); setErr('');
    try {
      const ds = await createDataset({ name: `Dataset ${new Date().toISOString().slice(0, 10)}`, freq: 'monthly' });
      setDatasets(d => [ds, ...d]);
      setDatasetId(ds.id);
    } catch (e: any) {
      setErr(e.message ?? String(e));
    } finally {
      setBusy(false);
    }
  };

  const onPickFile = async (f: File) => {
    if (!datasetId) return;
    setBusy(true); setErr(''); setPreview(null); setCommit(null);
    try {
      const up = await uploadFile(datasetId, f);
      setUploadId(up.uploadId);
      setColumns(up.columns);

      const dateGuess = up.columns.find(c => c.typeGuess === 'date')?.name ?? up.columns[0]?.name ?? '';
      const vals = up.columns
        .filter(c => c.typeGuess === 'number' && c.name !== dateGuess)
        .slice(0, 3);

      setMapping(m => {
        if (m.shape === 'wide') {
          // choose a non-date-looking text column as series key, fallback to first col
          const seriesKey =
            up.columns.find(c => c.typeGuess !== 'number' && c.name !== dateGuess)?.name
            ?? up.columns[0]?.name
            ?? '';
          return { ...m, seriesKeyColumn: seriesKey, year: new Date().getFullYear() };
        }
        return {
          ...m,
          dateColumn: dateGuess,
          valueColumns: vals.map(v => ({ name: v.name, key: v.name.toUpperCase() })),
        };
      });
    } catch (e: any) {
      setErr(e.message ?? String(e));
    } finally {
      setBusy(false);
    }
  };

  const onPreview = async () => {
    setBusy(true); setErr('');
    try {
      const p = await previewUpload(uploadId, mapping);
      setPreview(p);
    } catch (e: any) {
      setErr(e.message ?? String(e));
    } finally {
      setBusy(false);
    }
  };

  const onCommit = async () => {
    setBusy(true); setErr('');
    try {
      const c = await commitUpload(uploadId, { saveMappingToDataset: true, createSeries: true, upsertMode: 'merge' });
      setCommit(c);
    } catch (e: any) {
      setErr(e.message ?? String(e));
    } finally {
      setBusy(false);
    }
  };

  // ---- UI
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
        <input
          type="file"
          accept=".csv,.xls,.xlsx,text/csv,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
          onChange={e => e.target.files?.[0] && onPickFile(e.target.files[0])}
        />
      </div>

      {!!columns.length && (
        <div style={{ marginTop: 10 }}>
          {/* shape toggle */}
          <div style={{ marginTop: 8 }}>
            <label>
              <input
                type="checkbox"
                checked={mapping.shape === 'wide'}
                onChange={e =>
                  setMapping(m => ({
                    ...m,
                    shape: e.target.checked ? 'wide' : 'long',
                    ...(e.target.checked
                      ? { dateColumn: '', valueColumns: [] }
                      : { seriesKeyColumn: '', monthColumns: undefined, year: undefined })
                  }))
                }
              />{' '}
              Months are columns (wide)
            </label>
          </div>

          {/* LONG MODE */}
          {mapping.shape === 'long' && (
            <>
              <div style={{ marginTop: 10 }}>
                Date column:&nbsp;
                <select
                  value={mapping.dateColumn}
                  onChange={e =>
                    setMapping(m => ({
                      ...m,
                      dateColumn: e.target.value,
                      valueColumns: (m.valueColumns ?? []).filter(v => v.name !== e.target.value),
                    }))
                  }
                >
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
                  {valueColumnOptions.map(c => {
                    const inc = (mapping.valueColumns ?? []).some(v => v.name === c.name);
                    return (
                      <button
                        key={c.name}
                        onClick={() =>
                          setMapping(m => {
                            const exists = (m.valueColumns ?? []).some(v => v.name === c.name);
                            return exists
                              ? { ...m, valueColumns: (m.valueColumns ?? []).filter(v => v.name !== c.name) }
                              : { ...m, valueColumns: [...(m.valueColumns ?? []), { name: c.name, key: c.name.toUpperCase() }] };
                          })
                        }
                        style={{ color: 'black', padding: '4px 8px', borderRadius: 8, border: '1px solid #ccc', background: inc ? '#eef' : '#fff' }}
                      >
                        {c.name}{inc ? ' ✓' : ''}
                      </button>
                    );
                  })}
                </div>
              </div>
            </>
          )}

          {/* WIDE MODE */}
          {mapping.shape === 'wide' && (
            <>
              <div style={{ marginTop: 10 }}>
                Series key column:&nbsp;
                <select
                  value={mapping.seriesKeyColumn ?? ''}
                  onChange={e => setMapping(m => ({ ...m, seriesKeyColumn: e.target.value }))}
                >
                  {columns.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
                </select>
              </div>

              <div style={{ marginTop: 6 }}>
                Year (if month headers don’t include a year):&nbsp;
                <input
                  type="number"
                  value={mapping.year ?? ''}
                  onChange={e => setMapping(m => ({ ...m, year: e.target.value ? Number(e.target.value) : undefined }))}
                  style={{ width: 110 }}
                />
              </div>

              <div style={{ marginTop: 6 }}>
                Decimal:&nbsp;
                <select value={mapping.decimal} onChange={e => setMapping(m => ({ ...m, decimal: e.target.value as any }))}>
                  <option value="auto">auto</option>
                  <option value="dot">dot</option>
                  <option value="comma">comma</option>
                </select>
              </div>
            </>
          )}

          <button
            style={{ marginTop: 10 }}
            onClick={onPreview}
            disabled={busy || !canPreview}
          >
            Preview
          </button>
        </div>
      )}

      {preview && (
        <div style={{ marginTop: 12 }}>
          <div style={{ color: '#666' }}>
            Warnings: {preview.warnings.length ? preview.warnings.join('; ') : 'none'}
          </div>
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
          {commit && (
            <div style={{ color: '#2a6', marginTop: 6 }}>
              Upserted {commit.pointsUpserted} points; created: {commit.seriesCreated.join(', ') || '—'}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
