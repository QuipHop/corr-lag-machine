import React, { useEffect, useMemo, useState } from 'react';
import { useAnalysisSelection } from '../state/analysisSelection';
import type { SeriesIn, Point } from '../api/analysis';

const API_BASE = (import.meta.env.VITE_API_BASE as string) || '/api';
// хелпери
async function getJson(url: string) {
    const r = await fetch(url);
    if (!r.ok) {
        const t = await r.text().catch(() => '');
        throw new Error(`${r.status} ${r.statusText}: ${t}`);
    }
    return r.json();
}

// --- бекенд-виклики з fallback ---
// 1) датасети
async function fetchDatasets(): Promise<Array<{ id: string; name: string }>> {
    const list = await getJson(`${API_BASE}/api/datasets`);
    // приводимо до уніфікованої форми
    return (list || []).map((d: any) => ({
        id: String(d.id ?? d.datasetId ?? d.name ?? d.title ?? d),
        name: String(d.name ?? d.title ?? d.id ?? d),
    }));
}

// 2) коди серій для датасету
async function fetchSeriesCodes(datasetId: string): Promise<string[]> {
    try {
        const a = await getJson(`${API_BASE}/series/list?datasetId=${encodeURIComponent(datasetId)}`);
        return (a?.codes ?? a ?? []).map((x: any) => String(x));
    } catch {
        // альтернативний формат
        const b = await getJson(`${API_BASE}/series/list/${encodeURIComponent(datasetId)}`);
        return (b?.codes ?? b ?? []).map((x: any) => String(x));
    }
}

// 3) точки серії
async function fetchSeriesPoints(datasetId: string, code: string): Promise<Point[]> {
    try {
        const a = await getJson(`${API_BASE}/series/${encodeURIComponent(datasetId)}/${encodeURIComponent(code)}`);
        return (a?.points ?? a ?? []).map((p: any) => ({
            date: String(p.date).slice(0, 10),
            value: Number(p.value),
        }));
    } catch {
        const b = await getJson(
            `${API_BASE}/series?datasetId=${encodeURIComponent(datasetId)}&code=${encodeURIComponent(code)}`
        );
        return (b?.points ?? b ?? []).map((p: any) => ({
            date: String(p.date).slice(0, 10),
            value: Number(p.value),
        }));
    }
}

type Dataset = { id: string; name: string };

export default function SeriesBrowser() {
    const { upsertSeries, removeByCode, selected } = useAnalysisSelection();

    const [datasets, setDatasets] = useState<Dataset[]>([]);
    const [datasetId, setDatasetId] = useState<string>('');   // <-- РЯДОК
    const [codes, setCodes] = useState<string[]>([]);
    const [code, setCode] = useState<string>('');

    const [rows, setRows] = useState<Point[]>([]);
    const [loading, setLoading] = useState(false);
    const [err, setErr] = useState<string | null>(null);

    // завантажуємо список датасетів
    useEffect(() => {
        (async () => {
            try {
                const ds = await fetchDatasets();
                setDatasets(ds);
                if (ds.length) {
                    // беремо перший як дефолт
                    setDatasetId(ds[0].id);
                }
            } catch (e: any) {
                setErr(e?.message || 'Failed to load datasets');
            }
        })();
    }, []);

    // коди серій при зміні датасету
    useEffect(() => {
        if (!datasetId) { setCodes([]); setCode(''); setRows([]); return; }
        (async () => {
            try {
                const cs = await fetchSeriesCodes(datasetId);
                setCodes(cs);
                setCode(cs[0] ?? '');
            } catch (e: any) {
                setErr(e?.message || 'Failed to load series codes');
                setCodes([]); setCode('');
            }
        })();
    }, [datasetId]);

    // точки серії
    useEffect(() => {
        if (!datasetId || !code) { setRows([]); return; }
        (async () => {
            setLoading(true); setErr(null);
            try {
                const pts = await fetchSeriesPoints(datasetId, code);
                const norm = (pts || [])
                    .map((p) => ({ date: String(p.date).slice(0, 10), value: Number(p.value) }))
                    .filter((p) => p.date && Number.isFinite(p.value));
                setRows(norm);
            } catch (e: any) {
                setErr(e?.message || 'Failed to load points');
                setRows([]);
            } finally {
                setLoading(false);
            }
        })();
    }, [datasetId, code]);

    const inSelection = useMemo(() => selected.some(s => s.code === code), [selected, code]);

    const handleAdd = () => {
        if (!code || !rows.length) return;
        const series: SeriesIn = { id: Date.now(), code, points: rows };
        upsertSeries(series);
    };
    const handleRemove = () => { if (code) removeByCode(code); };

    return (
        <div style={{ textAlign: 'center' }}>
            <h3>Series Browser</h3>

            <div style={{ display: 'inline-flex', gap: 12, alignItems: 'center' }}>
                {/* dataset select: value — РЯДОК */}
                <select
                    value={datasetId}
                    onChange={e => setDatasetId(e.target.value)}
                    style={{ padding: '8px 12px', borderRadius: 8 }}
                >
                    {datasets.map(d => (
                        <option key={d.id} value={d.id}>{d.name}</option>
                    ))}
                </select>

                <select
                    value={code}
                    onChange={e => setCode(e.target.value)}
                    style={{ padding: '8px 12px', borderRadius: 8 }}
                >
                    {codes.map(c => <option key={c} value={c}>{c}</option>)}
                </select>

                <button onClick={handleAdd} disabled={!rows.length} style={{ padding: '8px 12px' }}>
                    {inSelection ? 'Update selection' : 'Add to analysis'}
                </button>
                <button onClick={handleRemove} disabled={!inSelection} style={{ padding: '8px 12px' }}>
                    Remove
                </button>
            </div>

            {err && <div style={{ color: 'tomato', marginTop: 8 }}>{err}</div>}

            <div style={{ marginTop: 16 }}>
                {loading && <div>Loading…</div>}
                {!loading && rows.length === 0 && <div style={{ opacity: 0.7 }}>No data</div>}

                {!!rows.length && (
                    <div style={{ display: 'inline-block', textAlign: 'left' }}>
                        {rows.map((r, i) => (
                            <div key={i} style={{ fontFamily: 'monospace' }}>
                                {r.date.padEnd(12)} {String(r.value).padStart(6)}
                            </div>
                        ))}
                    </div>
                )}
            </div>

            <div style={{ marginTop: 12, fontSize: 12, opacity: 0.8 }}>
                Selected: {selected.map(s => s.code).join(', ') || '—'}
            </div>
        </div>
    );
}
