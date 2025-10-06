import { useEffect, useState } from 'react';
import { listDatasets, listSeriesForDataset, getSeriesData } from './api.series';
import type { DatasetLite, SeriesLite, ObservationPoint } from '../types';

export default function SeriesBrowser() {
    const [datasets, setDatasets] = useState<DatasetLite[]>([]);
    const [datasetId, setDatasetId] = useState<string>('');
    const [series, setSeries] = useState<SeriesLite[]>([]);
    const [seriesId, setSeriesId] = useState<string>('');
    const [points, setPoints] = useState<ObservationPoint[]>([]);
    const [err, setErr] = useState<string>('');

    useEffect(() => {
        listDatasets().then(ds => {
            setDatasets(ds);
            if (ds[0]) setDatasetId(ds[0].id);
        }).catch(e => setErr(e.message ?? String(e)));
    }, []);

    useEffect(() => {
        if (!datasetId) return;
        listSeriesForDataset(datasetId).then(ss => {
            setSeries(ss);
            if (ss[0]) setSeriesId(ss[0].id);
        }).catch(e => setErr(e.message ?? String(e)));
    }, [datasetId]);

    useEffect(() => {
        if (!seriesId) return;
        getSeriesData(seriesId).then(r => setPoints(r.points)).catch(e => setErr(e.message ?? String(e)));
    }, [seriesId]);

    return (
        <div style={{ padding: 12 }}>
            <h3>Series Browser</h3>
            {err && <div style={{ color: '#c33' }}>{err}</div>}

            <div style={{ display: 'flex', gap: 8, margin: '8px 0' }}>
                <select value={datasetId} onChange={e => setDatasetId(e.target.value)}>
                    {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
                </select>
                <select value={seriesId} onChange={e => setSeriesId(e.target.value)}>
                    {series.map(s => <option key={s.id} value={s.id}>{s.key}</option>)}
                </select>
            </div>

            <pre style={{ fontFamily: 'monospace', fontSize: 12, whiteSpace: 'pre-wrap' }}>
                {points.slice(0, 20).map(p => `${p.date}  ${p.value}`).join('\n')}
                {points.length > 20 ? '\n…' : ''}
            </pre>
        </div>
    );
}
