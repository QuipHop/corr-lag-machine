import React from "react";
import { ML } from "../api/ml";
import type { CorrHeatmapReq, SeriesIn, TransformMode, CorrMethod } from "../types";

type Props = {
    series: SeriesIn[];
    targetCode: string;
};

export default function HeatmapPanel({ series, targetCode }: Props) {
    const [method, setMethod] = React.useState<CorrMethod>("pearson");
    const [transform, setTransform] = React.useState<TransformMode>("none");
    const [lagMin, setLagMin] = React.useState(-12);
    const [lagMax, setLagMax] = React.useState(12);
    const [minOverlap, setMinOverlap] = React.useState(12);
    const [topK, setTopK] = React.useState(10);
    const [fdrAlpha, setFdrAlpha] = React.useState<number | undefined>(0.1);
    const [loading, setLoading] = React.useState(false);
    const [error, setError] = React.useState<string | null>(null);
    const [resp, setResp] = React.useState<any>(null);

    const run = async () => {
        setLoading(true); setError(null);
        try {
            const body: CorrHeatmapReq = {
                series,
                targetCode,
                method,
                minOverlap,
                resample: { enabled: true, freq: "M", downsample: "last", upsample: "ffill", winsorize_q: 0 },
                lag: { min: lagMin, max: lagMax, ignoreZero: false },
                topK,
                returnStats: false,
                transform,
                returnP: true,
                fdrAlpha,
            };
            const data = await ML.corrHeatmap(body);
            setResp(data);
        } catch (e: any) {
            setError(e.message ?? String(e));
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="p-3 space-y-3">
            <div className="flex gap-2 flex-wrap">
                <label>Method:
                    <select value={method} onChange={e => setMethod(e.target.value as CorrMethod)}>
                        <option value="pearson">Pearson</option>
                        <option value="spearman">Spearman</option>
                    </select>
                </label>
                <label>Transform:
                    <select value={transform} onChange={e => setTransform(e.target.value as TransformMode)}>
                        <option value="none">none</option>
                        <option value="diff1">diff1</option>
                        <option value="pct">pct</option>
                    </select>
                </label>
                <label>Lag min:<input type="number" value={lagMin} onChange={e => setLagMin(Number(e.target.value))} className="w-20" /></label>
                <label>Lag max:<input type="number" value={lagMax} onChange={e => setLagMax(Number(e.target.value))} className="w-20" /></label>
                <label>Min overlap:<input type="number" value={minOverlap} onChange={e => setMinOverlap(Number(e.target.value))} className="w-20" /></label>
                <label>Top K:<input type="number" value={topK} onChange={e => setTopK(Number(e.target.value))} className="w-20" /></label>
                <label>FDR α:
                    <input type="number" step="0.01" value={fdrAlpha ?? 0} onChange={e => setFdrAlpha(e.target.value === "" ? undefined : Number(e.target.value))} className="w-24" />
                </label>
                <button onClick={run} disabled={loading} className="px-3 py-1 bg-blue-600 text-white rounded">{loading ? "Running..." : "Run"}</button>
            </div>

            {error && <div className="text-red-600">{error}</div>}

            {resp && (
                <div className="space-y-4">
                    <div>
                        <h3 className="font-semibold">Top factors (by |corr|)</h3>
                        <table className="min-w-full text-sm border">
                            <thead>
                                <tr className="bg-gray-100">
                                    <th className="p-2 border">Key</th>
                                    <th className="p-2 border">Lag (+ lead target)</th>
                                    <th className="p-2 border">Corr</th>
                                    <th className="p-2 border">n</th>
                                    <th className="p-2 border">p</th>
                                    <th className="p-2 border">FDR</th>
                                </tr>
                            </thead>
                            <tbody>
                                {resp.sortedTop.map((r: any) => (
                                    <tr key={r.key}>
                                        <td className="p-2 border">{r.key}</td>
                                        <td className="p-2 border">{r.lag}</td>
                                        <td className="p-2 border">{r.value.toFixed(4)}</td>
                                        <td className="p-2 border">{r.n}</td>
                                        <td className="p-2 border">{r.p !== undefined ? r.p.toExponential(2) : "-"}</td>
                                        <td className="p-2 border">{r.passed_fdr ? "✓" : (r.p ? "×" : "-")}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    {/* Можна додати heatmap-плитку з resp.matrix, якщо потрібно */}
                </div>
            )}
        </div>
    );
}
