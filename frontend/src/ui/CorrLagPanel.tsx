import React from "react";
import { ML } from "../api/ml";
import type { CorrLagReq, SeriesIn, TransformMode, CorrMethod } from "../types";

type Props = { series: SeriesIn[] };

export default function CorrLagPanel({ series }: Props) {
  const [method, setMethod] = React.useState<CorrMethod>("pearson");
  const [transform, setTransform] = React.useState<TransformMode>("none");
  const [minOverlap, setMinOverlap] = React.useState(12);
  const [edgeMin, setEdgeMin] = React.useState(0.3);
  const [lagMin, setLagMin] = React.useState(-12);
  const [lagMax, setLagMax] = React.useState(12);
  const [topK, setTopK] = React.useState(20);
  const [perNodeTopK, setPerNodeTopK] = React.useState<number | undefined>(undefined);
  const [fdrAlpha, setFdrAlpha] = React.useState<number | undefined>(0.1);

  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [edges, setEdges] = React.useState<any[]>([]);

  const run = async () => {
    setLoading(true); setError(null);
    try {
      const body: CorrLagReq = {
        series,
        method,
        minOverlap,
        edgeMin,
        resample: { enabled: true, freq: "M", downsample: "last", upsample: "ffill", winsorize_q: 0 },
        lag: { min: lagMin, max: lagMax, ignoreZero: false },
        normalizeOrientation: true,
        dedupeOpposite: true,
        topK,
        perNodeTopK,
        returnStats: false,
        transform,
        returnP: true,
        fdrAlpha,
      };
      const data = await ML.corrLag(body);
      setEdges(data.edges);
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
        <label>Edge min:<input type="number" step="0.01" value={edgeMin} onChange={e => setEdgeMin(Number(e.target.value))} className="w-20" /></label>
        <label>TopK:<input type="number" value={topK} onChange={e => setTopK(Number(e.target.value))} className="w-20" /></label>
        <label>Per-node TopK:<input type="number" value={perNodeTopK ?? 0} onChange={e => setPerNodeTopK(Number(e.target.value) || undefined)} className="w-24" /></label>
        <label>FDR α:
          <input type="number" step="0.01" value={fdrAlpha ?? 0} onChange={e => setFdrAlpha(e.target.value === "" ? undefined : Number(e.target.value))} className="w-24" />
        </label>
        <button onClick={run} disabled={loading} className="px-3 py-1 bg-blue-600 text-white rounded">{loading ? "Running..." : "Run"}</button>
      </div>

      {error && <div className="text-red-600">{error}</div>}

      {!!edges.length && (
        <table className="min-w-full text-sm border">
          <thead>
            <tr className="bg-gray-100">
              <th className="p-2 border">source</th>
              <th className="p-2 border">target</th>
              <th className="p-2 border">lag (+ lead src)</th>
              <th className="p-2 border">|corr|</th>
              <th className="p-2 border">n</th>
              <th className="p-2 border">p</th>
              <th className="p-2 border">FDR</th>
            </tr>
          </thead>
          <tbody>
            {edges.map((e, i) => (
              <tr key={i}>
                <td className="p-2 border">{e.source}</td>
                <td className="p-2 border">{e.target}</td>
                <td className="p-2 border">{e.lag}</td>
                <td className="p-2 border">{Math.abs(e.weight).toFixed(4)}</td>
                <td className="p-2 border">{e.n}</td>
                <td className="p-2 border">{e.p !== undefined ? Number(e.p).toExponential(2) : "-"}</td>
                <td className="p-2 border">{e.passed_fdr ? "✓" : (e.p ? "×" : "-")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
