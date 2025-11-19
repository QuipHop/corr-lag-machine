import React from "react";
import { ML } from "../api/ml";
import type { SeriesIn, SarimaxBacktestReq, SarimaxForecastReq, TransformMode } from "../types";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Area, AreaChart } from "recharts";

type Props = { series: SeriesIn[]; targetCode: string; selectedFeatures: string[]; lags: Record<string, number>; };

export default function SarimaxPanel({ series, targetCode, selectedFeatures, lags }: Props) {
    const [transform, setTransform] = React.useState<TransformMode>("none");
    const [horizon, setHorizon] = React.useState(6);
    const [minTrain, setMinTrain] = React.useState(36);
    const [sSeason, setSSeason] = React.useState(12);
    const [loading, setLoading] = React.useState(false);
    const [back, setBack] = React.useState<any>(null);
    const [fc, setFc] = React.useState<any>(null);
    const [err, setErr] = React.useState<string | null>(null);

    const runBacktest = async () => {
        setLoading(true); setErr(null);
        try {
            const body: SarimaxBacktestReq = {
                series,
                resample: { enabled: true, freq: "M", downsample: "last", upsample: "ffill", winsorize_q: 0 },
                transform,
                features_cfg: { targetCode, features: selectedFeatures, lags },
                train: { auto_grid: { p: [0, 1], d: [0, 1], q: [0, 1], P: [0, 1], D: [0, 1], Q: [0, 1], s: sSeason, max_models: 12 } },
                backtest: { horizon: Math.max(1, horizon), min_train: minTrain, step: 1, expanding: true },
            };
            const data = await ML.sarimaxBacktest(body);
            setBack(data);
        } catch (e: any) { setErr(e.message ?? String(e)); }
        finally { setLoading(false); }
    };

    const runForecast = async () => {
        setLoading(true); setErr(null);
        try {
            const body: SarimaxForecastReq = {
                series,
                resample: { enabled: true, freq: "M", downsample: "last", upsample: "ffill", winsorize_q: 0 },
                transform,
                features_cfg: { targetCode, features: selectedFeatures, lags },
                train: { auto_grid: { p: [0, 1], d: [0, 1], q: [0, 1], P: [0, 1], D: [0, 1], Q: [0, 1], s: sSeason, max_models: 12 } },
                horizon: Math.max(1, horizon),
                return_pi: true,
                alpha: 0.1
            };
            const data = await ML.sarimaxForecast(body);
            setFc(data);
        } catch (e: any) { setErr(e.message ?? String(e)); }
        finally { setLoading(false); }
    };

    return (
        <div className="p-3 space-y-3">
            <div className="flex gap-2 flex-wrap">
                <label>Transform:
                    <select value={transform} onChange={e => setTransform(e.target.value as TransformMode)}>
                        <option value="none">none</option>
                        <option value="diff1">diff1</option>
                        <option value="pct">pct</option>
                    </select>
                </label>
                <label>Horizon:<input type="number" value={horizon} onChange={e => setHorizon(Number(e.target.value))} className="w-20" /></label>
                <label>Min train:<input type="number" value={minTrain} onChange={e => setMinTrain(Number(e.target.value))} className="w-24" /></label>
                <label>Season s:<input type="number" value={sSeason} onChange={e => setSSeason(Number(e.target.value))} className="w-20" /></label>
                <button onClick={runBacktest} disabled={loading} className="px-3 py-1 bg-indigo-600 text-white rounded">{loading ? "..." : "Backtest"}</button>
                <button onClick={runForecast} disabled={loading} className="px-3 py-1 bg-green-600 text-white rounded">{loading ? "..." : "Forecast"}</button>
            </div>

            {err && <div className="text-red-600">{err}</div>}

            {back && (
                <div className="border p-2 rounded">
                    <div className="font-semibold mb-2">Backtest metrics</div>
                    <div>MAE: {back.metrics.MAE.toFixed(3)} | RMSE: {back.metrics.RMSE.toFixed(3)} | sMAPE: {back.metrics.sMAPE.toFixed(2)}%</div>
                    <div className="text-xs text-gray-500">folds: {back.meta.folds}, n_obs: {back.meta.n_obs}, t={back.meta.timing_s}s</div>
                </div>
            )}

            {fc && (
                <div className="border p-2 rounded">
                    <div className="font-semibold mb-2">Forecast (h={horizon})</div>

                    <div className="h-64 w-full">
                        <ResponsiveContainer>
                            <AreaChart data={fc.forecast}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="date" />
                                <YAxis />
                                <Tooltip />
                                {/* інтервал як напівпрозора стрічка */}
                                <Area type="monotone" dataKey="hi" dot={false} activeDot={false} />
                                <Area type="monotone" dataKey="lo" dot={false} activeDot={false} />
                                <Line type="monotone" dataKey="mean" dot={false} />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="text-xs text-gray-500 mt-1">order={String(fc.order)} seasonal={String(fc.seasonal_order)} AIC={fc.aic.toFixed(2)}</div>
                </div>
            )}
        </div>
    );
}
