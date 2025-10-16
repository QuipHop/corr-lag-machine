import { api } from "./client";
import type {
    CorrHeatmapReq, CorrHeatmapResp,
    CorrLagReq, CorrLagResp,
    SarimaxBacktestReq, SarimaxBacktestResp,
    SarimaxForecastReq, SarimaxForecastResp
} from "../types";

export const ML = {
    health: () => api<{ status: string }>("/ml/health"),
    corrHeatmap: (body: CorrHeatmapReq) =>
        api<CorrHeatmapResp, CorrHeatmapReq>("/ml/corr/heatmap", { method: "POST", body }),
    corrLag: (body: CorrLagReq) =>
        api<CorrLagResp, CorrLagReq>("/ml/corr/lag", { method: "POST", body }),
    sarimaxBacktest: (body: SarimaxBacktestReq) =>
        api<SarimaxBacktestResp, SarimaxBacktestReq>("/ml/sarimax/backtest", { method: "POST", body }),
    sarimaxForecast: (body: SarimaxForecastReq) =>
        api<SarimaxForecastResp, SarimaxForecastReq>("/ml/sarimax/forecast", { method: "POST", body }),
};
