// src/ui/ExperimentDetails.tsx
import React, { useEffect, useMemo, useState } from "react";
import {
  ExperimentDetails as ExperimentDetailsType,
  Forecast,
  getExperiment,
  getExperimentForecasts,
} from "../api/experiments";

type Props = {
  experimentId: string;
};

export const ExperimentDetails: React.FC<Props> = ({ experimentId }) => {
  const [experiment, setExperiment] = useState<ExperimentDetailsType | null>(
    null
  );
  const [forecasts, setForecasts] = useState<Forecast[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    (async () => {
      try {
        const [exp, fc] = await Promise.all([
          getExperiment(experimentId),
          getExperimentForecasts(experimentId),
        ]);
        if (cancelled) return;
        setExperiment(exp);
        setForecasts(fc);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } catch (err: any) {
        if (cancelled) return;
        console.error(err);
        setError(err?.message || "Помилка завантаження експерименту");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [experimentId]);

  const forecastsBySeries = useMemo(() => {
    const map: Record<string, Forecast[]> = {};
    for (const f of forecasts) {
      if (!map[f.seriesName]) map[f.seriesName] = [];
      map[f.seriesName].push(f);
    }
    // сортуємо за датою
    for (const key of Object.keys(map)) {
      map[key].sort(
        (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
      );
    }
    return map;
  }, [forecasts]);

  if (loading) {
    return <div style={{ padding: "1rem" }}>Завантаження...</div>;
  }

  if (error) {
    return <div style={{ padding: "1rem", color: "red" }}>{error}</div>;
  }

  if (!experiment) {
    return (
      <div style={{ padding: "1rem" }}>
        Обери експеримент, щоб побачити деталі.
      </div>
    );
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const diag = experiment.diagnostics as any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const seriesDiag = (diag?.series || {}) as Record<string, any>;
  const targets: string[] = diag?.targets ?? [];
  const baseVars: string[] = diag?.base_variables ?? [];

  const selectedModels = experiment.models.filter((m) => m.isSelected);

  return (
    <div style={{ padding: "1rem" }}>
      <h2>Експеримент: {experiment.name}</h2>
      <div style={{ fontSize: "0.9rem", marginBottom: "0.5rem" }}>
        <div>ID: {experiment.id}</div>
        <div>Контекст: {experiment.context || <em>не вказано</em>}</div>
        <div>
          Частота: {experiment.frequency}, горизонт: {experiment.horizon}
        </div>
        <div>
          Період даних:{" "}
          {diag?.meta
            ? `${diag.meta.start} — ${diag.meta.end} (${diag.meta.n_rows} спостережень)`
            : "—"}
        </div>
      </div>

      {/* 1. Діагностика рядів */}
      <section style={{ marginTop: "1rem" }}>
        <h3>1. Діагностика рядів</h3>
        <table style={{ width: "100%", fontSize: "0.9rem" }}>
          <thead>
            <tr>
              <th style={{ textAlign: "left" }}>Ряд</th>
              <th>Роль</th>
              <th>Mean</th>
              <th>Std</th>
              <th>ADF p</th>
              <th>ADF стац.</th>
              <th>KPSS p</th>
              <th>KPSS стац.</th>
              <th>Сезонність</th>
              <th>Seasonal corr</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(seriesDiag).map(([name, info]) => {
              const role =
                info.role ??
                (targets.includes(name)
                  ? "target"
                  : baseVars.includes(name)
                  ? "base"
                  : "candidate");

              const adf = info.adf || {};
              const kpss = info.kpss || {};

              return (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{role}</td>
                  <td>{info.mean?.toFixed?.(2) ?? ""}</td>
                  <td>{info.std?.toFixed?.(2) ?? ""}</td>
                  <td>{adf.pvalue != null ? adf.pvalue.toFixed(4) : ""}</td>
                  <td>
                    {adf.is_stationary === true
                      ? "так"
                      : adf.is_stationary === false
                      ? "ні"
                      : ""}
                  </td>
                  <td>{kpss.pvalue != null ? kpss.pvalue.toFixed(4) : ""}</td>
                  <td>
                    {kpss.is_stationary === true
                      ? "так"
                      : kpss.is_stationary === false
                      ? "ні"
                      : ""}
                  </td>
                  <td>
                    {info.has_seasonality ? `s=${info.seasonal_period}` : "—"}
                  </td>
                  <td>
                    {info.seasonal_corr != null
                      ? info.seasonal_corr.toFixed(3)
                      : ""}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      {/* 2. Вибір моделей */}
      <section style={{ marginTop: "1.5rem" }}>
        <h3>2. Вибір моделей (за MASE)</h3>
        <table style={{ width: "100%", fontSize: "0.9rem" }}>
          <thead>
            <tr>
              <th style={{ textAlign: "left" }}>Ряд</th>
              <th>Модель</th>
              <th>MASE</th>
              <th>sMAPE</th>
              <th>RMSE</th>
            </tr>
          </thead>
          <tbody>
            {selectedModels.map((m) => (
              <tr key={m.id}>
                <td>{m.seriesName}</td>
                <td>{m.modelType}</td>
                <td>{m.mase != null ? m.mase.toFixed(3) : ""}</td>
                <td>{m.smape != null ? m.smape.toFixed(1) + " %" : ""}</td>
                <td>{m.rmse != null ? m.rmse.toFixed(3) : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* 3. Метрики по таргетах */}
      <section style={{ marginTop: "1.5rem" }}>
        <h3>3. Метрики прогнозу (таргетні ряди)</h3>
        <table style={{ width: "100%", fontSize: "0.9rem" }}>
          <thead>
            <tr>
              <th style={{ textAlign: "left" }}>Ряд</th>
              <th>Обрана модель</th>
              <th>Горизонт</th>
              <th>MASE</th>
              <th>sMAPE</th>
              <th>RMSE</th>
            </tr>
          </thead>
          <tbody>
            {experiment.metrics.map((m) => (
              <tr key={m.id}>
                <td>{m.seriesName}</td>
                <td>{m.modelType}</td>
                <td>{m.horizon}</td>
                <td>{m.mase.toFixed(3)}</td>
                <td>{m.smape.toFixed(1)} %</td>
                <td>{m.rmse.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* 4. Прогнози */}
      <section style={{ marginTop: "1.5rem" }}>
        <h3>4. Прогнози</h3>
        {targets.length === 0 && (
          <div>Таргетні ряди не визначені в diagnostics.</div>
        )}

        {targets.map((seriesName) => {
          const seriesForecasts = forecastsBySeries[seriesName] || [];
          if (!seriesForecasts.length) return null;

          return (
            <details key={seriesName} style={{ marginBottom: "1rem" }} open>
              <summary>{seriesName} (train/test/future у таблиці)</summary>
              <table
                style={{
                  width: "100%",
                  fontSize: "0.85rem",
                  marginTop: "0.5rem",
                }}
              >
                <thead>
                  <tr>
                    <th style={{ textAlign: "left" }}>Дата</th>
                    <th>Набір</th>
                    <th>Факт</th>
                    <th>Прогноз</th>
                  </tr>
                </thead>
                <tbody>
                  {seriesForecasts.map((f) => (
                    <tr key={`${f.seriesName}-${f.date}-${f.setType}`}>
                      <td>{f.date.slice(0, 10)}</td>
                      <td>{f.setType}</td>
                      <td>
                        {f.valueActual != null ? f.valueActual.toFixed(3) : ""}
                      </td>
                      <td>
                        {f.valuePred != null ? f.valuePred.toFixed(3) : ""}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          );
        })}
      </section>
    </div>
  );
};
