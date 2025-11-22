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
  // targets у diagnostics — це об’єкт { targetName: { lb_pvalue, residuals_ok } }
  const targetsInfo: Record<string, any> = (diag?.targets || {}) as Record<
    string,
    any
  >;
  const targetNames: string[] = Object.keys(targetsInfo);
  const baseVars: string[] = diag?.base_variables ?? [];

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const correlations = (experiment.correlations || {}) as any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const factors = (experiment.factors || {}) as any;

  const selectedModels = experiment.models.filter((m) => m.isSelected);

  // метрики без SeasonalNaive, тільки для таргетних рядів
  const targetMetrics = experiment.metrics.filter(
    (m) => m.modelType !== "SeasonalNaive" && targetNames.includes(m.seriesName)
  );

  const meta = diag?.meta;

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
          {meta && meta.start && meta.end
            ? `${meta.start} — ${meta.end} (${meta.n_rows} спостережень)`
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
              <th>ACF(12)</th>
              <th>Трансформація</th>
              <th>Нелінійність</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(seriesDiag).map(([name, info]) => {
              const roleFromDiag = info.role as
                | "target"
                | "candidate"
                | "ignored"
                | "base"
                | undefined;

              const role =
                roleFromDiag ??
                (targetNames.includes(name)
                  ? "target"
                  : baseVars.includes(name)
                    ? "base"
                    : "candidate");

              const adf_p: number | null =
                typeof info.adf_p === "number" ? info.adf_p : null;
              const kpss_p: number | null =
                typeof info.kpss_p === "number" ? info.kpss_p : null;
              const hasSeasonality = !!info.has_seasonality;
              const acf12: number | null =
                typeof info.acf_12 === "number" ? info.acf_12 : null;

              return (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{role}</td>
                  <td>
                    {typeof info.mean === "number"
                      ? info.mean.toFixed(2)
                      : ""}
                  </td>
                  <td>
                    {typeof info.std === "number" ? info.std.toFixed(2) : ""}
                  </td>
                  <td>{adf_p != null ? adf_p.toFixed(4) : ""}</td>
                  <td>
                    {adf_p != null
                      ? adf_p < 0.05
                        ? "так"
                        : "ні"
                      : ""}
                  </td>
                  <td>{kpss_p != null ? kpss_p.toFixed(4) : ""}</td>
                  <td>
                    {kpss_p != null
                      ? kpss_p > 0.05
                        ? "так"
                        : "ні"
                      : ""}
                  </td>
                  <td>{hasSeasonality ? "є" : "нема"}</td>
                  <td>{acf12 != null ? acf12.toFixed(3) : ""}</td>
                  <td>{info.transform ?? ""}</td>
                  <td>{info.is_nonlinear ? "так" : "ні"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      {/* 2. Кореляції та лаги (простий вивід) */}
      <section style={{ marginTop: "1.5rem" }}>
        <h3>2. Кореляційний аналіз і лаги</h3>
        <div style={{ fontSize: "0.9rem", marginBottom: "0.5rem" }}>
          <strong>Базові змінні:</strong>{" "}
          {baseVars.length ? baseVars.join(", ") : "не визначені"}
        </div>

        {/* Топ звʼязки для таргетів */}
        {targetNames.map((t) => {
          const edges = (correlations?.edges || []) as any[];
          const rel = edges
            .filter((e) => e.target === t)
            .filter((e) => e.r_at_best_lag != null)
            .sort(
              (a, b) =>
                Math.abs(b.r_at_best_lag) - Math.abs(a.r_at_best_lag)
            )
            .slice(0, 5);

          if (!rel.length) return null;

          return (
            <div key={t} style={{ marginBottom: "0.75rem" }}>
              <strong>{t}: найсильніші звʼязки</strong>
              <table style={{ width: "100%", fontSize: "0.85rem" }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left" }}>Предиктор</th>
                    <th>Lag</th>
                    <th>r</th>
                  </tr>
                </thead>
                <tbody>
                  {rel.map((e, idx) => (
                    <tr key={idx}>
                      <td>{e.source}</td>
                      <td>{e.best_lag}</td>
                      <td>
                        {e.r_at_best_lag != null
                          ? e.r_at_best_lag.toFixed(3)
                          : ""}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        })}
      </section>

      {/* 3. Факторний аналіз / VIF */}
      <section style={{ marginTop: "1.5rem" }}>
        <h3>3. Базові змінні та факторний аналіз</h3>
        <div style={{ fontSize: "0.9rem", marginBottom: "0.5rem" }}>
          <strong>Обрані базові змінні:</strong>{" "}
          {baseVars.length ? baseVars.join(", ") : "—"}
        </div>

        {/* VIF */}
        {factors?.vif && (
          <div style={{ marginBottom: "0.75rem" }}>
            <strong>VIF (мультиколінеарність)</strong>
            <table style={{ width: "100%", fontSize: "0.85rem" }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left" }}>Змінна</th>
                  <th>VIF</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(factors.vif).map(([name, v]: [string, any]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>
                      {typeof v === "number"
                        ? v.toFixed(2)
                        : v != null
                          ? String(v)
                          : ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* 4. Вибір моделей */}
      <section style={{ marginTop: "1.5rem" }}>
        <h3>4. Вибір моделей (обрані за методом)</h3>
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
                <td>
                  {m.mase != null ? m.mase.toFixed(3) : ""}
                </td>
                <td>
                  {m.smape != null ? m.smape.toFixed(1) + " %" : ""}
                </td>
                <td>
                  {m.rmse != null ? m.rmse.toFixed(3) : ""}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* 5. Метрики прогнозу по таргетах */}
      <section style={{ marginTop: "1.5rem" }}>
        <h3>5. Метрики прогнозу (таргетні ряди)</h3>
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
            {targetMetrics.map((m) => (
              <tr key={m.id}>
                <td>{m.seriesName}</td>
                <td>{m.modelType}</td>
                <td>{m.horizon}</td>
                <td>
                  {m.mase != null ? m.mase.toFixed(3) : ""}
                </td>
                <td>
                  {m.smape != null ? m.smape.toFixed(1) + " %" : ""}
                </td>
                <td>
                  {m.rmse != null ? m.rmse.toFixed(3) : ""}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* 6. Прогнози */}
      <section style={{ marginTop: "1.5rem" }}>
        <h3>6. Прогнози</h3>
        {targetNames.length === 0 && (
          <div>Таргетні ряди не визначені в diagnostics.</div>
        )}

        {targetNames.map((seriesName) => {
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
                        {f.valueActual != null
                          ? f.valueActual.toFixed(3)
                          : ""}
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
