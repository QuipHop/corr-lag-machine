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

  // діагностика / кореляції / фактори
  const diag = (experiment?.diagnostics || {}) as any;
  const seriesDiag = (diag?.series || {}) as Record<string, any>;
  const targetsInfo: Record<string, any> = (diag?.targets ||
    {}) as Record<string, any>;
  const targetNames: string[] = Object.keys(targetsInfo);
  const baseVars: string[] = (diag?.base_variables || []) as string[];

  const correlations = (experiment?.correlations || {}) as any;
  const factors = (experiment?.factors || {}) as any;

  const selectedModels = experiment?.models?.filter((m) => m.isSelected) ?? [];
  const meta = diag?.meta;
  const comparison = diag?.comparison || {};

  // 1) ефективність комбінованого способу для таргетів:
  //    беремо ТІЛЬКИ обрану модель для кожного таргета, без SeasonalNaive.
  const efficiencyRows = useMemo(() => {
    if (!experiment) return [];

    return targetNames
      .map((seriesName) => {
        const selectedForTarget = selectedModels.find(
          (m) => m.seriesName === seriesName
        );
        if (!selectedForTarget) return null;

        const metric = experiment.metrics.find(
          (mm) =>
            mm.seriesName === seriesName &&
            mm.modelType === selectedForTarget.modelType
        );

        const tDiag = targetsInfo[seriesName] || {};
        const lbPvalue =
          typeof tDiag.lb_pvalue === "number" ? tDiag.lb_pvalue : null;
        const residualsOk =
          typeof tDiag.residuals_ok === "boolean"
            ? (tDiag.residuals_ok as boolean)
            : null;

        return {
          seriesName,
          modelType: selectedForTarget.modelType,
          mase:
            typeof metric?.mase === "number" && !Number.isNaN(metric.mase)
              ? metric.mase
              : null,
          smape:
            typeof metric?.smape === "number" && !Number.isNaN(metric.smape)
              ? metric.smape
              : null,
          rmse:
            typeof metric?.rmse === "number" && !Number.isNaN(metric.rmse)
              ? metric.rmse
              : null,
          lbPvalue,
          residualsOk,
        };
      })
      .filter(Boolean) as Array<{
        seriesName: string;
        modelType: string;
        mase: number | null;
        smape: number | null;
        rmse: number | null;
        lbPvalue: number | null;
        residualsOk: boolean | null;
      }>;
  }, [experiment, selectedModels, targetNames, targetsInfo]);

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
                  <td>
                    {role === "target"
                      ? "цільовий ряд"
                      : role === "base"
                        ? "базовий ряд"
                        : role}
                  </td>
                  <td>
                    {typeof info.mean === "number"
                      ? info.mean.toFixed(2)
                      : ""}
                  </td>
                  <td>
                    {typeof info.std === "number" ? info.std.toFixed(2) : ""}
                  </td>
                  <td>{adf_p != null ? adf_p.toFixed(4) : ""}</td>
                  <td>{adf_p != null ? (adf_p < 0.05 ? "так" : "ні") : ""}</td>
                  <td>{kpss_p != null ? kpss_p.toFixed(4) : ""}</td>
                  <td>
                    {kpss_p != null ? (kpss_p > 0.05 ? "так" : "ні") : ""}
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

      {/* 2. Кореляції та лаги */}
      <section style={{ marginTop: "1.5rem" }}>
        <h3>2. Кореляційний аналіз і лаги</h3>
        <div style={{ fontSize: "0.9rem", marginBottom: "0.5rem" }}>
          <strong>Базові змінні:</strong>{" "}
          {baseVars.length ? baseVars.join(", ") : "не визначені"}
        </div>

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

      {/* 4. Вибір моделей (обрані за методом) */}
      <section style={{ marginTop: "1.5rem" }}>
        <h3>4. Вибір моделей (обрані за методом)</h3>
        <table style={{ width: "100%", fontSize: "0.9rem" }}>
          <thead>
            <tr>
              <th style={{ textAlign: "left" }}>Ряд</th>
              <th>Роль</th>
              <th>Модель</th>
              <th>MASE</th>
              <th>sMAPE</th>
              <th>RMSE</th>
            </tr>
          </thead>
          <tbody>
            {selectedModels.map((m) => {
              const role = targetNames.includes(m.seriesName)
                ? "цільовий ряд"
                : baseVars.includes(m.seriesName)
                  ? "базовий ряд"
                  : "";
              return (
                <tr key={m.id}>
                  <td>{m.seriesName}</td>
                  <td>{role}</td>
                  <td>{m.modelType}</td>
                  <td>{m.mase != null ? m.mase.toFixed(3) : ""}</td>
                  <td>{m.smape != null ? m.smape.toFixed(1) + " %" : ""}</td>
                  <td>{m.rmse != null ? m.rmse.toFixed(3) : ""}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      {/* 5. Оцінка ефективності комбінованого способу (тільки таргети, без SeasonalNaive в UI) */}
      <section style={{ marginTop: "1.5rem" }}>
        <h3>5. Оцінка ефективності комбінованого способу (таргетні ряди)</h3>
        {efficiencyRows.length === 0 ? (
          <div style={{ fontSize: "0.9rem" }}>
            Немає даних для таргетних рядів.
          </div>
        ) : (
          <table style={{ width: "100%", fontSize: "0.8rem" }}>
            <thead>
              <tr>
                <th style={{ textAlign: "left" }}>Ряд</th>
                <th>Обрана модель</th>
                <th>MASE</th>
                <th>sMAPE</th>
                <th>RMSE</th>
                <th>p-value Ljung–Box</th>
                <th>Залишки ок?</th>
              </tr>
            </thead>
            <tbody>
              {efficiencyRows.map((row) => (
                <tr key={row.seriesName}>
                  <td>{row.seriesName}</td>
                  <td>{row.modelType}</td>
                  <td>
                    {row.mase != null ? row.mase.toFixed(3) : ""}
                  </td>
                  <td>
                    {row.smape != null ? row.smape.toFixed(1) + " %" : ""}
                  </td>
                  <td>
                    {row.rmse != null ? row.rmse.toFixed(3) : ""}
                  </td>
                  <td>
                    {row.lbPvalue != null
                      ? row.lbPvalue.toFixed(3)
                      : ""}
                  </td>
                  <td>
                    {row.residualsOk == null
                      ? ""
                      : row.residualsOk
                        ? "так"
                        : "ні"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {/* 6. Прогнози (тільки backtest) */}
      <section style={{ marginTop: "1.5rem" }}>
        <h3>6. Прогнози (backtest)</h3>
        {targetNames.length === 0 && (
          <div>Таргетні ряди не визначені в diagnostics.</div>
        )}

        {targetNames.map((seriesName) => {
          const seriesForecasts = (forecastsBySeries[seriesName] || []).filter(
            (f) => f.setType === "train" || f.setType === "test"
          );
          if (!seriesForecasts.length) return null;

          return (
            <details key={seriesName} style={{ marginBottom: "1rem" }} open>
              <summary>{seriesName} (train/test, без future)</summary>
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

      {/* 7. Порівняння моделей (горизонти 1–3 місяці) */}
      <section style={{ marginTop: "1.5rem" }}>
        <h3>7. Порівняння моделей (горизонти 1–3 місяці)</h3>

        {targetNames.map((t) => {
          const cmp = (comparison && (comparison as any)[t]) || null;
          if (!cmp) return null;

          const horizons = Object.keys(cmp)
            .map((h) => Number(h))
            .sort((a, b) => a - b);
          if (!horizons.length) return null;

          return (
            <div key={t} style={{ marginBottom: "1rem" }}>
              <h4>{t}</h4>

              {horizons.map((h) => {
                const famRes = (cmp as any)[h];
                if (!famRes) return null;
                const families = Object.keys(famRes);
                if (!families.length) return null;

                return (
                  <details
                    key={h}
                    style={{ marginBottom: "0.5rem" }}
                    open={h === 3}
                  >
                    <summary>Горизонт {h} місяць(і)</summary>
                    <table
                      style={{
                        width: "100%",
                        fontSize: "0.85rem",
                        marginTop: "0.5rem",
                      }}
                    >
                      <thead>
                        <tr>
                          <th style={{ textAlign: "left" }}>Модель</th>
                          <th>MASE</th>
                          <th>sMAPE</th>
                          <th>RMSE</th>
                          <th>t навчання, c</th>
                          <th>t прогнозу, c</th>
                        </tr>
                      </thead>
                      <tbody>
                        {families.map((fam) => {
                          const r = (famRes as any)[fam] || {};
                          const maseVal =
                            typeof r.mase === "number" ? r.mase.toFixed(3) : "";
                          const smapeVal =
                            typeof r.smape === "number"
                              ? r.smape.toFixed(2) + " %"
                              : "";
                          const rmseVal =
                            typeof r.rmse === "number"
                              ? r.rmse.toFixed(3)
                              : "";
                          const fitTimeVal =
                            typeof r.fit_time === "number"
                              ? r.fit_time.toFixed(3)
                              : "";
                          const predTimeVal =
                            typeof r.pred_time === "number"
                              ? r.pred_time.toFixed(3)
                              : "";

                          return (
                            <tr key={fam}>
                              <td>
                                {fam === "SARIMA"
                                  ? "SARIMA (наш метод)"
                                  : fam}
                              </td>
                              <td>{maseVal}</td>
                              <td>{smapeVal}</td>
                              <td>{rmseVal}</td>
                              <td>{fitTimeVal}</td>
                              <td>{predTimeVal}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </details>
                );
              })}
            </div>
          );
        })}
      </section>
    </div>
  );
};
