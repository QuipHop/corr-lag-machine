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

  const diag = (experiment?.diagnostics || {}) as any;
  const seriesDiag = (diag?.series || {}) as Record<string, any>;
  const targetsInfo: Record<string, any> = (diag?.targets ||
    {}) as Record<string, any>;
  const targetsExog: Record<string, any[]> = (diag?.targets_exog ||
    {}) as Record<string, any[]>;
  const targetNames: string[] = Object.keys(targetsInfo);
  const baseVars: string[] = (diag?.base_variables || []) as string[];
  const comparison = (diag?.comparison || {}) as any;

  const correlations = (experiment?.correlations || {}) as any;
  const factors = (experiment?.factors || {}) as any;

  const selectedModels = experiment?.models?.filter((m) => m.isSelected) ?? [];

  const selectedBySeries = useMemo(() => {
    const map: Record<string, (typeof selectedModels)[number]> = {};
    for (const m of selectedModels) {
      map[m.seriesName] = m;
    }
    return map;
  }, [selectedModels]);

  if (loading) {
    return (
      <div
        style={{
          padding: "1rem",
          backgroundColor: "#ffffff",
          color: "#000000",
        }}
      >
        Завантаження...
      </div>
    );
  }

  if (error) {
    return (
      <div
        style={{
          padding: "1rem",
          backgroundColor: "#ffffff",
          color: "red",
        }}
      >
        {error}
      </div>
    );
  }

  if (!experiment) {
    return (
      <div
        style={{
          padding: "1rem",
          backgroundColor: "#ffffff",
          color: "#000000",
        }}
      >
        Обери експеримент, щоб побачити деталі.
      </div>
    );
  }

  const meta = diag?.meta;

  const containerStyle: React.CSSProperties = {
    padding: "1rem",
    backgroundColor: "#ffffff",
    color: "#000000",
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
    fontSize: "14px",
    lineHeight: 1.4,
  };

  const tableStyle: React.CSSProperties = {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: "0.9rem",
  };

  const thStyle: React.CSSProperties = {
    borderBottom: "1px solid #000000",
    textAlign: "left",
    padding: "4px 6px",
  };

  const tdStyle: React.CSSProperties = {
    borderBottom: "1px solid #dddddd",
    padding: "4px 6px",
  };

  const sectionStyle: React.CSSProperties = { marginTop: "1.5rem" };

  return (
    <div style={containerStyle}>
      <h2 style={{ marginTop: 0 }}>
        Експеримент: {experiment.name || experiment.id}
      </h2>
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
      <section style={sectionStyle}>
        <h3>1. Діагностика рядів</h3>
        <table style={tableStyle}>
          <thead>
            <tr>
              <th style={thStyle}>Ряд</th>
              <th style={thStyle}>Роль</th>
              <th style={thStyle}>Mean</th>
              <th style={thStyle}>Std</th>
              <th style={thStyle}>ADF p</th>
              <th style={thStyle}>ADF стац.</th>
              <th style={thStyle}>KPSS p</th>
              <th style={thStyle}>KPSS стац.</th>
              <th style={thStyle}>Сезонність</th>
              <th style={thStyle}>ACF(12)</th>
              <th style={thStyle}>Трансформація</th>
              <th style={thStyle}>Нелінійність</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(seriesDiag).map(([name, info]) => {
              const isTarget = targetNames.includes(name);
              const isBase = baseVars.includes(name);

              const roleLabel = isTarget
                ? "цільовий ряд"
                : isBase
                  ? "базовий ряд"
                  : "кандидат";

              const adf_p: number | null =
                typeof info.adf_p === "number" ? info.adf_p : null;
              const kpss_p: number | null =
                typeof info.kpss_p === "number" ? info.kpss_p : null;
              const hasSeasonality = !!info.has_seasonality;
              const acf12: number | null =
                typeof info.acf_12 === "number" ? info.acf_12 : null;

              return (
                <tr key={name}>
                  <td style={tdStyle}>{name}</td>
                  <td style={tdStyle}>{roleLabel}</td>
                  <td style={tdStyle}>
                    {typeof info.mean === "number"
                      ? info.mean.toFixed(2)
                      : ""}
                  </td>
                  <td style={tdStyle}>
                    {typeof info.std === "number" ? info.std.toFixed(2) : ""}
                  </td>
                  <td style={tdStyle}>
                    {adf_p != null ? adf_p.toFixed(4) : ""}
                  </td>
                  <td style={tdStyle}>
                    {adf_p != null ? (adf_p < 0.05 ? "так" : "ні") : ""}
                  </td>
                  <td style={tdStyle}>
                    {kpss_p != null ? kpss_p.toFixed(4) : ""}
                  </td>
                  <td style={tdStyle}>
                    {kpss_p != null ? (kpss_p > 0.05 ? "так" : "ні") : ""}
                  </td>
                  <td style={tdStyle}>{hasSeasonality ? "є" : "нема"}</td>
                  <td style={tdStyle}>
                    {acf12 != null ? acf12.toFixed(3) : ""}
                  </td>
                  <td style={tdStyle}>{info.transform ?? ""}</td>
                  <td style={tdStyle}>
                    {info.is_nonlinear ? "так" : "ні"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      {/* 2. Кореляційний аналіз і лаги */}
      <section style={sectionStyle}>
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
              <table style={{ ...tableStyle, fontSize: "0.85rem" }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Предиктор</th>
                    <th style={thStyle}>Lag</th>
                    <th style={thStyle}>r</th>
                  </tr>
                </thead>
                <tbody>
                  {rel.map((e, idx) => (
                    <tr key={idx}>
                      <td style={tdStyle}>{e.source}</td>
                      <td style={tdStyle}>{e.best_lag}</td>
                      <td style={tdStyle}>
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

      {/* 3. Базові змінні та факторний аналіз */}
      <section style={sectionStyle}>
        <h3>3. Базові змінні та факторний аналіз</h3>
        <div style={{ fontSize: "0.9rem", marginBottom: "0.5rem" }}>
          <strong>Обрані базові змінні:</strong>{" "}
          {baseVars.length ? baseVars.join(", ") : "—"}
        </div>

        {factors?.vif && (
          <div style={{ marginBottom: "0.75rem" }}>
            <strong>VIF (мультиколінеарність)</strong>
            <table style={{ ...tableStyle, fontSize: "0.85rem" }}>
              <thead>
                <tr>
                  <th style={thStyle}>Змінна</th>
                  <th style={thStyle}>VIF</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(factors.vif).map(([name, v]: [string, any]) => (
                  <tr key={name}>
                    <td style={tdStyle}>{name}</td>
                    <td style={tdStyle}>
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
      <section style={sectionStyle}>
        <h3>4. Вибір моделей (обрані за методом)</h3>
        <table style={tableStyle}>
          <thead>
            <tr>
              <th style={thStyle}>Ряд</th>
              <th style={thStyle}>Роль</th>
              <th style={thStyle}>Модель</th>
              <th style={thStyle}>MASE</th>
              <th style={thStyle}>sMAPE</th>
              <th style={thStyle}>RMSE</th>
            </tr>
          </thead>
          <tbody>
            {selectedModels.map((m) => {
              const isTarget = targetNames.includes(m.seriesName);
              const isBase = baseVars.includes(m.seriesName);
              const roleLabel = isTarget
                ? "цільовий ряд"
                : isBase
                  ? "базовий ряд"
                  : "кандидат";

              return (
                <tr key={`${m.seriesName}-${m.modelType}`}>
                  <td style={tdStyle}>{m.seriesName}</td>
                  <td style={tdStyle}>{roleLabel}</td>
                  <td style={tdStyle}>{m.modelType}</td>
                  <td style={tdStyle}>
                    {m.mase != null ? m.mase.toFixed(3) : ""}
                  </td>
                  <td style={tdStyle}>
                    {m.smape != null ? m.smape.toFixed(1) + " %" : ""}
                  </td>
                  <td style={tdStyle}>
                    {m.rmse != null ? m.rmse.toFixed(3) : ""}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      {/* 5. Оцінка ефективності комбінованого способу */}
      <section style={sectionStyle}>
        <h3>5. Оцінка ефективності комбінованого способу (таргетні ряди)</h3>
        {targetNames.length === 0 ? (
          <div style={{ fontSize: "0.9rem" }}>
            Таргетні ряди не визначені.
          </div>
        ) : (
          <table style={{ ...tableStyle, fontSize: "0.85rem" }}>
            <thead>
              <tr>
                <th style={thStyle}>Ряд</th>
                <th style={thStyle}>Обрана модель (наш метод)</th>
                <th style={thStyle}>MASE</th>
                <th style={thStyle}>sMAPE</th>
                <th style={thStyle}>RMSE</th>
                <th style={thStyle}>p-value Ljung–Box</th>
                <th style={thStyle}>Залишки ок?</th>
              </tr>
            </thead>
            <tbody>
              {targetNames.map((t) => {
                const sel = selectedBySeries[t];
                const tDiag = targetsInfo[t] || {};
                const lb = tDiag.lb_pvalue;
                const residOk = tDiag.residuals_ok;

                if (!sel) return null;

                return (
                  <tr key={t}>
                    <td style={tdStyle}>{t}</td>
                    <td style={tdStyle}>{sel.modelType}</td>
                    <td style={tdStyle}>
                      {sel.mase != null ? sel.mase.toFixed(3) : ""}
                    </td>
                    <td style={tdStyle}>
                      {sel.smape != null ? sel.smape.toFixed(1) + " %" : ""}
                    </td>
                    <td style={tdStyle}>
                      {sel.rmse != null ? sel.rmse.toFixed(3) : ""}
                    </td>
                    <td style={tdStyle}>
                      {typeof lb === "number" ? lb.toFixed(3) : ""}
                    </td>
                    <td style={tdStyle}>
                      {typeof residOk === "boolean"
                        ? residOk
                          ? "так"
                          : "ні"
                        : ""}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}

        {/* Пояснення, які базові змінні реально зайшли в exog */}
        {targetNames.map((t) => {
          const exogList = targetsExog[t] || [];
          if (!exogList.length) return null;
          return (
            <div key={t} style={{ marginTop: "0.5rem", fontSize: "0.85rem" }}>
              <strong>
                Для ряду {t} в комбінованому способі використано exogenous:
              </strong>{" "}
              {exogList
                .map((e) => {
                  const base = e.base ?? e.base_name ?? "?";
                  const lag = typeof e.lag === "number" ? e.lag : 0;
                  return `${base}${lag > 0 ? ` (lag ${lag})` : ""}`;
                })
                .join(", ")}
            </div>
          );
        })}
      </section>

      {/* 6. Прогнози (backtest + future) */}
      <section style={sectionStyle}>
        <h3>6. Прогнози</h3>
        {targetNames.length === 0 && (
          <div>Таргетні ряди не визначені в diagnostics.</div>
        )}

        {targetNames.map((seriesName) => {
          const seriesForecasts = (forecastsBySeries[seriesName] || []).filter(
            (f) =>
              f.setType === "train" ||
              f.setType === "test" ||
              f.setType === "future"
          );
          if (!seriesForecasts.length) return null;

          return (
            <details key={seriesName} style={{ marginBottom: "1rem" }} open>
              <summary>{seriesName}</summary>

              <div
                style={{
                  marginTop: "0.5rem",
                  marginBottom: "0.5rem",
                  border: "1px solid #000000",
                  padding: "4px",
                }}
              >
                <svg
                  viewBox="0 0 400 200"
                  preserveAspectRatio="none"
                  style={{
                    width: "100%",
                    height: "160px",
                    background: "#ffffff",
                  }}
                >
                  {(() => {
                    const points = seriesForecasts;
                    if (!points.length) return null;
                    const xs = points.map((_, i) => i);
                    const ysActual = points
                      .map((p) => p.valueActual)
                      .filter((v) => v != null) as number[];
                    const ysPred = points
                      .map((p) => p.valuePred)
                      .filter((v) => v != null) as number[];

                    if (!ysPred.length) return null;

                    const allY = [...ysActual, ...ysPred];
                    const yMin = Math.min(...allY);
                    const yMax = Math.max(...allY);
                    const padY = (yMax - yMin) * 0.05 || 1;
                    const ymin = yMin - padY;
                    const ymax = yMax + padY;
                    const xMin = 0;
                    const xMax = xs.length - 1;

                    const scaleX = (i: number) =>
                      xMax === xMin
                        ? 0
                        : ((i - xMin) / (xMax - xMin)) * 400;
                    const scaleY = (v: number) =>
                      ymax === ymin
                        ? 100
                        : 200 - ((v - ymin) / (ymax - ymin)) * 200;

                    const line = (vals: (number | null | undefined)[]) => {
                      const coords: string[] = [];
                      vals.forEach((v, idx) => {
                        if (v == null) return;
                        const x = scaleX(idx);
                        const y = scaleY(v);
                        coords.push(`${x},${y}`);
                      });
                      return coords.join(" ");
                    };

                    const actualLine = line(
                      points.map((p) => p.valueActual ?? null)
                    );

                    const predLine = line(
                      points.map((p) => p.valuePred ?? null)
                    );

                    const firstFutureIdx = points.findIndex(
                      (p) => p.setType === "future"
                    );
                    const futureX =
                      firstFutureIdx >= 0 ? scaleX(firstFutureIdx) : null;

                    return (
                      <>
                        {futureX !== null && (
                          <line
                            x1={futureX}
                            y1={0}
                            x2={futureX}
                            y2={200}
                            stroke="#000000"
                            strokeWidth={1}
                            strokeDasharray="3 3"
                          />
                        )}
                        <polyline
                          points={actualLine}
                          fill="none"
                          stroke="#000000"
                          strokeWidth={1.5}
                        />
                        <polyline
                          points={predLine}
                          fill="none"
                          stroke="#000000"
                          strokeWidth={1.5}
                          strokeDasharray="4 2"
                        />
                      </>
                    );
                  })()}
                </svg>
              </div>

              <table
                style={{
                  ...tableStyle,
                  fontSize: "0.85rem",
                  marginTop: "0.5rem",
                }}
              >
                <thead>
                  <tr>
                    <th style={thStyle}>Дата</th>
                    <th style={thStyle}>Набір</th>
                    <th style={thStyle}>Факт</th>
                    <th style={thStyle}>Прогноз</th>
                  </tr>
                </thead>
                <tbody>
                  {seriesForecasts.slice(-12).map((f) => (
                    <tr key={`${f.seriesName}-${f.date}-${f.setType}`}>
                      <td style={tdStyle}>{f.date.slice(0, 10)}</td>
                      <td style={tdStyle}>{f.setType}</td>
                      <td style={tdStyle}>
                        {f.valueActual != null
                          ? f.valueActual.toFixed(3)
                          : "—"}
                      </td>
                      <td style={tdStyle}>
                        {f.valuePred != null ? f.valuePred.toFixed(3) : "—"}
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
      <section style={sectionStyle}>
        <h3>7. Порівняння моделей (горизонти 1–3 місяці)</h3>

        {targetNames.map((t) => {
          const cmp = comparison ? (comparison as any)[t] : null;
          if (!cmp) return null;

          const horizons = Object.keys(cmp)
            .map((h) => Number(h))
            .sort((a, b) => a - b);
          if (!horizons.length) return null;

          const selected = selectedBySeries[t];

          // Мапимо modelType → ключ у comparison
          const selectedFamilyKey: string | null = (() => {
            if (!selected || !selected.modelType) return null;
            const mt = selected.modelType;
            if (mt === "GBR" || mt === "GB") return "GB";
            if (mt === "RF") return "RF";
            if (mt === "SARIMAX") return "SARIMAX";
            if (mt === "SARIMA") return "SARIMA";
            if (mt === "ARIMA") return "ARIMA";
            return mt;
          })();

          return (
            <div key={t} style={{ marginBottom: "1rem" }}>
              <h4>{t}</h4>

              {selected && (
                <div style={{ fontSize: "0.85rem", marginBottom: "0.25rem" }}>
                  Наш метод (пайплайн):{" "}
                  <strong>{selected.modelType}</strong>
                </div>
              )}

              {horizons.map((h) => {
                const famRes = (cmp as any)[h];
                if (!famRes) return null;

                const familyKeys = Object.keys(famRes);
                if (!familyKeys.length) return null;

                // стабільний порядок виводу
                const orderedFamilies = [
                  "SeasonalNaive",
                  "ARIMA",
                  "SARIMA",
                  "GB",
                  "RF",
                  "SARIMAX",
                  "OUR_METHOD",
                  ...familyKeys,
                ].filter(
                  (v, idx, arr) => familyKeys.includes(v) && arr.indexOf(v) === idx
                );

                // для обчислення % поліпшення по MASE
                const selectedMetrics =
                  selectedFamilyKey && famRes[selectedFamilyKey]
                    ? famRes[selectedFamilyKey]
                    : null;

                const improvements: string[] = [];

                if (
                  selectedFamilyKey &&
                  selectedMetrics &&
                  typeof selectedMetrics.mase === "number" &&
                  isFinite(selectedMetrics.mase)
                ) {
                  const maseOur = selectedMetrics.mase;
                  const smapeOur = selectedMetrics.smape; // Беремо наш sMAPE

                  for (const fam of orderedFamilies) {
                    if (fam === selectedFamilyKey) continue;
                    const rOther = famRes[fam];

                    if (!rOther) continue;

                    let textParts: string[] = [];
                    let hasData = false;

                    // --- 1. Порівняння MASE ---
                    if (
                      typeof rOther.mase === "number" &&
                      isFinite(rOther.mase) &&
                      rOther.mase > 0
                    ) {
                      // Формула: (Чужий - Наш) / Чужий * 100
                      // Якщо результат > 0, значить Наш менший (кращий) -> "+"
                      const impMase =
                        ((rOther.mase - maseOur) / rOther.mase) * 100.0;
                      const sign = impMase > 0 ? "+" : "";
                      textParts.push(`MASE ${sign}${impMase.toFixed(1)}%`);
                      hasData = true;
                    }

                    // --- 2. Порівняння sMAPE ---
                    if (
                      typeof rOther.smape === "number" &&
                      typeof smapeOur === "number" &&
                      isFinite(rOther.smape) &&
                      rOther.smape > 0
                    ) {
                      const impSmape =
                        ((rOther.smape - smapeOur) / rOther.smape) * 100.0;
                      const sign = impSmape > 0 ? "+" : "";
                      textParts.push(`sMAPE ${sign}${impSmape.toFixed(1)}%`);
                      hasData = true;
                    }

                    if (hasData) {
                      improvements.push(`${fam}: ${textParts.join(", ")}`);
                    }
                  }
                }

                return (
                  <details
                    key={h}
                    style={{ marginBottom: "0.5rem" }}
                    open={h === 1}
                  >
                    <summary>Горизонт {h} місяць(і)</summary>
                    <table
                      style={{
                        ...tableStyle,
                        fontSize: "0.85rem",
                        marginTop: "0.5rem",
                      }}
                    >
                      <thead>
                        <tr>
                          <th style={thStyle}>Модель</th>
                          <th style={thStyle}>MASE</th>
                          <th style={thStyle}>sMAPE</th>
                          <th style={thStyle}>RMSE</th>
                          <th style={thStyle}>t навчання, c</th>
                          <th style={thStyle}>t прогнозу, c</th>
                        </tr>
                      </thead>
                      <tbody>
                        {orderedFamilies.map((fam) => {
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

                          const isOur =
                            selectedFamilyKey &&
                            fam === selectedFamilyKey;

                          let label = fam;
                          if (isOur) {
                            label = `Наш метод (пайплайн, сімейство: ${fam})`;
                          }

                          return (
                            <tr key={fam}>
                              <td style={tdStyle}>{label}</td>
                              <td style={tdStyle}>{maseVal}</td>
                              <td style={tdStyle}>{smapeVal}</td>
                              <td style={tdStyle}>{rmseVal}</td>
                              <td style={tdStyle}>{fitTimeVal}</td>
                              <td style={tdStyle}>{predTimeVal}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>

                    {improvements.length > 0 && (
                      <div
                        style={{
                          fontSize: "0.8rem",
                          marginTop: "0.25rem",
                          fontStyle: "italic",
                        }}
                      >
                        Поліпшення нашого методу за MASE:&nbsp;
                        {improvements.join("; ")}.
                      </div>
                    )}
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
