// src/ui/ExperimentRunForm.tsx
import React, { useState } from "react";
import Papa from "papaparse";
import {
  Frequency,
  Imputation,
  RunExperimentPayload,
  RunExperimentResponse,
  SeriesRole,
  runExperiment,
} from "../api/experiments";

type Props = {
  onExperimentCompleted?: (experimentId: string) => void;
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ParsedRow = Record<string, any>;

const defaultImputation: Imputation = "ffill";

export const ExperimentRunForm: React.FC<Props> = ({
  onExperimentCompleted,
}) => {
  const [fileName, setFileName] = useState<string>("");
  const [rows, setRows] = useState<ParsedRow[]>([]);
  const [columns, setColumns] = useState<string[]>([]);
  const [dateColumn, setDateColumn] = useState<string>("");
  const [columnRoles, setColumnRoles] = useState<Record<string, SeriesRole>>(
    {}
  );
  const [frequency, setFrequency] = useState<Frequency>("M");
  const [horizon, setHorizon] = useState<number>(12);
  const [imputation, setImputation] = useState<Imputation>(defaultImputation);
  const [maxLag, setMaxLag] = useState<number>(12);
  const [experimentName, setExperimentName] = useState<string>(""); // можна автозаповнювати з fileName
  const [context, setContext] = useState<string>("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RunExperimentResponse | null>(null);

  const handleFileChange: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setExperimentName((prev) => prev || file.name);

    Papa.parse<ParsedRow>(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      complete: (res: any) => {
        const data = res.data as ParsedRow[];
        const fields = (res.meta.fields || []) as string[];

        if (!data.length || !fields.length) {
          setError("Не вдалось прочитати дані з CSV");
          return;
        }

        setRows(data);
        setColumns(fields);
        setError(null);

        // перша колонка — дата за замовчуванням
        setDateColumn(fields[0]);

        // ролі: перша після дати — target, решта — candidate
        const roles: Record<string, SeriesRole> = {};
        fields.forEach((col, idx) => {
          if (idx === 0) return; // дата
          roles[col] = idx === 1 ? "target" : "candidate";
        });
        setColumnRoles(roles);
      },
    });
  };

  const handleRoleChange = (col: string, role: SeriesRole) => {
    setColumnRoles((prev) => ({
      ...prev,
      [col]: role,
    }));
  };

  const buildPayload = (): RunExperimentPayload => {
    if (!dateColumn) {
      throw new Error("Не обрано колонку дати");
    }
    if (!rows.length) {
      throw new Error("Немає даних (завантаж CSV файл)");
    }

    const dates = rows.map((r) => String(r[dateColumn]));

    const series = columns
      .filter((col) => col !== dateColumn)
      .map((col) => {
        const role = columnRoles[col] ?? "candidate";
        const values = rows.map((r) => {
          const v = r[col];
          if (v === "" || v === null || v === undefined) return null;
          const num = Number(v);
          return Number.isFinite(num) ? num : null;
        });

        return {
          name: col,
          role,
          values,
        };
      });

    const payload: RunExperimentPayload = {
      name: experimentName || fileName || "Experiment",
      context: context || undefined,
      dates,
      series,
      frequency,
      horizon,
      imputation,
      maxLag,
    };

    return payload;
  };

  const handleSubmit: React.FormEventHandler = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);

    try {
      const payload = buildPayload();
      setLoading(true);

      const res = await runExperiment(payload);
      setResult(res);

      const experimentId = res.experiment?.id;
      if (experimentId && onExperimentCompleted) {
        onExperimentCompleted(experimentId);
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (err: any) {
      console.error(err);
      setError(err?.message || "Помилка запуску експерименту");
    } finally {
      setLoading(false);
    }
  };

  const targetCount = Object.values(columnRoles).filter(
    (r) => r === "target"
  ).length;

  return (
    <div style={{ padding: "1rem", border: "1px solid #ddd" }}>
      <h2>Новий експеримент</h2>

      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: "0.75rem" }}>
          <label>
            CSV файл:{" "}
            <input type="file" accept=".csv" onChange={handleFileChange} />
          </label>
          {fileName && <div>Файл: {fileName}</div>}
        </div>

        {columns.length > 0 && (
          <>
            <div style={{ marginBottom: "0.75rem" }}>
              <label>
                Назва експерименту:{" "}
                <input
                  type="text"
                  value={experimentName}
                  onChange={(e) => setExperimentName(e.target.value)}
                  style={{ width: "280px" }}
                />
              </label>
            </div>

            <div style={{ marginBottom: "0.75rem" }}>
              <label>
                Контекст (країна / опис):{" "}
                <input
                  type="text"
                  value={context}
                  onChange={(e) => setContext(e.target.value)}
                  style={{ width: "280px" }}
                />
              </label>
            </div>

            <div style={{ marginBottom: "0.75rem" }}>
              <label>
                Колонка дати:{" "}
                <select
                  value={dateColumn}
                  onChange={(e) => setDateColumn(e.target.value)}
                >
                  {columns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div style={{ marginBottom: "0.75rem" }}>
              <label>
                Частота:{" "}
                <select
                  value={frequency}
                  onChange={(e) => setFrequency(e.target.value as Frequency)}
                >
                  <option value="M">Місячна</option>
                  <option value="Q">Квартальна</option>
                  <option value="Y">Річна</option>
                </select>
              </label>

              <label style={{ marginLeft: "1rem" }}>
                Горизонт прогнозу:{" "}
                <input
                  type="number"
                  value={horizon}
                  min={1}
                  onChange={(e) =>
                    setHorizon(parseInt(e.target.value, 10) || 1)
                  }
                  style={{ width: "80px" }}
                />
              </label>

              <label style={{ marginLeft: "1rem" }}>
                maxLag для кореляцій:{" "}
                <input
                  type="number"
                  value={maxLag}
                  min={0}
                  onChange={(e) => setMaxLag(parseInt(e.target.value, 10) || 0)}
                  style={{ width: "80px" }}
                />
              </label>

              <label style={{ marginLeft: "1rem" }}>
                Імпутація:{" "}
                <select
                  value={imputation}
                  onChange={(e) => setImputation(e.target.value as Imputation)}
                >
                  <option value="ffill">forward-fill</option>
                  <option value="bfill">backward-fill</option>
                  <option value="interp">interpolate</option>
                  <option value="none">без імпутації</option>
                </select>
              </label>
            </div>

            <div style={{ marginBottom: "0.75rem" }}>
              <strong>Колонки та ролі (target / candidate / ignored)</strong>
              <table style={{ width: "100%", marginTop: "0.5rem" }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left" }}>Колонка</th>
                    <th>Роль</th>
                  </tr>
                </thead>
                <tbody>
                  {columns
                    .filter((col) => col !== dateColumn)
                    .map((col) => (
                      <tr key={col}>
                        <td>{col}</td>
                        <td>
                          <select
                            value={columnRoles[col] ?? "candidate"}
                            onChange={(e) =>
                              handleRoleChange(
                                col,
                                e.target.value as SeriesRole
                              )
                            }
                          >
                            <option value="target">Target</option>
                            <option value="candidate">Candidate</option>
                            <option value="ignored">Ignored</option>
                          </select>
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
              <div style={{ marginTop: "0.25rem", fontSize: "0.9rem" }}>
                Target-рядів: {targetCount}. Мінімум один має бути.
              </div>
            </div>
          </>
        )}

        {error && (
          <div style={{ color: "red", marginBottom: "0.75rem" }}>{error}</div>
        )}

        <button
          type="submit"
          disabled={loading || !rows.length}
          style={{ padding: "0.5rem 1rem" }}
        >
          {loading ? "Запуск..." : "Запустити експеримент"}
        </button>
      </form>

      {result && (
        <div style={{ marginTop: "1rem", fontSize: "0.9rem" }}>
          <div>
            Експеримент ID:{" "}
            <strong>{result.experiment?.id ?? "— (не збережено)"}</strong>
          </div>
          <div>
            Таргетні ряди:{" "}
            {Array.isArray(result.mlResult.diagnostics?.targets)
              ? result.mlResult.diagnostics.targets.join(", ")
              : "—"}
          </div>
        </div>
      )}
    </div>
  );
};
