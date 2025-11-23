// src/App.tsx
import React, { useEffect, useState } from "react";
import "./App.css";
import { ExperimentRunForm } from "./ui/ExperimentRunForm";
import { ExperimentDetails } from "./ui/ExperimentDetails";
import { ExperimentWithMetrics, listExperiments } from "./api/experiments";

const App: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentWithMetrics[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loadingList, setLoadingList] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadList = async () => {
    setLoadingList(true);
    setError(null);
    try {
      const data = await listExperiments();
      setExperiments(data);
      if (!selectedId && data.length > 0) {
        setSelectedId(data[0].id);
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (err: any) {
      console.error(err);
      setError(err?.message || "Помилка завантаження експериментів");
    } finally {
      setLoadingList(false);
    }
  };

  useEffect(() => {
    void loadList();
  }, []);

  const handleCompleted = (experimentId: string) => {
    // оновлюємо список і виділяємо новий експеримент
    setSelectedId(experimentId);
    void loadList();
  };

  return (
    <div className="App" style={{ display: "flex", minHeight: "100vh" }}>
      <div style={{ flex: "0 0 420px", borderRight: "1px solid #ddd" }}>
        {/* <h1 style={{ padding: "0.5rem 1rem" }}>Prediction</h1> */}
        <ExperimentRunForm onExperimentCompleted={handleCompleted} />

        <div style={{ padding: "0.5rem 1rem" }}>
          <h3>Історія експериментів</h3>
          {loadingList && <div>Завантаження...</div>}
          {error && <div style={{ color: "red" }}>{error}</div>}
          <ul style={{ listStyle: "none", paddingLeft: 0 }}>
            {experiments.map((exp) => (
              <li
                key={exp.id}
                style={{
                  cursor: "pointer",
                  padding: "0.25rem 0.5rem",
                  backgroundColor:
                    exp.id === selectedId ? "#eef" : "transparent",
                }}
                onClick={() => setSelectedId(exp.id)}
              >
                <div style={{ fontWeight: 600 }}>{exp.name}</div>
                <div style={{ fontSize: "0.8rem" }}>
                  {exp.context || "—"} | {exp.frequency} | h=
                  {exp.horizon}
                </div>
                {exp.metrics[0] && (
                  <div style={{ fontSize: "0.8rem" }}>
                    {exp.metrics[0].seriesName}: MASE{" "}
                    {exp.metrics[0].mase.toFixed(3)}
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div style={{ flex: "1 1 auto" }}>
        {selectedId ? (
          <ExperimentDetails experimentId={selectedId} />
        ) : (
          <div style={{ padding: "1rem" }}>
            Запусти перший експеримент або обери його зліва.
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
