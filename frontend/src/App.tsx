import React from "react";
import HeatmapPanel from "./ui/HeatmapPanel";
import CorrLagPanel from "./ui/CorrLagPanel";
import SarimaxPanel from "./ui/SarimaxPanel";
import type { SeriesIn } from "./types";

// Припустимо, що series вже приходять з SeriesBrowser/SeriesPanel
export default function App() {
  const [series, setSeries] = React.useState<SeriesIn[]>([]);
  const [targetCode, setTargetCode] = React.useState<string>("");
  const [features, setFeatures] = React.useState<string[]>([]);
  const [lags, setLags] = React.useState<Record<string, number>>({});

  return (
    <div className="container mx-auto p-4 space-y-4">
      {/* ...тут твій існуючий вибір серій/цілі ... */}

      <div className="grid md:grid-cols-2 gap-4">
        <div className="border rounded">
          <div className="p-2 font-semibold">Heatmap</div>
          <HeatmapPanel series={series} targetCode={targetCode} />
        </div>

        <div className="border rounded">
          <div className="p-2 font-semibold">Corr Lag</div>
          <CorrLagPanel series={series} />
        </div>
      </div>

      <div className="border rounded">
        <div className="p-2 font-semibold">SARIMAX</div>
        <SarimaxPanel
          series={series}
          targetCode={targetCode}
          selectedFeatures={features}
          lags={lags}
        />
      </div>
    </div>
  );
}
