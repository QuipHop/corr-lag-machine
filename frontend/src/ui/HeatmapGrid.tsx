import React from "react";

type Row = { key: string; values: Record<string, number | null> };
type Props = { rows: Row[] };

export default function HeatmapGrid({ rows }: Props) {
    if (!rows?.length) return null;

    const lags = Array.from(
        new Set(
            rows.flatMap(r => Object.keys(r.values).map(k => Number(k))).sort((a, b) => a - b)
        )
    );

    // нор­ма­лі­за­ція до [0..1] за |corr|
    const flat = rows.flatMap(r => lags.map(l => Math.abs(r.values[String(l)] ?? 0)));
    const max = Math.max(0.0001, ...flat);

    return (
        <div className="overflow-auto border rounded">
            <table className="text-xs min-w-full border-collapse">
                <thead>
                    <tr>
                        <th className="sticky left-0 bg-white p-2 border">Series</th>
                        {lags.map(l => (
                            <th key={l} className="p-1 border text-center">{l}</th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {rows.map(r => (
                        <tr key={r.key}>
                            <td className="sticky left-0 bg-white p-1 border font-medium">{r.key}</td>
                            {lags.map(l => {
                                const v = r.values[String(l)];
                                const a = v == null ? 0 : Math.abs(v) / max;
                                // синій = +, червоний = -
                                const hue = v == null ? 0 : (v >= 0 ? 210 : 0);
                                const alpha = v == null ? 0 : Math.min(1, 0.1 + a * 0.9);
                                const bg = v == null ? "transparent" : `hsla(${hue}, 70%, 50%, ${alpha})`;
                                return (
                                    <td
                                        key={l}
                                        title={v == null ? "NA" : v.toFixed(4)}
                                        className="w-8 h-6 border text-center align-middle"
                                        style={{ background: bg }}
                                    />
                                );
                            })}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
