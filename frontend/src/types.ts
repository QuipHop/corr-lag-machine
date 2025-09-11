export type Point = { date: string; value: number | string };
export type SeriesMeta = {
  id: number;
  indicatorId: number;
  region: string | null;
  frequency: 'monthly' | 'quarterly' | 'annual';
  extraMeta: Record<string, unknown> | null;
  indicator: { id: number; code: string; name: string; unit?: string | null };
};


export type CorrLagEdge = { source: string; target: string; lag: number; weight: number };
export type CorrLagRes = { nodes: { id: string }[]; edges: CorrLagEdge[] };