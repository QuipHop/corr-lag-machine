import type { CorrLagRes, Point, SeriesMeta } from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:3000';


export async function fetchSeriesMeta(id: number): Promise<SeriesMeta> {
  const r = await fetch(`${BASE}/series/${id}`);
  if (!r.ok) throw new Error(`Failed to load series ${id}`);
  return r.json();
}


export async function fetchSeriesData(id: number, from?: string, to?: string): Promise<{ seriesId: number; points: Point[] }> {
  const u = new URL(`${BASE}/series/${id}/data`);
  if (from) u.searchParams.set('from', from);
  if (to) u.searchParams.set('to', to);
  const r = await fetch(u);
  if (!r.ok) throw new Error(`Failed to load series data ${id}`);
  return r.json();
}


export async function corrLag
  (seriesIds: number[], params: { maxLag?: number; method?: 'pearson' | 'spearman'; minOverlap?: number; edgeMin?: number } = {}): Promise<CorrLagRes> {
  const body = JSON.stringify({ seriesIds, ...params });
  const r = await fetch(`${BASE}/ml/corr-lag`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body });
  if (!r.ok) throw new Error('corr-lag failed');
  return r.json();
}