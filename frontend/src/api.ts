const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:3000';
const API_PREFIX = '/api';

const full = (p: string) => `${API_BASE}${API_PREFIX}${p}`;

export async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(full(path));
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<T>;
}

export async function postJSON<T>(path: string, body: any, headers: Record<string, string> = {}): Promise<T> {
  const res = await fetch(full(path), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...headers },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<T>;
}

export async function postFile<T = any>(path: string, file: File): Promise<T> {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch(full(path), { method: 'POST', body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<T>;
}
