import { BadRequestException, Injectable } from '@nestjs/common';
import { PrismaService } from '../shared/prisma.service';

function applyTransform(arr: number[], type: string) {
    switch (type) {
        case 'pct_change': return arr.slice(1).map((v, i) => 100 * (v / arr[i] - 1));
        case 'diff': return arr.slice(1).map((v, i) => v - arr[i]);
        case 'log': if (arr.some(v => v <= 0)) throw new Error('log requires positive'); return arr.map(Math.log);
        case 'zscore': {
            const m = arr.reduce((a, b) => a + b, 0) / arr.length;
            const sd = Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length - 1));
            return arr.map(v => (v - m) / (sd || 1));
        }
        default: return arr;
    }
}
const rank = (a: number[]) => {
    const w = a.map((v, i) => ({ v, i })).sort((x, y) => x.v - y.v);
    const r = Array(a.length).fill(0);
    for (let i = 0; i < w.length; i++) { let j = i; while (j + 1 < w.length && w[j + 1].v === w[i].v) j++; const avg = (i + j + 2) / 2; for (let k = i; k <= j; k++) r[w[k].i] = avg; i = j; }
    return r;
};
const pearson = (x: number[], y: number[]) => {
    const n = x.length, mx = x.reduce((a, b) => a + b, 0) / n, my = y.reduce((a, b) => a + b, 0) / n;
    let num = 0, sx = 0, sy = 0; for (let i = 0; i < n; i++) { const dx = x[i] - mx, dy = y[i] - my; num += dx * dy; sx += dx * dx; sy += dy * dy; }
    return num / Math.sqrt((sx || 1) * (sy || 1));
};
const spearman = (x: number[], y: number[]) => pearson(rank(x), rank(y));

@Injectable()
export class CorrelationService {
    constructor(private prisma: PrismaService) { }

    async run(dto: {
        datasetId: string;
        series: string[];
        method?: 'spearman' | 'pearson';
        transforms?: Record<string, { type: string }>;
        pearsonAlso?: boolean;
    }) {
        const series = await this.prisma.series.findMany({
            where: { datasetId: dto.datasetId, key: { in: dto.series } },
            include: { observations: true },
        });
        if (series.length < 2) throw new BadRequestException('need >=2 series');

        const maps = series.map(s => new Map(
            s.observations.map(o => [o.date.toISOString().slice(0, 10), Number(o.value)])
        ));
        const dates = [...maps[0].keys()].filter(d => maps.every(m => m.has(d))).sort();
        if (dates.length < 6) throw new BadRequestException('too few overlapping points');

        let data = series.map((_, i) => dates.map(d => maps[i].get(d)!));
        data = data.map((arr, i) => {
            const key = series[i].key;
            const t = dto.transforms?.[key]?.type ?? 'none';
            return applyTransform(arr, t);
        });

        const n = Math.min(...data.map(a => a.length));
        if (n < 6) throw new BadRequestException('too few points after transforms');

        const tail = (a: number[]) => a.slice(-n);
        const out: any[] = [];
        for (let i = 0; i < series.length; i++) {
            for (let j = i + 1; j < series.length; j++) {
                const xi = tail(data[i]), yj = tail(data[j]);
                const rS = dto.method === 'pearson' ? null : spearman(xi, yj);
                const rP = dto.pearsonAlso || dto.method === 'pearson' ? pearson(xi, yj) : null;
                out.push({ x: series[i].key, y: series[j].key, n, spearman: rS, pearson: rP, overlapFrom: dates[dates.length - n], overlapTo: dates[dates.length - 1] });
            }
        }
        return { freq: 'M', pairs: out };
    }
}
