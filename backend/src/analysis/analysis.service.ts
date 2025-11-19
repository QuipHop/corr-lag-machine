import { Injectable, NotFoundException } from '@nestjs/common';
import { PrismaService } from '../shared/prisma.service';
import { MlService } from '../ml/ml.service';

type SeriesIn = { id: number; code: string; points: { date: string; value: number }[] };

@Injectable()
export class AnalysisService {
    constructor(private prisma: PrismaService, private ml: MlService) { }

    /**
     * Завантажити ряди з БД за datasetId та списком кодів (codes = Series.key),
     * перетворити у SeriesIn[], яке очікує ml-svc.
     * id -> послідовний індекс (1..N), code -> Series.key
     */
    // analysis.service.ts

    /** Толерантне завантаження серій: case-insensitive, trim */
    private async loadSeries(datasetId: string, codes: string[]): Promise<{
        found: SeriesIn[];
        missing: string[];
    }> {
        // 1) стягуємо всі серії датасету (id, key)
        const all = await this.prisma.series.findMany({
            where: { datasetId },
            select: { id: true, key: true },
        });

        // 2) робимо case-insensitive мапу: lower(trim(key)) -> { id, key }
        const norm = (s: string) => s.trim().toLowerCase();
        const byNorm = new Map<string, { id: string; key: string }>();
        for (const s of all) byNorm.set(norm(s.key), { id: s.id, key: s.key });

        // 3) зіставляємо запитані codes з наявними ключами
        const matched: { id: string; key: string }[] = [];
        const missing: string[] = [];
        for (const c of codes) {
            const hit = byNorm.get(norm(c));
            if (hit) matched.push(hit);
            else missing.push(c);
        }

        if (matched.length === 0) {
            return { found: [], missing };
        }

        // 4) підтягуємо спостереження для знайдених серій
        const seriesIds = matched.map(s => s.id);
        const obs = await this.prisma.observation.findMany({
            where: { seriesId: { in: seriesIds } },
            select: { seriesId: true, date: true, value: true },
            orderBy: { date: 'asc' },
        });

        // 5) seriesId -> key
        const idToKey = new Map<string, string>(matched.map(s => [s.id, s.key]));

        // 6) групуємо точки по key
        const byKey = new Map<string, { date: string; value: number }[]>();
        for (const row of obs) {
            const code = idToKey.get(row.seriesId);
            if (!code) continue;
            const arr = byKey.get(code) ?? [];
            arr.push({ date: row.date.toISOString().slice(0, 10), value: Number(row.value) });
            byKey.set(code, arr);
        }

        // 7) формуємо SeriesIn (id -> 1..N; code -> оригінальний key з БД)
        const found: SeriesIn[] = matched.map((s, idx) => ({
            id: idx + 1,
            code: s.key,
            points: byKey.get(s.key) ?? [],
        }));

        return { found, missing };
    }


    /** Heatmap з БД */
    async corrHeatmapFromDb(body: {
        datasetId: string;
        targetCode: string;               // Series.key
        candidateCodes?: string[];        // Series.key[]
        method?: 'pearson' | 'spearman';
        minOverlap?: number;
        lag?: { min?: number; max?: number; ignoreZero?: boolean };
        resample?: { enabled?: boolean; freq?: 'M'; downsample?: 'last' | 'mean' | 'sum'; upsample?: 'ffill' | 'bfill' | 'interpolate' | 'none'; winsorize_q?: number };
        topK?: number;
        transform?: 'none' | 'diff1' | 'pct';
        returnP?: boolean;
        fdrAlpha?: number;
    }) {
        const {
            datasetId, targetCode,
            candidateCodes,
            method = 'pearson',
            minOverlap = 12,
            lag = { min: -12, max: 12, ignoreZero: false },
            resample = { enabled: true, freq: 'M', downsample: 'last', upsample: 'ffill', winsorize_q: 0 },
            topK = 20,
            transform = 'none',
            returnP = true,
            fdrAlpha = 0.1,
        } = body;

        // якщо кандидати не передані — беремо всі key у датасеті, крім target
        const candidates = candidateCodes ?? (
            (await this.prisma.series.findMany({
                where: { datasetId },
                select: { key: true },
            })).map(x => x.key).filter(k => k !== targetCode)
        );

        const codes = [targetCode, ...candidates];
        const { found: series, missing } = await this.loadSeries(datasetId, codes);

        // якщо не знайшовся target — це помилка
        if (!series.some(s => s.code === targetCode)) {
            throw new NotFoundException(`Target '${targetCode}' not found in dataset ${datasetId}`);
        }
        // якщо взагалі немає кандидатів — теж помилка
        const foundCandidates = series.filter(s => s.code !== targetCode).map(s => s.code);
        if (foundCandidates.length === 0) {
            throw new NotFoundException(`No candidate series found in dataset ${datasetId}`);
        }


        const req = {
            series,
            targetCode,
            candidateCodes: candidates,
            method,
            minOverlap,
            resample,
            lag,
            topK,
            returnStats: false,
            transform,
            returnP,
            fdrAlpha,
        };

        const resp: any = await this.ml.corrHeatmap(req);
        return { ...resp, meta2: { missing, usedCandidates: foundCandidates } };

    }

    /** Backtest з БД */
    async sarimaxBacktestFromDb(body: {
        datasetId: string;
        targetCode: string;
        featureCodes?: string[];
        lags?: Record<string, number>;
        resample?: any;
        transform?: 'none' | 'diff1' | 'pct';
        train?: any;
        backtest: { horizon: number; min_train: number; step: number; expanding: boolean };
        saveRun?: boolean;
        runName?: string;
    }) {
        const {
            datasetId, targetCode,
            featureCodes = [],
            lags = {},
            resample = { enabled: true, freq: 'M', downsample: 'last', upsample: 'ffill', winsorize_q: 0 },
            transform = 'none',
            train = { auto_grid: { p: [0, 1], d: [0, 1], q: [0, 1], P: [0, 1], D: [0, 1], Q: [0, 1], s: 12, max_models: 12 } },
            backtest,
            saveRun,
            runName,
        } = body;

        const codes = [targetCode, ...featureCodes];
        const { found: series, missing } = await this.loadSeries(datasetId, codes);

        const hasTarget = series.some(s => s.code === targetCode);
        if (!hasTarget) {
            throw new NotFoundException(`Target '${targetCode}' not found in dataset ${datasetId}`);
        }

        const presentFeatures = series.filter(s => s.code !== targetCode).map(s => s.code);
        const missingFeatures = missing.filter(c => c !== targetCode);

        // залишаємо лаги тільки для реально наявних фіч
        const lagsFiltered = Object.fromEntries(
            Object.entries(lags).filter(([k]) => presentFeatures.includes(k))
        );

        const req = {
            series,
            resample,
            transform,
            features_cfg: { targetCode, features: presentFeatures, lags: lagsFiltered },
            train,
            backtest,
        };

        const out: any = await this.ml.sarimaxBacktest(req);

        if (saveRun) {
            await this.prisma.modelRun.create({
                data: {
                    name: runName ?? `sarimax_backtest_${new Date().toISOString()}`,
                    type: 'SARIMAX_BACKTEST',
                    configJson: req as any,
                    resultJson: out as any,
                },
            });
        }

        return { ...out, usedFeatures: presentFeatures, missingFeatures };
    }


    /** Forecast з БД */
    async sarimaxForecastFromDb(body: {
        datasetId: string;
        targetCode: string;
        featureCodes?: string[];
        lags?: Record<string, number>;
        resample?: any;
        transform?: 'none' | 'pct' | 'diff1';
        train?: any;
        horizon: number;
        return_pi?: boolean;
        alpha?: number;
        saveRun?: boolean;
        runName?: string;
    }) {
        const {
            datasetId, targetCode,
            featureCodes = [],
            lags = {},
            resample = { enabled: true, freq: 'M', downsample: 'last', upsample: 'ffill', winsorize_q: 0 },
            transform = 'none',
            train = { auto_grid: { p: [0, 1], d: [0, 1], q: [0, 1], P: [0, 1], D: [0, 1], Q: [0, 1], s: 12, max_models: 12 } },
            horizon,
            return_pi = true,
            alpha = 0.1,
            saveRun,
            runName,
        } = body;

        const codes = [targetCode, ...featureCodes];
        const { found: series, missing } = await this.loadSeries(datasetId, codes);

        const hasTarget = series.some(s => s.code === targetCode);
        if (!hasTarget) {
            throw new NotFoundException(`Target '${targetCode}' not found in dataset ${datasetId}`);
        }

        const presentFeatures = series.filter(s => s.code !== targetCode).map(s => s.code);
        const missingFeatures = missing.filter(c => c !== targetCode);

        const lagsFiltered = Object.fromEntries(
            Object.entries(lags).filter(([k]) => presentFeatures.includes(k))
        );

        const req = {
            series,
            resample,
            transform,
            features_cfg: { targetCode, features: presentFeatures, lags: lagsFiltered },
            train,
            horizon,
            return_pi,
            alpha,
        };

        const out: any = await this.ml.sarimaxForecast(req);

        if (saveRun) {
            await this.prisma.modelRun.create({
                data: {
                    name: runName ?? `sarimax_forecast_${new Date().toISOString()}`,
                    type: 'SARIMAX_FORECAST',
                    configJson: req as any,
                    resultJson: out as any,
                },
            });
        }

        return { ...out, usedFeatures: presentFeatures, missingFeatures };
    }


    async recommendFeatures(body: {
        datasetId: string;
        targetCode: string;
        candidateCodes?: string[];
        method?: 'pearson' | 'spearman';
        minOverlap?: number;
        lag?: { min?: number; max?: number; ignoreZero?: boolean };
        transform?: 'none' | 'diff1' | 'pct';
        edgeMin?: number;
        maxLagAbs?: number;
        topK?: number;
        fdrAlpha?: number;
    }) {
        const {
            datasetId, targetCode,
            candidateCodes,
            method = 'pearson',
            minOverlap = 12,
            lag = { min: -12, max: 12, ignoreZero: false },
            transform = 'none',
            edgeMin = 0.3,
            maxLagAbs = 3,
            topK = 10,
            fdrAlpha = 0.1,
        } = body;

        // Використовуємо наш heatmap з БД
        const heat: any = await this.corrHeatmapFromDb({
            datasetId, targetCode, candidateCodes,
            method, minOverlap, lag,
            resample: { enabled: true, freq: 'M', downsample: 'last', upsample: 'ffill', winsorize_q: 0 },
            topK: undefined, // беремо всі, самі відфільтруємо
            transform, returnP: true, fdrAlpha,
        });

        // Відберемо: passed_fdr === true (якщо є), |corr| >= edgeMin, |lag| <= maxLagAbs
        const best = Array.isArray(heat.sortedTop) ? heat.sortedTop : [];
        const picked = best
            .filter((r: any) =>
                (Math.abs(r.value ?? 0) >= edgeMin) &&
                (Math.abs(r.lag ?? 0) <= maxLagAbs) &&
                (r.passed_fdr !== false) // true або undefined -> беремо
            )
            .slice(0, topK)
            .map((r: any) => ({ code: r.key ?? r.code ?? r.series ?? r.key, corr: r.value, lag: r.lag, p: r.p }));

        // повертаємо просто список кодів і трохи діагностики
        return {
            targetCode,
            features: picked.map((x: any) => x.code),
            picked,
            meta: {
                method, minOverlap, transform, edgeMin, maxLagAbs, topK, fdrAlpha,
                totalCandidates: best.length,
            },
        };
    }

    /** 2) Forecast + SAVE у Forecast/ForecastPoint */
    async sarimaxForecastAndSave(body: {
        datasetId: string;
        targetCode: string;
        featureCodes?: string[];
        lags?: Record<string, number>;
        resample?: any;
        transform?: 'none' | 'diff1' | 'pct';
        train?: any;
        horizon: number;
        return_pi?: boolean;
        alpha?: number;
        name?: string;
    }) {
        const out: any = await this.sarimaxForecastFromDb(body); // вже формує req і викликає ml-svc

        // збережемо у таблиці Forecast/ForecastPoint згідно з твоєю схемою
        const rec = await this.prisma.forecast.create({
            data: {
                datasetId: body.datasetId,
                targetCode: body.targetCode,
                featuresJson: (body.featureCodes ?? []) as any,
                lagsJson: (body.lags ?? {}) as any,
                transform: body.transform ?? 'none',
                seasonS: (body.train?.order?.s ?? body.train?.auto_grid?.s ?? 12),
                horizon: body.horizon,
                order: JSON.stringify(out.order ?? out.best_order ?? null),
                seasonalOrder: JSON.stringify(out.seasonal_order ?? out.best_seasonal_order ?? null),
                aic: Number(out.aic ?? out.best_aic ?? 0),
                metaJson: out.meta ?? {},
            },
        });

        if (Array.isArray(out.forecast)) {
            await this.prisma.$transaction(
                out.forecast.map((p: any) =>
                    this.prisma.forecastPoint.create({
                        data: {
                            forecastId: rec.id,
                            date: new Date(p.date),
                            mean: Number(p.mean),
                            lo: p.lo != null ? Number(p.lo) : null,
                            hi: p.hi != null ? Number(p.hi) : null,
                        },
                    })
                )
            );
        }

        return { id: rec.id, forecast: out.forecast, model: out, saved: true };
    }

    /** 3) Список прогнозів */
    async listForecasts(datasetId: string, targetCode?: string, limit = 20) {
        const where: any = { datasetId };
        if (targetCode) where.targetCode = targetCode;
        const rows = await this.prisma.forecast.findMany({
            where,
            orderBy: { createdAt: 'desc' },
            take: limit,
            select: {
                id: true, createdAt: true, datasetId: true, targetCode: true,
                transform: true, seasonS: true, horizon: true, aic: true,
                order: true, seasonalOrder: true,
            },
        });
        return rows;
    }

    /** 4) Один прогноз + точки */
    async getForecast(id: string) {
        const rec = await this.prisma.forecast.findUnique({
            where: { id },
            include: { points: { orderBy: { date: 'asc' } } },
        });
        if (!rec) throw new NotFoundException('Forecast not found');
        return rec;
    }

    async listSeriesKeys(datasetId: string) {
        const rows = await this.prisma.series.findMany({
            where: { datasetId },
            select: { key: true },
            orderBy: { key: 'asc' },
        });
        return { datasetId, count: rows.length, keys: rows.map(r => r.key) };
    }
}
