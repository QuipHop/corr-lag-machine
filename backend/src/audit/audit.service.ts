import { Injectable, Logger } from '@nestjs/common';
import { PrismaClient, $Enums } from '@prisma/client';
import * as crypto from 'crypto';

// ---- Глобальний Singleton PrismaClient для dev hot-reload ----
const g = globalThis as unknown as { __prisma?: PrismaClient };
const prismaSingleton = g.__prisma ?? new PrismaClient();
if (!g.__prisma) g.__prisma = prismaSingleton;
// --------------------------------------------------------------

@Injectable()
export class AuditService {
    private prisma = prismaSingleton;
    private readonly log = new Logger('AuditService');

    async record(input: {
        requestId: string;
        endpoint: 'corr_heatmap' | 'corr_lag';
        dto: any;
        cacheHit: boolean;
        cacheAge: number;
        httpAttempts: number;
        httpRtMs: number;
        status: number;
        error?: any;
    }): Promise<void> {
        try {
            const series = Array.isArray(input.dto?.series) ? input.dto.series : [];
            const seriesCount = series.length;
            const pointsCount = series.reduce(
                (acc: number, s: any) => acc + (Array.isArray(s?.points) ? s.points.length : 0),
                0
            );

            const legacyMax = input.dto?.maxLag ?? 12;
            const lag = input.dto?.lag ?? (
                input.endpoint === 'corr_lag' ? { min: -legacyMax, max: legacyMax } : { min: -12, max: 12 }
            );

            // map transform -> $Enums.TransformMode
            const transformStr = String(input.dto?.transform ?? 'none').toLowerCase();
            let transformEnum: $Enums.TransformMode = 'NONE';
            switch (transformStr) {
                case 'diff1': transformEnum = 'DIFF1'; break;
                case 'pct': transformEnum = 'PCT'; break;
                default: transformEnum = 'NONE';
            }

            // map endpoint -> $Enums.AnalysisEndpoint
            const endpointEnum: $Enums.AnalysisEndpoint =
                input.endpoint === 'corr_heatmap' ? 'CORR_HEATMAP' : 'CORR_LAG';

            const minOverlap: number | null = input.dto?.minOverlap ?? null;
            const sha = crypto.createHash('sha256').update(JSON.stringify(input.dto)).digest('hex');

            await this.prisma.analysisAudit.create({
                data: {
                    requestId: input.requestId,
                    endpoint: endpointEnum,
                    sha,
                    seriesCount,
                    pointsCount,
                    minOverlap,
                    lagMin: Number(lag.min),
                    lagMax: Number(lag.max),
                    transform: transformEnum,
                    cacheHit: input.cacheHit,
                    cacheAgeSeconds: input.cacheAge,
                    httpAttempts: input.httpAttempts,
                    httpRtMs: Math.round(input.httpRtMs),
                    status: input.status,
                    error: input.error ? JSON.stringify(input.error).slice(0, 4000) : null,
                },
            });
        } catch (e: any) {
            // не валимо запит, тільки логуємо
            this.log.warn(`audit failed: ${e?.message}`);
        }
    }

    async recent(limit = 50) {
        try {
            const take = Math.min(Math.max(limit, 1), 200);
            return await this.prisma.analysisAudit.findMany({
                orderBy: { createdAt: 'desc' },
                take,
                select: {
                    id: true,
                    createdAt: true,
                    requestId: true,
                    endpoint: true,
                    status: true,
                    seriesCount: true,
                    pointsCount: true,
                    lagMin: true,
                    lagMax: true,
                    transform: true,
                    cacheHit: true,
                    cacheAgeSeconds: true,
                    httpAttempts: true,
                    httpRtMs: true,
                },
            });
        } catch (e: any) {
            this.log.warn(`audit recent failed: ${e?.message}`);
            return [];
        }
    }
}
