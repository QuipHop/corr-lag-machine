import { BadRequestException, Body, Controller, Get, Headers, Post, Query, Res } from '@nestjs/common';
import { Response } from 'express';
import { CorrelationService } from './correlation.service';
import { CorrHeatmapRequestDto, CorrLagRequestDto } from './dto/correlate.dto';
import * as crypto from 'crypto';
import { AuditService } from '../audit/audit.service';
import { SkipThrottle } from '@nestjs/throttler';


const LIMITS = {
    maxSeries: parseInt(process.env.ML_LIMIT_MAX_SERIES || '50', 10),
    maxPoints: parseInt(process.env.ML_LIMIT_MAX_POINTS || '5000', 10),
    maxLagSpan: parseInt(process.env.ML_LIMIT_MAX_LAG_SPAN || '60', 10),
    minMinOverlap: parseInt(process.env.ML_LIMIT_MIN_OVERLAP_MIN || '3', 10),
};

@Controller('analysis')
export class CorrelationController {
    constructor(
        private readonly svc: CorrelationService,
        private readonly audit: AuditService,
    ) { }

    @Get('ml/health')
    @SkipThrottle()
    health() { return this.svc.health(); }

    @Get('limits')
    @SkipThrottle()
    limits() { return LIMITS; }


    @Get('audit/recent')
    @SkipThrottle()
    async recent(@Query('limit') limit?: string) {
        const n = limit ? parseInt(limit, 10) : 50;
        return this.audit.recent(n);
    }

    @Post('corr-heatmap')
    async corrHeatmap(
        @Body() dto: CorrHeatmapRequestDto,
        @Query('refresh') refresh: string | undefined,
        @Res({ passthrough: true }) res: Response,
        @Headers('x-request-id') reqId?: string,
    ) {
        this.validateHeatmap(dto);
        const requestId = reqId || crypto.randomUUID();
        const force = refresh === '1' || refresh === 'true';

        try {
            const { data, cacheMeta, httpMeta } = await this.svc.corrHeatmap(dto, force, requestId);
            res.setHeader('x-request-id', requestId);
            res.setHeader('x-ml-cache', cacheMeta.hit ? 'HIT' : 'MISS');
            res.setHeader('x-ml-cache-age', cacheMeta.age_s.toFixed(3));
            res.setHeader('x-ml-attempts', String(httpMeta.attempts));
            res.setHeader('x-ml-rt-ms', httpMeta.rt_ms.toFixed(1));

            // AUDIT OK
            await this.audit.record({
                requestId, endpoint: 'corr_heatmap', dto,
                cacheHit: cacheMeta.hit, cacheAge: cacheMeta.age_s,
                httpAttempts: httpMeta.attempts, httpRtMs: httpMeta.rt_ms,
                status: 201,
            });

            return data;
        } catch (err: any) {
            // AUDIT ERROR
            await this.audit.record({
                requestId, endpoint: 'corr_heatmap', dto,
                cacheHit: false, cacheAge: 0, httpAttempts: 0, httpRtMs: 0,
                status: err?.status ?? 500, error: err?.response?.data ?? err?.message,
            });
            throw err;
        }
    }

    @Post('corr-lag')
    async corrLag(
        @Body() dto: CorrLagRequestDto,
        @Query('refresh') refresh: string | undefined,
        @Res({ passthrough: true }) res: Response,
        @Headers('x-request-id') reqId?: string,
    ) {
        this.validateLag(dto);
        const requestId = reqId || crypto.randomUUID();
        const force = refresh === '1' || refresh === 'true';

        try {
            const { data, cacheMeta, httpMeta } = await this.svc.corrLag(dto, force, requestId);
            res.setHeader('x-request-id', requestId);
            res.setHeader('x-ml-cache', cacheMeta.hit ? 'HIT' : 'MISS');
            res.setHeader('x-ml-cache-age', cacheMeta.age_s.toFixed(3));
            res.setHeader('x-ml-attempts', String(httpMeta.attempts));
            res.setHeader('x-ml-rt-ms', httpMeta.rt_ms.toFixed(1));

            await this.audit.record({
                requestId, endpoint: 'corr_lag', dto,
                cacheHit: cacheMeta.hit, cacheAge: cacheMeta.age_s,
                httpAttempts: httpMeta.attempts, httpRtMs: httpMeta.rt_ms,
                status: 201,
            });

            return data;
        } catch (err: any) {
            await this.audit.record({
                requestId, endpoint: 'corr_lag', dto,
                cacheHit: false, cacheAge: 0, httpAttempts: 0, httpRtMs: 0,
                status: err?.status ?? 500, error: err?.response?.data ?? err?.message,
            });
            throw err;
        }
    }

    // ---- прості валідації (як було) ----
    private validateSeries(series: { code: string; points: { date: string; value: number }[] }[]) {
        if (!Array.isArray(series) || series.length === 0) {
            throw new BadRequestException('series: must be non-empty array');
        }
        if (series.length > LIMITS.maxSeries) {
            throw new BadRequestException(`series: too many (${series.length}) > ${LIMITS.maxSeries}`);
        }
        let totalPoints = 0;
        for (const s of series) totalPoints += Array.isArray((s as any).points) ? (s as any).points.length : 0;
        if (totalPoints > LIMITS.maxPoints) {
            throw new BadRequestException(`points: too many (${totalPoints}) > ${LIMITS.maxPoints}`);
        }
    }
    private validateLagSpan(minLag: number, maxLag: number) {
        const span = maxLag - minLag;
        if (span < 0) throw new BadRequestException('lag: max must be >= min');
        if (span > LIMITS.maxLagSpan) throw new BadRequestException(`lag span too wide (${span}) > ${LIMITS.maxLagSpan}`);
    }
    private validateMinOverlap(minOverlap: number) {
        if (minOverlap < LIMITS.minMinOverlap) throw new BadRequestException(`minOverlap must be >= ${LIMITS.minMinOverlap}`);
    }
    private validateHeatmap(dto: CorrHeatmapRequestDto) {
        this.validateSeries(dto.series as any);
        this.validateMinOverlap(dto.minOverlap);
        const L = dto.lag || { min: -12, max: 12 };
        this.validateLagSpan(L.min, L.max);
    }
    private validateLag(dto: CorrLagRequestDto) {
        this.validateSeries(dto.series as any);
        this.validateMinOverlap(dto.minOverlap);
        const legacyMax = dto.maxLag ?? 12;
        const L = dto.lag ? dto.lag : { min: -legacyMax, max: legacyMax };
        this.validateLagSpan(L.min, L.max);
    }
}
