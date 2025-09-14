import { Body, Controller, Get, NotFoundException, Param, Post, Query } from '@nestjs/common';
import { PrismaService } from '../shared/prisma.service';
import { CorrelationService } from '../corellation/correlation.service';
import { toStringIds } from '../shared/utils';

type GraphReq = {
    datasetId: string;
    /** Optional: series keys; if omitted, we use ALL series in dataset */
    series?: string[];
    method?: 'spearman' | 'pearson';     // default 'spearman'
    pearsonAlso?: boolean;             // default false
    minOverlap?: number;               // default 6
    edgeMin?: number;                  // default 0.5 (abs threshold)
};

@Controller('api/analysis')
export class AnalysisController {
    constructor(
        private prisma: PrismaService,
        private corr: CorrelationService,
    ) { }

    /** Build & persist a correlation graph into AnalysisRun/AnalysisEdge */
    @Post('graph')
    async graph(@Body() body: GraphReq) {
        const datasetId = body.datasetId;
        const method = body.method ?? 'spearman';
        const pearsonAlso = body.pearsonAlso ?? false;
        const minOverlap = body.minOverlap ?? 6;
        const edgeMin = body.edgeMin ?? 0.5;

        // Get series KEYS to analyze
        const seriesKeys = Array.isArray(body.series) && body.series.length
            ? body.series
            : (await this.prisma.series.findMany({
                where: { datasetId },
                select: { key: true },
            })).map(s => s.key);

        if (seriesKeys.length < 2) {
            return { runId: null, edges: [], message: 'Need at least 2 series' };
        }

        // Compute correlations
        const result = await this.corr.run({
            datasetId,
            series: seriesKeys,
            method,
            pearsonAlso,
        });

        // Map series.key -> series.id
        const series = await this.prisma.series.findMany({
            where: { datasetId, key: { in: seriesKeys } },
            select: { id: true, key: true },
        });
        const idByKey = new Map(series.map(s => [s.key, s.id]));

        // Filter edges
        const edges = result.pairs
            .filter(p => p.n >= minOverlap)
            .map(p => {
                const r = method === 'pearson' ? (p.pearson ?? 0) : (p.spearman ?? 0);
                return { ...p, r, abs: Math.abs(r) };
            })
            .filter(p => p.abs >= edgeMin)
            .map(p => ({
                sourceId: idByKey.get(p.x)!,
                targetId: idByKey.get(p.y)!,
                lag: 0,
                weight: p.r,
            }))
            // Drop any null mappings (shouldn’t happen, but safe)
            .filter(e => !!e.sourceId && !!e.targetId);

        // Persist run + edges
        const run = await this.prisma.analysisRun.create({
            data: {
                method,
                maxLag: 0,
                minOverlap,
                edgeMin,
                seriesIds: series.map(s => s.id).join(','),
                datasetId,
            },
        });

        if (edges.length) {
            await this.prisma.analysisEdge.createMany({
                data: edges.map(e => ({ ...e, runId: run.id })),
            });
        }

        return {
            runId: run.id,
            datasetId,
            method,
            minOverlap,
            edgeMin,
            edgesInserted: edges.length,
        };
    }

    /** Optional helper: remap a mixed seriesIds payload (string|number[]|CSV) */
    @Post('payload')
    async makePayload(@Body() body: any) {
        const seriesIds = toStringIds(body?.seriesIds);
        if (!seriesIds.length) return { series: [], mapByCode: {} };

        const series = await this.prisma.series.findMany({
            where: { id: { in: seriesIds } },
            include: { indicator: true } as const,
        });

        const payload = [];
        for (const s of series) {
            const obs = await this.prisma.observation.findMany({
                where: { seriesId: s.id },
                orderBy: { date: 'asc' },
                select: { date: true, value: true },
            });
            const code = s.indicator?.code ?? s.key;
            payload.push({
                id: s.id,
                code,
                points: obs.map(o => ({ date: o.date.toISOString().slice(0, 10), value: Number(o.value) })),
            });
        }
        const mapByCode = Object.fromEntries(series.map(s => [(s.indicator?.code ?? s.key), s.id]));
        return { series: payload, mapByCode };
    }

    @Get('runs')
    async listRuns(@Query('datasetId') datasetId?: string) {
        const runs = await this.prisma.analysisRun.findMany({
            where: datasetId ? { datasetId } : undefined,
            orderBy: { id: 'desc' },
            include: { edges: { select: { id: true } } },
        });
        return runs.map(r => ({
            id: r.id,
            datasetId: r.datasetId,
            createdAt: r.createdAt,
            method: r.method,
            minOverlap: r.minOverlap,
            edgeMin: r.edgeMin,
            edgeCount: r.edges.length,
        }));
    }

    /** Get a run with edges resolved to series keys */
    @Get('run/:id')
    async getRun(@Param('id') id: string) {
        const runId = Number(id);
        const run = await this.prisma.analysisRun.findUnique({
            where: { id: runId },
            include: { edges: true },
        });
        if (!run) throw new NotFoundException('Run not found');

        const seriesIds = Array.from(new Set(run.edges.flatMap(e => [e.sourceId, e.targetId])));
        const series = await this.prisma.series.findMany({
            where: { id: { in: seriesIds } },
            select: { id: true, key: true, label: true },
        });
        const keyById = new Map(series.map(s => [s.id, s.key]));

        return {
            id: run.id,
            datasetId: run.datasetId,
            createdAt: run.createdAt,
            method: run.method,
            minOverlap: run.minOverlap,
            edgeMin: run.edgeMin,
            edges: run.edges.map(e => ({
                sourceId: e.sourceId,
                targetId: e.targetId,
                sourceKey: keyById.get(e.sourceId) ?? e.sourceId,
                targetKey: keyById.get(e.targetId) ?? e.targetId,
                lag: e.lag,
                weight: e.weight,
            })),
        };
    }
}
