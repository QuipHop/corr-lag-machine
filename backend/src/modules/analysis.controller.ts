import { Body, Controller, Get, HttpException, Param, Post } from '@nestjs/common';
import { PrismaService } from '../shared/prisma.service';
import axios from 'axios';


@Controller('analysis')
export class AnalysisController {
    constructor(private prisma: PrismaService) { }


    @Post('run')
    async run(@Body() body: { seriesIds: number[]; maxLag?: number; method?: 'pearson' | 'spearman'; minOverlap?: number; edgeMin?: number }) {
        const { seriesIds, maxLag = 12, method = 'pearson', minOverlap = 12, edgeMin = 0.3 } = body;
        if (!seriesIds?.length) throw new HttpException('seriesIds required', 400);


        // fetch series + points for ML
        const series = await this.prisma.series.findMany({ where: { id: { in: seriesIds } }, include: { indicator: true } });
        const payload = [] as any[];
        for (const s of series) {
            const points = await this.prisma.observation.findMany({ where: { seriesId: s.id }, orderBy: { date: 'asc' }, select: { date: true, value: true } });
            payload.push({ id: s.id, code: `${s.indicator.code}${s.region ? ':' + s.region : ''}`, points: points.map(p => ({ date: p.date.toISOString().slice(0, 10), value: Number(p.value) })) });
        }


        const mlUrl = `${process.env.ML_SVC_URL || 'http://localhost:8000'}/corr-lag`;
        const { data } = await axios.post(mlUrl, { series: payload, maxLag, method, minOverlap, edgeMin });


        // persist run + edges
        const run = await this.prisma.analysisRun.create({
            data: {
                method,
                maxLag,
                minOverlap,
                edgeMin,
                seriesIds: seriesIds.join(','),
            },
        });


        if (Array.isArray(data?.edges)) {
            const mapByCode = new Map(series.map(s => [`${s.indicator.code}${s.region ? ':' + s.region : ''}`, s.id] as const));
            const rows = data.edges.map((e: any) => ({ runId: run.id, sourceId: mapByCode.get(e.source)!, targetId: mapByCode.get(e.target)!, lag: e.lag, weight: e.weight }));
            // transaction bulk insert
            await this.prisma.$transaction(rows.map((r: any) => this.prisma.analysisEdge.create({ data: r })));
        }


        return { runId: run.id, nodes: data?.nodes ?? [], edges: data?.edges ?? [] };
    }


    @Get(':runId')
    async get(@Param('runId') runId: string) {
        const id = Number(runId);
        const run = await this.prisma.analysisRun.findUnique({ where: { id } });
        if (!run) throw new HttpException('not found', 404);
        const edges = await this.prisma.analysisEdge.findMany({ where: { runId: id } });
        return { run, edges };
    }


    @Get('latest/one')
    async latest() {
        const run = await this.prisma.analysisRun.findFirst({ orderBy: { id: 'desc' } });
        if (!run) return { run: null, edges: [] };
        const edges = await this.prisma.analysisEdge.findMany({ where: { runId: run.id } });
        return { run, edges };
    }
}