import { Body, Controller, Post, Query } from '@nestjs/common';
import { PrismaService } from '../shared/prisma.service';
import { toStringIds } from '../shared/utils';

type PointDTO = { date: string; value: number };
type MLSeries = { id: string; code: string; points: PointDTO[] };

@Controller('api/ml')
export class MlController {
    constructor(private prisma: PrismaService) { }

    /**
     * Bundle selected series into an object keyed by seriesId:
     * { [seriesId]: { id, code, points[] } }
     */
    @Post('bundle')
    async bundle(@Body() body: any, @Query() query: any) {
        const seriesIds = toStringIds(body?.seriesIds ?? query?.seriesIds);
        if (!seriesIds.length) return { data: {} };

        const series = await this.prisma.series.findMany({
            where: { id: { in: seriesIds } },
            include: { indicator: true } as const,
        });

        const data: Record<string, MLSeries> = {};

        for (const s of series) {
            const obs = await this.prisma.observation.findMany({
                where: { seriesId: s.id },
                orderBy: { date: 'asc' },
                select: { date: true, value: true },
            });

            const code = s.indicator?.code ?? s.key;

            data[s.id] = {
                id: s.id,
                code,
                points: obs.map(o => ({
                    date: o.date.toISOString().slice(0, 10),
                    value: Number(o.value),
                })),
            };
        }

        return { data };
    }
}
