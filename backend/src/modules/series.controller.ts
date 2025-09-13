import { Controller, Get, NotFoundException, Param, Query } from '@nestjs/common';
import { PrismaService } from '../shared/prisma.service';

@Controller('api/series') // if you prefer old path, change to just 'series'
export class SeriesController {
    constructor(private prisma: PrismaService) { }

    @Get(':id')
    async getSeries(@Param('id') id: string) {
        // Series.id is a STRING (cuid), do NOT coerce to number
        const s = await this.prisma.series.findUnique({
            where: { id },
            include: { indicator: true, dataset: true },
        });
        if (!s) throw new NotFoundException('Series not found');
        return s;
    }

    @Get(':id/data')
    async getSeriesData(
        @Param('id') id: string,
        @Query('from') from?: string,
        @Query('to') to?: string,
    ) {
        // Observation.seriesId is also STRING
        const where: any = { seriesId: id };
        if (from || to) {
            where.date = {};
            if (from) where.date.gte = new Date(from);
            if (to) where.date.lte = new Date(to);
        }

        const points = await this.prisma.observation.findMany({
            where,
            orderBy: { date: 'asc' },
            select: { date: true, value: true },
        });

        // Convert Decimal to number and ISO date for JSON
        const normalized = points.map(p => ({
            date: p.date.toISOString().slice(0, 10),
            value: Number(p.value),
        }));

        return { seriesId: id, points: normalized };
    }
}
