import { Controller, Get, Param, Query } from '@nestjs/common';
import { PrismaService } from '../shared/prisma.service';


@Controller('series')
export class SeriesController {
    constructor(private prisma: PrismaService) { }


    @Get(':id')
    async getSeries(@Param('id') id: string) {
        const seriesId = Number(id);
        const s = await this.prisma.series.findUnique({
            where: { id: seriesId },
            include: { indicator: true },
        });
        return s;
    }


    @Get(':id/data')
    async getSeriesData(
        @Param('id') id: string,
        @Query('from') from?: string,
        @Query('to') to?: string,
    ) {
        const seriesId = Number(id);
        const where: any = { seriesId };
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
        return { seriesId, points };
    }
}