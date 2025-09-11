import { Controller, Get, Query } from '@nestjs/common';
import { PrismaService } from '../shared/prisma.service';


@Controller('series')
export class SeriesListController {
    constructor(private prisma: PrismaService) { }


    @Get()
    async list(@Query('indicatorId') indicatorId?: string) {
        const where = indicatorId ? { indicatorId: Number(indicatorId) } : {};
        return this.prisma.series.findMany({ where, include: { indicator: true }, orderBy: { id: 'asc' } });
    }
}