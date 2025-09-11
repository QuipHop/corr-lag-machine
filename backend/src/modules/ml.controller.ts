import { Body, Controller, HttpException, Post } from '@nestjs/common';
import { PrismaService } from '../shared/prisma.service';
import axios from 'axios';


@Controller('ml')
export class MlController {
    constructor(private prisma: PrismaService) { }


    @Post('corr-lag')
    async corrLag(@Body() body: { seriesIds: number[]; maxLag?: number; method?: 'pearson' | 'spearman'; minOverlap?: number }) {
        const { seriesIds, maxLag = 12, method = 'pearson', minOverlap = 12 } = body;
        if (!seriesIds?.length) throw new HttpException('seriesIds required', 400);


        const series = await this.prisma.series.findMany({ where: { id: { in: seriesIds } }, include: { indicator: true } });
        const data: Record<number, { id: number; code: string; points: { date: string; value: number }[] }> = {};


        for (const s of series) {
            const points = await this.prisma.observation.findMany({
                where: { seriesId: s.id },
                orderBy: { date: 'asc' },
                select: { date: true, value: true },
            });
            data[s.id] = {
                id: s.id,
                code: `${s.indicator.code}${s.region ? ':' + s.region : ''}`,
                points: points.map((p: any) => ({ date: p.date.toISOString().slice(0, 10), value: Number(p.value) })),
            };
        }


        const resp = await axios.post(`${process.env.ML_SVC_URL || 'http://localhost:8000'}/corr-lag`, {
            series: Object.values(data),
            maxLag,
            method,
            minOverlap,
        });
        return resp.data;
    }
}