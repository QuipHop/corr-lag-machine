import { Body, Controller, Get, Param, Post } from '@nestjs/common';
import { DatasetsService } from './datasets.service';
import { PrismaService } from '../shared/prisma.service';
import { CreateDatasetDto } from './dto/create-dataset.dto';

@Controller('api/datasets')
export class DatasetsController {
    constructor(
        private readonly svc: DatasetsService,
        private readonly prisma: PrismaService,
    ) { }

    @Post()
    create(@Body() dto: CreateDatasetDto) {
        return this.svc.create(dto);
    }

    /** List datasets (with series count) */
    @Get()
    async list() {
        const rows = await this.prisma.dataset.findMany({
            orderBy: { createdAt: 'desc' },
            select: {
                id: true, name: true, freq: true, createdAt: true,
                _count: { select: { Series: true } }, // relation name from your schema
            },
        });
        return rows.map(r => ({
            id: r.id,
            name: r.name,
            freq: r.freq,
            createdAt: r.createdAt,
            seriesCount: r._count.Series,
        }));
    }

    /** Dataset detail (light) */
    @Get(':id')
    async get(@Param('id') id: string) {
        return this.prisma.dataset.findUnique({
            where: { id },
            select: { id: true, name: true, freq: true, createdAt: true, mappingJson: true },
        });
    }

    /** List series for a dataset */
    @Get(':id/series')
    async listSeries(@Param('id') id: string) {
        return this.prisma.series.findMany({
            where: { datasetId: id },
            orderBy: { key: 'asc' },
            select: { id: true, key: true, label: true, units: true, createdAt: true },
        });
    }

    /** Get saved mapping to prefill upload wizard */
    @Get(':id/mapping')
    async getMapping(@Param('id') id: string) {
        const ds = await this.prisma.dataset.findUnique({
            where: { id }, select: { mappingJson: true },
        });
        return ds?.mappingJson ?? {};
    }
}
