import { Controller, Get } from '@nestjs/common';
import { PrismaService } from '../shared/prisma.service';


@Controller('indicators')
export class IndicatorController {
    constructor(private prisma: PrismaService) { }


    @Get()
    async list() {
        return this.prisma.indicator.findMany({ orderBy: { code: 'asc' } });
    }
}