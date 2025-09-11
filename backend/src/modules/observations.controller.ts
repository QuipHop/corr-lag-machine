import { Body, Controller, Post } from '@nestjs/common';
import { PrismaService } from '../shared/prisma.service';


type ObservationInput = {
    seriesId: number;
    date: string; // ISO date
    value: number | string;
};


@Controller('observations')
export class ObservationsController {
    constructor(private prisma: PrismaService) { }


    @Post('bulk')
    async bulk(@Body() body: { items: ObservationInput[] }) {
        const items = body.items ?? [];
        if (!items.length) return { inserted: 0 };


        const tx = items.map((it) =>
            this.prisma.observation.upsert({
                where: { seriesId_date: { seriesId: it.seriesId, date: new Date(it.date) } },
                update: { value: it.value as any },
                create: { seriesId: it.seriesId, date: new Date(it.date), value: it.value as any },
            }),
        );
        await this.prisma.$transaction(tx);
        return { inserted: items.length };
    }
}