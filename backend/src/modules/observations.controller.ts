import { Body, Controller, NotFoundException, Post } from '@nestjs/common';
import { PrismaService } from '../shared/prisma.service';

type UpsertItem = {
    seriesId: string | number;
    date: string;   // ISO yyyy-mm-dd or yyyy-mm
    value: number | string;
};

@Controller('api/observations')
export class ObservationsController {
    constructor(private prisma: PrismaService) { }

    /** Bulk upsert observations. Accepts array of { seriesId, date, value }. */
    @Post('bulk-upsert')
    async bulkUpsert(@Body() body: { items: UpsertItem[] }) {
        if (!body?.items?.length) return { upserts: 0 };

        let count = 0;

        for (const it of body.items) {
            const sid = String(it.seriesId); // <-- critical: Series.id is STRING
            // Allow yyyy-MM, normalize to yyyy-MM-01 for monthly
            const dateStr = /^\d{4}-\d{2}$/.test(it.date) ? `${it.date}-01` : it.date;
            const d = new Date(dateStr);
            if (isNaN(d.getTime())) throw new NotFoundException(`Bad date: ${it.date}`);

            await this.prisma.observation.upsert({
                where: { seriesId_date: { seriesId: sid, date: d } },
                update: { value: it.value as any },
                create: { seriesId: sid, date: d, value: it.value as any },
            });

            count++;
        }

        return { upserts: count };
    }
}
