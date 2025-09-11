/**
 * Usage: npx ts-node scripts/import_csv.ts path/to/file.csv SERIES_ID
 * CSV must have: date,value
 */
import fs from 'node:fs';
import path from 'node:path';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
    const [, , csvPath, seriesIdStr] = process.argv;
    if (!csvPath || !seriesIdStr) {
        console.error('Usage: npx ts-node scripts/import_csv.ts <csvPath> <seriesId>');
        process.exit(1);
    }
    const seriesId = Number(seriesIdStr);
    const content = fs.readFileSync(path.resolve(csvPath), 'utf-8');
    const lines = content.split(/\r?\n/).filter(Boolean);
    const header = lines.shift()!.split(',').map((s) => s.trim().toLowerCase());
    const dateIdx = header.indexOf('date');
    const valueIdx = header.indexOf('value');
    if (dateIdx === -1 || valueIdx === -1) throw new Error('CSV must have date,value');

    const items = lines.map((line) => {
        const cols = line.split(',');
        return { date: cols[dateIdx], value: cols[valueIdx] };
    });

    const tx = items.map((it) =>
        prisma.observation.upsert({
            where: { seriesId_date: { seriesId, date: new Date(it.date) } },
            update: { value: it.value as any },
            create: { seriesId, date: new Date(it.date), value: it.value as any },
        }),
    );
    await prisma.$transaction(tx);
    console.log(`Imported ${items.length} rows into series ${seriesId}`);
}

main().finally(async () => prisma.$disconnect());