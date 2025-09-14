import { BadRequestException, Injectable, NotFoundException } from '@nestjs/common';
import { PrismaService } from '../shared/prisma.service';
import Papa from 'papaparse';
import XLSX from 'xlsx';
import type { Express } from 'express';

import { CommitDto } from './dto/commit.dto';
import { PreviewDto } from './dto/preview.dto';

type Row = Record<string, string>;
type NormRow = { date: string; value: number };

function pickHeaderRow(aoa: any[][], maxScan = 10) {
    // Heuristic: choose the row with the highest share of non-empty strings and unique cells
    let bestIdx = 0, bestScore = -1;
    for (let i = 0; i < Math.min(maxScan, aoa.length); i++) {
        const row = aoa[i] ?? [];
        const nonEmpty = row.filter(c => c !== null && c !== undefined && String(c).trim() !== '');
        if (!nonEmpty.length) continue;
        const strings = nonEmpty.filter(c => typeof c === 'string');
        const uniq = new Set(nonEmpty.map(c => String(c).trim().toLowerCase())).size;
        const score = strings.length * 2 + uniq; // weight strings higher than numbers
        if (score > bestScore) { bestScore = score; bestIdx = i; }
    }
    return bestIdx;
}

function sanitizeHeader(name: string, index: number, seen: Set<string>) {
    let base = String(name ?? '').trim();
    if (!base) base = `col_${index + 1}`;
    // normalize spaces and punctuation
    base = base.replace(/\s+/g, ' ').replace(/[^\w\s\-:/.%]/g, '').trim();
    if (!base) base = `col_${index + 1}`;
    let out = base, k = 2;
    while (seen.has(out)) out = `${base}_${k++}`;
    seen.add(out);
    return out;
}
/** Decide CSV vs Excel using mimetype/ext; default to CSV; parse into Row[]. */
function parseToRows(file: Express.Multer.File): Row[] {
    const name = (file.originalname || '').toLowerCase();
    const mime = (file.mimetype || '').toLowerCase();

    const isExcelMime = /spreadsheetml|ms-excel/.test(mime);
    const isExcelExt = /\.(xlsx|xls)$/.test(name);
    const isCsvMime = /text\/csv|application\/csv/.test(mime);
    const isCsvExt = /\.csv$/.test(name);

    // Fallback sniff: look for delimiters in first few lines
    const asText = file.buffer.toString('utf-8');
    const looksCsv = asText.split('\n').slice(0, 3).some(l => l.includes(',') || l.includes(';'));

    if (isExcelMime || isExcelExt) {
        const wb = XLSX.read(file.buffer, { type: 'buffer' });
        const first = wb.SheetNames[0];
        const ws = wb.Sheets[first];
        return XLSX.utils.sheet_to_json<Row>(ws, { raw: false, defval: '' });
    }

    // CSV (mime/ext/sniff) or default
    const res = Papa.parse<Row>(asText, { header: true, skipEmptyLines: true });
    if (res.errors?.length) throw new BadRequestException('CSV parse error: ' + res.errors[0].message);
    return res.data as Row[];
}

function normalizeMonthly(
    rows: Row[],
    dateCol: string,
    valCol: string,
    opts: { decimal: 'auto' | 'dot' | 'comma'; dateFormat?: string; dropBlanks?: boolean }
) {
    const warns: string[] = [];
    const out: NormRow[] = [];

    const toNum = (s: string) => {
        if (opts.decimal === 'comma' || (opts.decimal === 'auto' && s.includes(','))) {
            s = s.replace(/\./g, '').replace(',', '.');
        }
        s = s.replace(/\s+/g, '').replace('%', '');
        const v = Number(s);
        return Number.isFinite(v) ? v : NaN;
    };

    // for "Jan 2024" / "Jan-2024"
    const MONTHS_EN = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'];

    for (const r of rows) {
        const dRaw = (r[dateCol] ?? '').trim();
        const vRaw = (r[valCol] ?? '').trim();

        if (!dRaw || !vRaw) {
            if (!opts.dropBlanks) warns.push('blank row');
            continue;
        }

        // Accept:
        //  - YYYY-MM
        //  - YYYY-MM-DD (snap to month)
        //  - DD.MM.YYYY (snap to month)
        //  - MM/YYYY
        //  - MMM YYYY or MMM-YYYY (English 3-letter month)
        let iso = '';
        const s = dRaw;

        if (/^\d{4}-\d{2}$/.test(s)) {
            iso = `${s}-01`;
        } else if (/^\d{4}-\d{2}-\d{2}$/.test(s)) {
            iso = s.slice(0, 7) + '-01';
        } else if (/^\d{2}\.\d{2}\.\d{4}$/.test(s)) {
            const [dd, mm, yy] = s.split('.');
            iso = `${yy}-${mm}-01`;
        } else if (/^\d{1,2}\/\d{4}$/.test(s)) {
            // MM/YYYY
            const [mm, yy] = s.split('/');
            iso = `${yy}-${String(Number(mm)).padStart(2, '0')}-01`;
        } else {
            // MMM YYYY or MMM-YYYY (e.g., Jan 2024 or Jan-2024)
            const m = s.match(/^([A-Za-z]{3})[ -](\d{4})$/);
            if (m) {
                const idx = MONTHS_EN.indexOf(m[1].toLowerCase());
                if (idx >= 0) iso = `${m[2]}-${String(idx + 1).padStart(2, '0')}-01`;
            }
        }

        if (!iso) { warns.push(`unparseable date: ${s}`); continue; }

        const v = toNum(vRaw);
        if (!Number.isFinite(v)) { warns.push(`non-numeric: ${vRaw}`); continue; }

        out.push({ date: iso, value: v });
    }

    // dedupe per month (keep last), then sort
    const map = new Map<string, number>();
    for (const r of out) map.set(r.date, r.value);
    const rowsNorm = [...map.entries()]
        .map(([date, value]) => ({ date, value }))
        .sort((a, b) => a.date.localeCompare(b.date));

    return { rows: rowsNorm, warnings: warns };
}


@Injectable()
export class UploadsService {
    constructor(private prisma: PrismaService) { }

    /** in-memory temp store while previewing */
    private uploads = new Map<string, { datasetId: string; file: Buffer; meta: { originalname: string; mimetype: string }; cached?: any }>();

    ingestTemp(datasetId: string, file: Express.Multer.File, meta: any) {
        if (!file) throw new BadRequestException('file required');

        const uploadId = 'up_' + Math.random().toString(36).slice(2);
        this.uploads.set(uploadId, {
            datasetId,
            file: file.buffer,
            meta: { originalname: file.originalname, mimetype: file.mimetype },
            cached: undefined,
        });

        // parse once to guess columns
        const rows = parseToRows(file);
        const first = rows[0] ?? {};
        const columns = Object.keys(first).map(name => ({
            name,
            typeGuess: /date|month|period/i.test(name) ? 'date' : 'number',
            examples: [first[name] as string],
        }));

        return { uploadId, columns, sampleRows: Math.min(10, rows.length) };
    }

    preview(uploadId: string, dto: PreviewDto) {
        const up = this.uploads.get(uploadId); if (!up) throw new NotFoundException('upload not found');

        // Reuse stored meta to detect format again
        const pseudoFile = {
            buffer: up.file,
            originalname: up.meta.originalname,
            mimetype: up.meta.mimetype,
        } as unknown as Express.Multer.File;

        const rows = parseToRows(pseudoFile);

        const series = dto.valueColumns.map((vc) => {
            const { rows: norm, warnings } = normalizeMonthly(rows, dto.dateColumn, (vc as any).name, dto as any);
            return { key: (vc as any).key, label: (vc as any).label, units: (vc as any).units, rows: norm, rowCount: norm.length, warnings };
        });

        up.cached = { dto, series };
        return { normalized: { freq: 'M', series }, warnings: series.flatMap((s: any) => s.warnings) };
    }

    async commit(uploadId: string, dto: CommitDto, idem?: string) {
        const up = this.uploads.get(uploadId); if (!up?.cached) throw new BadRequestException('preview first');
        const { series } = up.cached;

        return this.prisma.$transaction(async (tx) => {
            const ds = await tx.dataset.findUnique({ where: { id: up.datasetId } });
            if (!ds) throw new NotFoundException('dataset not found');

            const created: string[] = [];
            for (const s of series) {
                // ensure Series exists (connect to dataset)
                let ser = await tx.series.findFirst({ where: { datasetId: ds.id, key: s.key } });
                if (!ser && dto.createSeries) {
                    ser = await tx.series.create({
                        data: {
                            key: s.key,
                            label: s.label,
                            units: s.units,
                            dataset: { connect: { id: ds.id } },
                        },
                    });
                    created.push(s.key);
                }
                if (!ser) throw new BadRequestException(`series ${s.key} missing; set createSeries=true`);

                if (dto.upsertMode === 'replace') {
                    await tx.observation.deleteMany({ where: { seriesId: ser.id } });
                }

                if (s.rows.length) {
                    const values = s.rows.map((r: NormRow) => `('${ser!.id}', '${r.date}', ${r.value})`).join(',');
                    await tx.$executeRawUnsafe(`
            INSERT INTO "Observation" ("seriesId","date","value")
            VALUES ${values}
            ON CONFLICT ("seriesId","date") DO UPDATE SET "value" = EXCLUDED."value";
          `);
                }
            }

            if (dto.saveMappingToDataset) {
                await tx.dataset.update({ where: { id: ds.id }, data: { mappingJson: up.cached.dto } });
            }

            const pointsUpserted = series.reduce((a: number, s: any) => a + s.rows.length, 0);
            return { datasetId: ds.id, seriesCreated: created, pointsUpserted };
        });
    }
}
