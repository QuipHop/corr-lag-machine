import { HttpService } from '@nestjs/axios';
import { Injectable, InternalServerErrorException, HttpException } from '@nestjs/common';
import { lastValueFrom } from 'rxjs';
import {
    CorrHeatmapRequestDto,
    CorrLagRequestDto,
} from './dto/correlate.dto';
import * as crypto from 'crypto';

type CacheEntry = { ts: number; data: any };
type CacheMeta = { hit: boolean; age_s: number };
type HttpMeta = { attempts: number; rt_ms: number };

@Injectable()
export class CorrelationService {
    private readonly base = process.env.ML_SVC_URL || 'http://localhost:8000';

    // in-memory LRU cache
    private readonly cache = new Map<string, CacheEntry>();
    private readonly TTL = parseInt(process.env.ML_CACHE_TTL_SECONDS || '600', 10); // 10 хв
    private readonly MAX = parseInt(process.env.ML_CACHE_MAX_ENTRIES || '200', 10); // 200 ключів

    // ретраї
    private readonly MAX_ATTEMPTS = parseInt(process.env.ML_HTTP_MAX_ATTEMPTS || '3', 10);
    private readonly BACKOFF_MS = parseInt(process.env.ML_HTTP_BACKOFF_MS || '200', 10);
    private readonly TIMEOUT_MS = parseInt(process.env.ML_HTTP_TIMEOUT_MS || '120000', 10);

    constructor(private readonly http: HttpService) { }

    async health() {
        try {
            const { data } = await lastValueFrom(this.http.get(`${this.base}/health`));
            return data;
        } catch (e: any) {
            throw this._wrapError(e);
        }
    }

    async corrHeatmap(dto: CorrHeatmapRequestDto, refresh = false, requestId?: string) {
        return this.cachedPost('/corr-heatmap', dto, refresh, requestId);
    }
    async corrLag(dto: CorrLagRequestDto, refresh = false, requestId?: string) {
        return this.cachedPost('/corr-lag', dto, refresh, requestId);
    }

    private async cachedPost(
        endpoint: string,
        body: unknown,
        refresh: boolean,
        requestId?: string,
    ): Promise<{ data: any; cacheMeta: CacheMeta; httpMeta: HttpMeta }> {
        const key = this._makeKey(endpoint, body);           // <- ДОДАНО
        const now = Date.now();

        // HIT-перевірка
        if (!refresh) {
            const entry = this.cache.get(key);
            if (entry && (now - entry.ts) / 1000 <= this.TTL) {
                // LRU bump
                this.cache.delete(key);
                this.cache.set(key, entry);
                return {
                    data: entry.data,
                    cacheMeta: { hit: true, age_s: (Date.now() - entry.ts) / 1000 },
                    httpMeta: { attempts: 0, rt_ms: 0 },
                };
            }
        }

        // MISS / REFRESH — ретраї з бекофом
        const started = Date.now();
        let attempts = 0;
        let lastError: any;

        for (let attempt = 1; attempt <= this.MAX_ATTEMPTS; attempt++) {
            attempts = attempt;
            try {
                const { data } = await lastValueFrom(
                    this.http.post(
                        `${this.base}${endpoint}`,
                        body,
                        {
                            timeout: this.TIMEOUT_MS,
                            headers: requestId ? { 'x-request-id': requestId } : undefined,
                        },
                    ),
                );
                const rt_ms = Date.now() - started;
                this._lruSet(key, { ts: Date.now(), data });
                return { data, cacheMeta: { hit: false, age_s: 0 }, httpMeta: { attempts, rt_ms } };
            } catch (e: any) {
                lastError = e;
                if (!this._isRetryable(e) || attempt === this.MAX_ATTEMPTS) {
                    throw this._wrapError(e);
                }
                const wait = this.BACKOFF_MS * Math.pow(2, attempt - 1);
                await this._delay(wait);
            }
        }

        throw this._wrapError(lastError);
    }

    private _isRetryable(e: any): boolean {
        const code = e?.code;
        const status = e?.response?.status;
        // мережеві помилки або серверні 5xx — ретраймо
        if (code && ['ECONNABORTED', 'ECONNRESET', 'ETIMEDOUT', 'EAI_AGAIN'].includes(code)) return true;
        if (typeof status === 'number' && status >= 500 && status < 600) return true;
        return false;
    }

    private _delay(ms: number) { return new Promise(res => setTimeout(res, ms)); }

    private _errInfo(e: any) {
        const status = e?.response?.status;
        const msg = e?.message || 'error';
        return status ? `${status} ${msg}` : msg;
    }

    private _wrapError(e: any) {
        const status = e?.response?.status ?? 500;
        const payload = e?.response?.data ?? { error: e?.message ?? 'ML service error' };
        if (status >= 400 && status < 600) return new HttpException(payload, status);
        return new InternalServerErrorException(payload);
    }

    private _lruSet(key: string, value: CacheEntry) {
        if (this.cache.has(key)) this.cache.delete(key);
        this.cache.set(key, value);
        if (this.cache.size > this.MAX) {
            const oldestKey = this.cache.keys().next().value as string;
            this.cache.delete(oldestKey);
        }
    }

    private _makeKey(endpoint: string, body: unknown): string {
        const canon = stableStringify(body);
        return crypto.createHash('sha256').update(endpoint + '|' + canon).digest('hex');
    }
}

// ---------- утиліта стабільної серіалізації ----------
function stableStringify(x: any): string {
    if (x === null || typeof x !== 'object') return JSON.stringify(x);
    if (Array.isArray(x)) return '[' + x.map(stableStringify).join(',') + ']';
    const keys = Object.keys(x).sort();
    return '{' + keys.map(k => JSON.stringify(k) + ':' + stableStringify(x[k])).join(',') + '}';
}
