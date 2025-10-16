import { Injectable, HttpException } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';
import { firstValueFrom } from 'rxjs';
import type { AxiosError } from 'axios';

@Injectable()
export class MlService {
    private base: string;
    constructor(private readonly http: HttpService, cfg: ConfigService) {
        this.base = cfg.get<string>('ML_SVC_URL') ?? 'http://127.0.0.1:8000';
    }

    private async get<T>(path: string): Promise<T> {
        try {
            const { data } = await firstValueFrom(this.http.get<T>(`${this.base}${path}`));
            return data;
        } catch (err) {
            throw this._toHttpException(err, 'GET', path);
        }
    }

    private async post<T>(path: string, body: any): Promise<T> {
        try {
            const { data } = await firstValueFrom(this.http.post<T>(`${this.base}${path}`, body));
            return data;
        } catch (err) {
            throw this._toHttpException(err, 'POST', path, body);
        }
    }

    private _toHttpException(err: any, method: string, path: string, body?: any): HttpException {
        const e = err as AxiosError;
        const status = e.response?.status ?? 502;
        const data = e.response?.data ?? { message: e.message };
        // корисний payload у відгук — щоб дебажити з фронту/curl
        return new HttpException(
            { message: 'ML request failed', method, url: `${this.base}${path}`, status, error: data, body },
            status,
        );
    }

    health() { return this.get<{ status: string }>('/health'); }
    corrLag(p: any) { return this.post('/corr-lag', p); }
    corrHeatmap(p: any) { return this.post('/corr-heatmap', p); }
    sarimaxBacktest(p: any) { return this.post('/sarimax/backtest', p); }
    sarimaxForecast(p: any) { return this.post('/sarimax/forecast', p); }
}
