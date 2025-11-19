import { Body, Controller, Get, HttpCode, Param, Post, Query } from '@nestjs/common';
import { AnalysisService } from './analysis.service';

@Controller('analysis')
export class AnalysisController {
    constructor(private readonly svc: AnalysisService) { }

    /** 1) Heatmap з БД */
    @Post('corr/heatmap-db')
    @HttpCode(200)
    corrHeatmapFromDb(@Body() body: {
        datasetId: string;
        targetCode: string;
        candidateCodes?: string[];
        method?: 'pearson' | 'spearman';
        minOverlap?: number;
        lag?: { min?: number; max?: number; ignoreZero?: boolean };
        resample?: any;
        topK?: number;
        transform?: 'none' | 'diff1' | 'pct';
        returnP?: boolean;
        fdrAlpha?: number;
    }) {
        return this.svc.corrHeatmapFromDb(body);
    }

    /** 2) SARIMAX backtest з БД */
    @Post('sarimax/backtest-db')
    @HttpCode(200)
    sarimaxBacktestFromDb(@Body() body: {
        datasetId: string;
        targetCode: string;
        featureCodes?: string[];                 // ← список фіч за Series.key
        lags?: Record<string, number>;
        resample?: any;
        transform?: 'none' | 'diff1' | 'pct';
        train?: any;                             // order/seasonal/auto_grid...
        backtest: { horizon: number; min_train: number; step: number; expanding: boolean };
        saveRun?: boolean;
        runName?: string;
    }) {
        return this.svc.sarimaxBacktestFromDb(body);
    }

    /** 3) SARIMAX forecast з БД */
    @Post('sarimax/forecast-db')
    @HttpCode(200)
    sarimaxForecastFromDb(@Body() body: {
        datasetId: string;
        targetCode: string;
        featureCodes?: string[];
        lags?: Record<string, number>;
        resample?: any;
        transform?: 'none' | 'diff1' | 'pct';
        train?: any;
        horizon: number;                         // ← обов'язковий
        return_pi?: boolean;
        alpha?: number;
        saveRun?: boolean;
        runName?: string;
    }) {
        return this.svc.sarimaxForecastFromDb(body);
    }

    @Post('recommend-features')
    @HttpCode(200)
    recommendFeatures(@Body() body: {
        datasetId: string;
        targetCode: string;
        candidateCodes?: string[];
        method?: 'pearson' | 'spearman';
        minOverlap?: number;
        lag?: { min?: number; max?: number; ignoreZero?: boolean };
        transform?: 'none' | 'diff1' | 'pct';
        edgeMin?: number;        // мінімальний |corr|
        maxLagAbs?: number;      // обмежити |lag| (напр. 3)
        topK?: number;           // максимум фіч
        fdrAlpha?: number;       // 0.1
    }) {
        return this.svc.recommendFeatures(body);
    }

    /** Forecast з БД + збереження у таблиці Forecast/ForecastPoint */
    @Post('sarimax/forecast-db/save')
    @HttpCode(200)
    sarimaxForecastAndSave(@Body() body: {
        datasetId: string;
        targetCode: string;
        featureCodes?: string[];
        lags?: Record<string, number>;
        resample?: any;
        transform?: 'none' | 'diff1' | 'pct';
        train?: any;
        horizon: number;
        return_pi?: boolean;
        alpha?: number;
        name?: string;  // назва запису Forecast
    }) {
        return this.svc.sarimaxForecastAndSave(body);
    }

    /** Список збережених прогнозів */
    @Get('forecasts')
    listForecasts(
        @Query('datasetId') datasetId: string,
        @Query('targetCode') targetCode?: string,
        @Query('limit') limit?: string,
    ) {
        return this.svc.listForecasts(datasetId, targetCode, limit ? Number(limit) : 20);
    }

    /** Один прогноз із точками */
    @Get('forecast/:id')
    getForecast(@Param('id') id: string) {
        return this.svc.getForecast(id);
    }

    @Get('datasets/:datasetId/series-keys')
    listSeriesKeys(@Param('datasetId') datasetId: string) {
        return this.svc.listSeriesKeys(datasetId);
    }
}
