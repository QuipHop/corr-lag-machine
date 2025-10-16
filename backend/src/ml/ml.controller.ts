import { Body, Controller, Get, HttpCode, Post, UsePipes, ValidationPipe } from '@nestjs/common';
import { MlService } from './ml.service';
import { CorrLagDto } from './dto/corr-lag.dto';
import { CorrHeatmapDto } from './dto/corr-heatmap.dto';
import { SarimaxBacktestDto } from './dto/sarimax-backtest.dto';
import { SarimaxForecastDto } from './dto/sarimax-forecast.dto';

@Controller('ml')
@UsePipes(new ValidationPipe({ transform: true, whitelist: true, forbidNonWhitelisted: false }))
export class MlController {
    constructor(private readonly svc: MlService) { }

    @Get('health')
    getHealth() { return this.svc.health(); }

    @Post('corr/lag')
    @HttpCode(200)
    corrLag(@Body() body: CorrLagDto) { return this.svc.corrLag(body); }

    @Post('corr/heatmap')
    @HttpCode(200)
    corrHeatmap(@Body() body: CorrHeatmapDto) { return this.svc.corrHeatmap(body); }

    @Post('sarimax/backtest')
    @HttpCode(200)
    backtest(@Body() body: SarimaxBacktestDto) { return this.svc.sarimaxBacktest(body); }

    @Post('sarimax/forecast')
    @HttpCode(200)
    forecast(@Body() body: SarimaxForecastDto) { return this.svc.sarimaxForecast(body); }
}
