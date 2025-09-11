import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { PrismaService } from '../shared/prisma.service';
import { HealthController } from './health.controller';
import { SeriesController } from './series.controller';
import { ObservationsController } from './observations.controller';
import { MlController } from './ml.controller';
import { IndicatorController } from './indicator.controller';
import { SeriesListController } from './series.list.controller';
import { AnalysisController } from './analysis.controller';


@Module({
    imports: [ConfigModule.forRoot({ isGlobal: true })],
    controllers: [HealthController, SeriesController, ObservationsController, MlController, IndicatorController, SeriesListController, AnalysisController],
    providers: [PrismaService],
})
export class AppModule { }