import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';

import { PrismaModule } from '../shared/prisma.module';
import { DatasetsModule } from '../datasets/datasets.module';
import { UploadsModule } from '../uploads/uploads.module';
import { CorellationModule } from '../corellation/corellation.module'; // <-- here

// your existing controllers
import { HealthController } from './health.controller';
import { SeriesController } from './series.controller';
import { ObservationsController } from './observations.controller';
import { MlController } from './ml.controller';
import { IndicatorController } from './indicator.controller';
import { SeriesListController } from './series.list.controller';
import { AnalysisController } from './analysis.controller';
import { ThrottlerGuard, ThrottlerModule } from '@nestjs/throttler';
import { APP_GUARD } from '@nestjs/core';
import { MlModule } from '../ml/ml.module';
import { AnalysisModule } from '../analysis/analysis.module';
import { ExperimentsModule } from '../experiments/experiments.module';
import { ExperimentsController } from '../experiments/experiments.controller';

@Module({
    imports: [
        ThrottlerModule.forRoot([{
            ttl: parseInt(process.env.RATE_LIMIT_TTL_SECONDS ?? '60', 10),
            limit: parseInt(process.env.RATE_LIMIT_LIMIT ?? '60', 10),
        }]),
        ConfigModule.forRoot({ isGlobal: true }),
        PrismaModule,
        DatasetsModule,
        UploadsModule,
        CorellationModule,
        MlModule,
        AnalysisModule,
        ExperimentsModule
    ],
    controllers: [
        HealthController,
        SeriesController,
        ObservationsController,
        MlController,
        IndicatorController,
        SeriesListController,
        AnalysisController,
        ExperimentsController
    ],
    providers: [
        { provide: APP_GUARD, useClass: ThrottlerGuard }
    ]
})
export class AppModule { }
