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

@Module({
    imports: [
        ConfigModule.forRoot({ isGlobal: true }),
        PrismaModule,
        DatasetsModule,
        UploadsModule,
        CorellationModule, // <-- and here
    ],
    controllers: [
        HealthController,
        SeriesController,
        ObservationsController,
        MlController,
        IndicatorController,
        SeriesListController,
        AnalysisController,
    ],
})
export class AppModule { }
