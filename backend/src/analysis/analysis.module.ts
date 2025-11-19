import { Module } from '@nestjs/common';
import { AnalysisController } from './analysis.controller';
import { AnalysisService } from './analysis.service';
import { PrismaModule } from '../shared/prisma.module';
import { MlModule } from '../ml/ml.module';

@Module({
    imports: [PrismaModule, MlModule],
    controllers: [AnalysisController],
    providers: [AnalysisService],
})
export class AnalysisModule { }
