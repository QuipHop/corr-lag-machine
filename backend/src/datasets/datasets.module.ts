// src/datasets/datasets.module.ts
import { Module } from '@nestjs/common';
import { DatasetsController } from './datasets.controller';
import { DatasetsService } from './datasets.service';
import { PrismaModule } from '../shared/prisma.module';

@Module({
    imports: [PrismaModule],
    controllers: [DatasetsController],
    providers: [DatasetsService],
    exports: [DatasetsService],
})
export class DatasetsModule { }
