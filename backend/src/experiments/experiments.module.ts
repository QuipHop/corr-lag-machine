// src/experiments/experiments.module.ts
import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';

import { ExperimentsService } from './experiments.service';
import { ExperimentsController } from './experiments.controller';
import { PrismaModule } from '../shared/prisma.module';

@Module({
  imports: [PrismaModule, HttpModule],
  controllers: [ExperimentsController],
  providers: [ExperimentsService],
})
export class ExperimentsModule {}
