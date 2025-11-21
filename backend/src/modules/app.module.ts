// src/app.module.ts
import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { ThrottlerGuard, ThrottlerModule } from '@nestjs/throttler';
import { APP_GUARD } from '@nestjs/core';


import { HealthController } from './health.controller';
import { PrismaModule } from '../shared/prisma.module';
import { ExperimentsModule } from '../experiments/experiments.module';

@Module({
    imports: [
        ThrottlerModule.forRoot([
            {
                ttl: parseInt(process.env.RATE_LIMIT_TTL_SECONDS ?? '60', 10),
                limit: parseInt(process.env.RATE_LIMIT_LIMIT ?? '60', 10),
            },
        ]),
        ConfigModule.forRoot({ isGlobal: true }),
        PrismaModule,
        ExperimentsModule, // усередині вже є ExperimentsController + ExperimentsService
    ],
    controllers: [
        HealthController, // /health для docker-compose
    ],
    providers: [
        { provide: APP_GUARD, useClass: ThrottlerGuard },
    ],
})
export class AppModule { }
