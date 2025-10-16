import { Module } from '@nestjs/common';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { HttpModule } from '@nestjs/axios';

import { MlService } from './ml.service';
import { MlController } from './ml.controller';

@Module({
    imports: [
        // даємо доступ до .env в межах модуля
        ConfigModule,

        // реєструємо саме тут HttpModule, щоб HttpService був у контексті MlModule
        HttpModule.registerAsync({
            imports: [ConfigModule],
            inject: [ConfigService],
            useFactory: (cfg: ConfigService) => ({
                timeout: Number(cfg.get('ML_HTTP_TIMEOUT') ?? 15000),
                maxRedirects: 0,
                // тут можна додати baseURL: cfg.get('ML_SVC_URL') і тоді у сервісі викликати просто .get('/health')
                // але ми лишаємо baseURL у MlService для прозорості
            }),
        }),
    ],
    controllers: [MlController],
    providers: [MlService],
    exports: [MlService],
})
export class MlModule { }
