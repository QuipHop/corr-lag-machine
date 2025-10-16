import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { CorrelationController } from './correlation.controller';
import { CorrelationService } from './correlation.service';
import { AuditService } from '../audit/audit.service';

@Module({
    imports: [
        HttpModule.register({
            timeout: 30000,
            maxRedirects: 0,
        }),
    ],
    controllers: [CorrelationController],
    providers: [CorrelationService, AuditService],
    exports: [CorrelationService, AuditService],
})
export class CorellationModule { }
