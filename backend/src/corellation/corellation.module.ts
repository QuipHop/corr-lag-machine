import { Module /*, Global*/ } from '@nestjs/common';
import { PrismaModule } from '../shared/prisma.module';
import { CorrelationController } from './correlation.controller';
import { CorrelationService } from './correlation.service';

// If you want it available app-wide without importing, uncomment @Global()
// @Global()
@Module({
    imports: [PrismaModule],
    controllers: [CorrelationController],
    providers: [CorrelationService],
    exports: [CorrelationService],
})
export class CorellationModule { }
