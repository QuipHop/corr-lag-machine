import { Body, Controller, Post } from '@nestjs/common';
import { CorrelationService } from './correlation.service';
import { CorrelateDto } from './dto/correlate.dto';

@Controller('api/correlate')
export class CorrelationController {
    constructor(private readonly svc: CorrelationService) { }

    @Post()
    run(@Body() dto: CorrelateDto) {
        return this.svc.run(dto);
    }
}
