// src/experiments/experiments.controller.ts
import {
  Body,
  Controller,
  Get,
  Param,
  Post,
} from '@nestjs/common';
import { ExperimentsService } from './experiments.service';
import { RunExperimentDto } from './dto/run-experiment.dto';

@Controller('experiments')
export class ExperimentsController {
  constructor(private readonly service: ExperimentsService) {}

  @Post('run')
  async run(@Body() dto: RunExperimentDto) {
    return this.service.runExperiment(dto);
  }

  @Get()
  async list() {
    return this.service.listExperiments();
  }

  @Get(':id')
  async get(@Param('id') id: string) {
    return this.service.getExperiment(id);
  }

  @Get(':id/forecasts')
  async getForecasts(@Param('id') id: string) {
    return this.service.getForecasts(id);
  }

  @Get(':id/metrics')
  async getMetrics(@Param('id') id: string) {
    return this.service.getMetrics(id);
  }
}
