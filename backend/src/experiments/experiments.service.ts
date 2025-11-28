// src/experiments/experiments.service.ts
import {
  BadRequestException,
  Injectable,
  Logger,
} from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { firstValueFrom } from 'rxjs';
import { randomUUID } from 'crypto';

import { RunExperimentDto } from './dto/run-experiment.dto';
import { PrismaService } from '../shared/prisma.service';

// Типи під ml-svc (мають співпасти з Python schemas)
type MlSeriesPayload = {
  name: string;
  role: 'target' | 'candidate' | 'ignored';
  values: (number | null)[];
};

type MlExperimentRequest = {
  experiment_id: string;
  dates: string[];
  series: MlSeriesPayload[];
  frequency: 'M' | 'Q' | 'Y';
  horizon: number;
  imputation: 'none' | 'ffill' | 'bfill' | 'interp';
  max_lag: number;
  extra: Record<string, unknown>;
};

type MlModel = {
  seriesName: string;
  modelType: string;
  mase: number | null;
  smape: number | null;
  rmse: number | null;
  fit_time?: number | null;
  pred_time?: number | null;
  isSelected: boolean;
};

type MlForecastPoint = {
  seriesName: string;
  date: string;
  valueActual: number | null;
  valuePred: number | null;
  setType: string; // 'train' | 'test' | 'future'
};

type MlMetric = {
  seriesName: string;
  modelType: string;
  horizon: number;
  mase: number | null;
  smape: number | null;
  rmse: number | null;
};

type MlExperimentResult = {
  diagnostics: any;
  correlations: any;
  factors: any;
  models: MlModel[];
  forecasts: MlForecastPoint[];     // <-- тепер просто масив
  metrics: MlMetric[];              // <-- окрема таблиця метрик
};

@Injectable()
export class ExperimentsService {
  private readonly logger = new Logger(ExperimentsService.name);

  // TODO: винести в конфіг
  private readonly mlSvcUrl = process.env.ML_SVC_URL || 'http://ml-svc:8000';

  constructor(
    private readonly prisma: PrismaService,
    private readonly http: HttpService,
  ) { }

  async runExperiment(dto: RunExperimentDto) {
    const { name, context, dates, series, frequency, horizon } = dto;

    if (!name || !dates?.length || !series?.length) {
      throw new BadRequestException(
        'name, dates and series are required for experiment',
      );
    }

    // 1. Створюємо Experiment в БД
    const experimentId = randomUUID();

    await this.prisma.experiment.create({
      data: {
        id: experimentId,
        name,
        context: context ?? null,
        frequency,
        horizon,
        createdAt: new Date(),
        diagnostics: {},
        correlations: {},
        factors: {},
      },
    });

    // 2. Формуємо payload для ml-svc
    const seriesPayload: MlSeriesPayload[] = series.map((s) => {
      if (!s.name || !s.values) {
        throw new BadRequestException('Each series must have name and values');
      }

      return {
        name: s.name,
        role: (s.role ?? 'candidate') as 'target' | 'candidate' | 'ignored',
        values: s.values,
      };
    });

    const mlPayload: MlExperimentRequest = {
      experiment_id: experimentId,
      dates,
      series: seriesPayload,
      frequency,
      horizon,
      imputation: dto.imputation ?? 'ffill',
      max_lag: dto.maxLag ?? 12,
      extra: {},
    };

    // 3. Викликаємо ml-svc
    const url = `${this.mlSvcUrl}/experiment/run`;
    this.logger.log(
      `Calling ML service at ${url} for experiment ${experimentId}`,
    );

    const { data: result } = await firstValueFrom(
      this.http.post<MlExperimentResult>(url, mlPayload),
    );

    // 4. Зберігаємо результат у БД в транзакції
    await this.prisma.$transaction(async (tx) => {
      // Оновлюємо Experiment з diagnostics / correlations / factors
      await tx.experiment.update({
        where: { id: experimentId },
        data: {
          diagnostics: result.diagnostics,
          correlations: result.correlations,
          factors: result.factors,
          finishedAt: new Date(),
        },
      });

      // Models
      if (result.models?.length) {
        const modelRows = result.models
          .filter((m) => !!m.seriesName && !!m.modelType)
          .map((m) => ({
            experimentId,
            seriesName: m.seriesName,
            modelType: m.modelType,
            paramsJson: {}, // в Python params поки не заповнюємо
            mase: m.mase,
            smape: m.smape,
            rmse: m.rmse,
            isSelected: m.isSelected ?? false,
          }));

        if (modelRows.length) {
          await tx.model.createMany({ data: modelRows });
        }
      }

      // Forecasts
      if (result.forecasts?.length) {
        await tx.forecast.createMany({
          data: result.forecasts.map((f) => ({
            experimentId,
            seriesName: f.seriesName,
            date: new Date(f.date),
            valueActual: f.valueActual,
            valuePred: f.valuePred,
            lowerPi: null,
            upperPi: null,
            setType: f.setType,
          })),
        });
      }

      // Metrics
      if (result.metrics?.length) {
        await tx.experimentMetric.createMany({
          data: result.metrics.map((m) => ({
            experimentId,
            seriesName: m.seriesName,
            modelType: m.modelType,
            horizon: m.horizon,
            mase: m.mase ?? 0,
            smape: m.smape ?? 0,
            rmse: m.rmse ?? 0,
          })),
        });
      }
    });

    // 5. Повертаємо експеримент + основні метрики
    const saved = await this.prisma.experiment.findUnique({
      where: { id: experimentId },
      include: {
        metrics: true,
      },
    });

    return {
      experiment: saved,
      mlResult: result,
    };
  }

  async listExperiments() {
    return this.prisma.experiment.findMany({
      orderBy: { createdAt: 'desc' },
      include: {
        metrics: true,
      },
    });
  }

  async getExperiment(id: string) {
    return this.prisma.experiment.findUnique({
      where: { id },
      include: {
        models: true,
        metrics: true,
      },
    });
  }

  async getForecasts(id: string) {
    return this.prisma.forecast.findMany({
      where: { experimentId: id },
      orderBy: { date: 'asc' },
    });
  }

  async getMetrics(id: string) {
    return this.prisma.experimentMetric.findMany({
      where: { experimentId: id },
    });
  }
}
