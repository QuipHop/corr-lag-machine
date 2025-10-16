import { Type } from 'class-transformer';
import { IsArray, IsBoolean, IsIn, IsInt, IsNumber, IsOptional, IsString, ValidateNested } from 'class-validator';

class PointDto { @IsString() date!: string; @IsNumber() value!: number; }
class SeriesInDto { @IsInt() id!: number; @IsString() code!: string; @ValidateNested({ each: true }) @Type(() => PointDto) @IsArray() points!: PointDto[]; }
class ResampleCfgDto { @IsOptional() @IsBoolean() enabled?: boolean; @IsOptional() @IsIn(['M']) freq?: 'M'; @IsOptional() @IsIn(['last', 'mean', 'sum']) downsample?: 'last' | 'mean' | 'sum'; @IsOptional() @IsIn(['ffill', 'bfill', 'interpolate', 'none']) upsample?: 'ffill' | 'bfill' | 'interpolate' | 'none'; @IsOptional() @IsNumber() winsorize_q?: number; }
class FeaturesCfgDto { @IsString() targetCode!: string; @IsOptional() @IsArray() @IsString({ each: true }) features?: string[]; @IsOptional() lags?: Record<string, number>; }
class OrderDto { @IsOptional() @IsInt() p?: number; @IsOptional() @IsInt() d?: number; @IsOptional() @IsInt() q?: number; }
class SOrderDto { @IsOptional() @IsInt() P?: number; @IsOptional() @IsInt() D?: number; @IsOptional() @IsInt() Q?: number; @IsOptional() @IsInt() s?: number; }
class AutoGridDto { @IsOptional() p?: [number, number]; @IsOptional() d?: [number, number]; @IsOptional() q?: [number, number]; @IsOptional() P?: [number, number]; @IsOptional() D?: [number, number]; @IsOptional() Q?: [number, number]; @IsOptional() @IsInt() s?: number; @IsOptional() @IsInt() max_models?: number; }
class TrainCfgDto {
    @IsOptional() @ValidateNested() @Type(() => OrderDto) order?: OrderDto;
    @IsOptional() @ValidateNested() @Type(() => SOrderDto) seasonal_order?: SOrderDto;
    @IsOptional() @IsIn(['n', 'c', 't', 'ct']) trend?: 'n' | 'c' | 't' | 'ct';
    @IsOptional() @IsBoolean() enforce_stationarity?: boolean;
    @IsOptional() @IsBoolean() enforce_invertibility?: boolean;
    @IsOptional() @ValidateNested() @Type(() => AutoGridDto) auto_grid?: AutoGridDto;
}

export class SarimaxForecastDto {
    @ValidateNested({ each: true }) @Type(() => SeriesInDto) @IsArray()
    series!: SeriesInDto[];

    @ValidateNested() @Type(() => ResampleCfgDto)
    resample!: ResampleCfgDto;

    @IsIn(['none', 'diff1', 'pct'])
    transform!: 'none' | 'diff1' | 'pct';

    @ValidateNested() @Type(() => FeaturesCfgDto)
    features_cfg!: FeaturesCfgDto;

    @ValidateNested() @Type(() => TrainCfgDto)
    train!: TrainCfgDto;

    @IsInt()
    horizon!: number;

    @IsOptional() @IsBoolean()
    return_pi?: boolean;

    @IsOptional() @IsNumber()
    alpha?: number;
}
