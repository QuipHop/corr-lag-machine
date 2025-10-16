import { Type } from 'class-transformer';
import {
    IsArray, IsBoolean, IsIn, IsInt, IsNumber, IsOptional, IsString,
    Min, ValidateNested
} from 'class-validator';

export class PointDto {
    @IsString() date!: string;
    @IsNumber() value!: number;
}

export class SeriesInDto {
    @IsInt() id!: number;
    @IsString() code!: string;
    @IsArray() @ValidateNested({ each: true }) @Type(() => PointDto)
    points!: PointDto[];
}

export class ResampleCfgDto {
    @IsBoolean() enabled: boolean = true;
    @IsIn(['M']) freq: 'M' = 'M';
    @IsIn(['last', 'mean', 'sum']) downsample: 'last' | 'mean' | 'sum' = 'last';
    @IsIn(['ffill', 'bfill', 'interpolate', 'none']) upsample: 'ffill' | 'bfill' | 'interpolate' | 'none' = 'ffill';
    @IsNumber() @Min(0) winsorize_q: number = 0.0;
}

export class LagCfgDto {
    @IsInt() min: number = -12;
    @IsInt() max: number = 12;
    @IsBoolean() ignoreZero: boolean = false;
}

export class CorrLagRequestDto {
    @IsArray() @ValidateNested({ each: true }) @Type(() => SeriesInDto)
    series!: SeriesInDto[];

    // legacy
    @IsOptional() @IsInt() maxLag?: number;

    @IsIn(['pearson', 'spearman']) method: 'pearson' | 'spearman' = 'pearson';
    @IsInt() @Min(3) minOverlap: number = 12;
    @IsNumber() edgeMin: number = 0.3;

    @ValidateNested() @Type(() => ResampleCfgDto)
    resample: ResampleCfgDto = new ResampleCfgDto();

    @IsBoolean() normalizeOrientation: boolean = true;
    @IsBoolean() dedupeOpposite: boolean = true;
    @IsOptional() @IsInt() topK?: number;
    @IsOptional() @IsInt() perNodeTopK?: number;

    @IsOptional() @ValidateNested() @Type(() => LagCfgDto)
    lag?: LagCfgDto;

    @IsBoolean() returnStats: boolean = false;
    @IsIn(['none', 'diff1', 'pct']) transform: 'none' | 'diff1' | 'pct' = 'none';
}

export class CorrHeatmapRequestDto {
    @IsArray() @ValidateNested({ each: true }) @Type(() => SeriesInDto)
    series!: SeriesInDto[];

    @IsString() targetCode!: string;
    @IsOptional() @IsArray() @IsString({ each: true }) candidateCodes?: string[];

    @IsIn(['pearson', 'spearman']) method: 'pearson' | 'spearman' = 'pearson';
    @IsInt() @Min(3) minOverlap: number = 12;

    @ValidateNested() @Type(() => ResampleCfgDto)
    resample: ResampleCfgDto = new ResampleCfgDto();

    @ValidateNested() @Type(() => LagCfgDto)
    lag: LagCfgDto = new LagCfgDto();

    @IsOptional() @IsInt() topK?: number;

    @IsBoolean() returnStats: boolean = false;
    @IsIn(['none', 'diff1', 'pct']) transform: 'none' | 'diff1' | 'pct' = 'none';
}
