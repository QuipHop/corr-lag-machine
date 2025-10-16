import { Type } from 'class-transformer';
import { IsArray, IsBoolean, IsIn, IsInt, IsNumber, IsOptional, IsString, ValidateNested } from 'class-validator';

class PointDto { @IsString() date!: string; @IsNumber() value!: number; }
class SeriesInDto { @IsInt() id!: number; @IsString() code!: string; @ValidateNested({ each: true }) @Type(() => PointDto) @IsArray() points!: PointDto[]; }
class ResampleCfgDto {
    @IsOptional() @IsBoolean() enabled?: boolean;
    @IsOptional() @IsIn(['M']) freq?: 'M';
    @IsOptional() @IsIn(['last', 'mean', 'sum']) downsample?: 'last' | 'mean' | 'sum';
    @IsOptional() @IsIn(['ffill', 'bfill', 'interpolate', 'none']) upsample?: 'ffill' | 'bfill' | 'interpolate' | 'none';
    @IsOptional() @IsNumber() winsorize_q?: number;
}
class LagCfgDto { @IsOptional() @IsInt() min?: number; @IsOptional() @IsInt() max?: number; @IsOptional() @IsBoolean() ignoreZero?: boolean; }

export class CorrLagDto {
    @ValidateNested({ each: true }) @Type(() => SeriesInDto) @IsArray()
    series!: SeriesInDto[];

    @IsOptional() @IsInt()
    maxLag?: number;

    @IsIn(['pearson', 'spearman'])
    method!: 'pearson' | 'spearman';

    @IsInt() minOverlap!: number;
    @IsNumber() edgeMin!: number;

    @ValidateNested() @Type(() => ResampleCfgDto)
    resample!: ResampleCfgDto;

    @IsOptional() @IsBoolean() normalizeOrientation?: boolean;
    @IsOptional() @IsBoolean() dedupeOpposite?: boolean;
    @IsOptional() @IsInt() topK?: number;
    @IsOptional() @IsInt() perNodeTopK?: number;

    @IsOptional() @ValidateNested() @Type(() => LagCfgDto)
    lag?: LagCfgDto;

    @IsOptional() @IsBoolean() returnStats?: boolean;

    @IsIn(['none', 'diff1', 'pct'])
    transform!: 'none' | 'diff1' | 'pct';

    // NEW:
    @IsOptional() @IsBoolean() returnP?: boolean;
    @IsOptional() @IsNumber() fdrAlpha?: number;
}
