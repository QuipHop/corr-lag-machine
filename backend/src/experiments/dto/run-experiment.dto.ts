import {
  IsArray,
  IsIn,
  IsInt,
  IsNotEmpty,
  IsOptional,
  IsString,
  ValidateNested,
} from 'class-validator';
import { Type } from 'class-transformer';

export class SeriesDto {
  @IsString()
  @IsNotEmpty()
  name!: string;   // ← ОБОВʼЯЗКОВЕ поле

  @IsIn(['target', 'candidate', 'ignored'])
  role!: 'target' | 'candidate' | 'ignored';   // ← ОБОВʼЯЗКОВЕ поле

  @IsArray()
  values!: (number | null)[];   // ← ОБОВʼЯЗКОВЕ поле
}

export class RunExperimentDto {
  @IsString()
  @IsNotEmpty()
  name!: string;

  @IsOptional()
  @IsString()
  context?: string;

  @IsArray()
  dates!: string[];

  @IsArray()
  @ValidateNested({ each: true })
  @Type(() => SeriesDto)
  series!: SeriesDto[];

  @IsIn(['M', 'Q', 'Y'])
  frequency!: 'M' | 'Q' | 'Y';

  @IsInt()
  horizon!: number;

  @IsOptional()
  @IsIn(['none', 'ffill', 'bfill', 'interp'])
  imputation?: 'none' | 'ffill' | 'bfill' | 'interp';

  @IsOptional()
  @IsInt()
  maxLag?: number;
}
