import { Type } from 'class-transformer';
import { IsArray, IsBoolean, IsIn, IsNumber, IsOptional, IsString, ValidateNested } from 'class-validator';

class ValueColDto {
    @IsString() name!: string;
    @IsString() key!: string;
    @IsOptional() @IsString() label?: string;
    @IsOptional() @IsString() units?: string;
}

export class PreviewDto {
    @IsIn(['auto', 'dot', 'comma']) decimal!: 'auto' | 'dot' | 'comma';
    @IsOptional() @IsBoolean() dropBlanks?: boolean;
    @IsOptional() @IsString() dateFormat?: string;

    // Long (default)
    @IsOptional() @IsString() dateColumn?: string;
    @IsOptional() @ValidateNested({ each: true }) @Type(() => ValueColDto)
    valueColumns?: ValueColDto[];

    // Wide (months are columns)
    @IsOptional() @IsIn(['long', 'wide']) shape?: 'long' | 'wide';
    @IsOptional() @IsString() seriesKeyColumn?: string;
    @IsOptional() @IsArray() monthColumns?: string[];  // optional override; will auto-detect if omitted
    @IsOptional() @IsNumber() year?: number;           // used if headers have no year
}
