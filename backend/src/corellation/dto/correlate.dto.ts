import { IsArray, IsBoolean, IsIn, IsOptional, IsString } from 'class-validator';

export class CorrelateDto {
    @IsString() datasetId!: string;
    @IsArray() series!: string[]; // series keys
    @IsIn(['spearman', 'pearson']) method: 'spearman' | 'pearson' = 'spearman';
    @IsOptional() transforms?: Record<string, { type: 'none' | 'pct_change' | 'diff' | 'log' | 'zscore' }>;
    @IsBoolean() @IsOptional() pearsonAlso?: boolean = false;
}
