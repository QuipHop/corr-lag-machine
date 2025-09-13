import { IsArray, IsBoolean, IsIn, IsOptional, IsString } from 'class-validator';

export class PreviewDto {
    @IsString() dateColumn!: string;
    @IsArray() valueColumns!: { name: string; key: string; label?: string; units?: string }[];
    @IsIn(['auto', 'dot', 'comma']) decimal: 'auto' | 'dot' | 'comma' = 'auto';
    @IsString() @IsOptional() dateFormat?: string;
    @IsBoolean() @IsOptional() dropBlanks?: boolean = true;
}
