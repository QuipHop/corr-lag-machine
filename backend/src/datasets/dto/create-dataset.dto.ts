import { IsEnum, IsOptional, IsString } from 'class-validator';
import { Frequency } from '@prisma/client';

export class CreateDatasetDto {
    @IsString() name!: string;
    @IsEnum(Frequency) @IsOptional() freq?: Frequency; // monthly|quarterly|annual
}
