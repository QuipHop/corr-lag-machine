import { IsArray, IsInt, IsOptional, IsString, IsIn, IsObject } from "class-validator";

export class RunForecastDto {
    @IsString() datasetId!: string;
    @IsString() targetCode!: string;

    @IsArray() @IsString({ each: true })
    features!: string[];

    @IsObject()
    lags!: Record<string, number>; // { "CPI": 0, ... }

    @IsIn(["none", "diff1", "pct"])
    transform!: "none" | "diff1" | "pct";

    @IsInt() seasonS!: number; // 12
    @IsInt() horizon!: number; // 6..24 типово

    @IsOptional() @IsString() presetName?: string; // зберегти/оновити пресет
}
