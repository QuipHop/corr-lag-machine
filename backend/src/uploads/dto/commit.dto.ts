import { IsBoolean, IsIn } from 'class-validator';

export class CommitDto {
    @IsBoolean() saveMappingToDataset!: boolean;
    @IsBoolean() createSeries!: boolean;
    @IsIn(['replace', 'merge']) upsertMode!: 'replace' | 'merge';
}
