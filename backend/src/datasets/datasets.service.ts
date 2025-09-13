import { Injectable } from '@nestjs/common';
import { PrismaService } from '../shared/prisma.service';
import { CreateDatasetDto } from './dto/create-dataset.dto';
import { Frequency } from '@prisma/client';

@Injectable()
export class DatasetsService {
    constructor(private prisma: PrismaService) { }

    async create(dto: CreateDatasetDto) {
        return this.prisma.dataset.create({
            data: { name: dto.name, freq: dto.freq ?? Frequency.monthly },
            select: { id: true, name: true, freq: true, createdAt: true },
        });
    }
}
