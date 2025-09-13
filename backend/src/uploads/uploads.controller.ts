import { Body, Controller, Headers, Param, Post, UploadedFile, UseInterceptors } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import type { Express } from 'express';
import { Multer } from 'multer';
import { UploadsService } from './uploads.service';
import { PreviewDto } from './dto/preview.dto';
import { CommitDto } from './dto/commit.dto';

@Controller('api')
export class UploadsController {
  constructor(private readonly svc: UploadsService) { }

  @Post('datasets/:datasetId/upload')
  @UseInterceptors(FileInterceptor('file', {
    limits: { fileSize: 20 * 1024 * 1024 },
    fileFilter: (_req, f, cb) => {
      const ok =
        /csv|excel|spreadsheetml/.test((f.mimetype || '').toLowerCase()) ||
        /\.(csv|xls|xlsx)$/i.test(f.originalname || '');
      cb(ok ? null : new Error('Unsupported file. Please upload CSV or Excel.'), ok);
    },
  }))
  upload(
    @Param('datasetId') id: string,
    @UploadedFile() file: Express.Multer.File,
    @Body() q: any
  ) {
    return this.svc.ingestTemp(id, file, q);
  }

  @Post('uploads/preview/:uploadId')
  preview(@Param('uploadId') up: string, @Body() dto: PreviewDto) {
    return this.svc.preview(up, dto);
  }

  @Post('uploads/commit/:uploadId')
  commit(
    @Param('uploadId') up: string,
    @Body() dto: CommitDto,
    @Headers('Idempotency-Key') idem?: string
  ) {
    return this.svc.commit(up, dto, idem);
  }
}
