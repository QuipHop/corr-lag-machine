// backend/src/main.ts
import { NestFactory } from '@nestjs/core';
import { AppModule } from './modules/app.module'; // шлях як у тебе
import { Logger } from '@nestjs/common';

async function bootstrap() {
    const app = await NestFactory.create(AppModule, { cors: false });

    // дозволяємо localhost дев-оріджини; у dev можна origin:true
    const isProd = process.env.NODE_ENV === 'production';
    const origins = (process.env.CORS_ORIGINS || 'http://localhost:5173,http://127.0.0.1:5173')
        .split(',')
        .map(s => s.trim());

    app.enableCors({
        origin: isProd ? origins : true,        // dev: будь-який origin
        credentials: true,
        methods: ['GET', 'HEAD', 'PUT', 'PATCH', 'POST', 'DELETE', 'OPTIONS'],
        allowedHeaders: ['Content-Type', 'Authorization', 'x-request-id'],
        // щоб FE міг прочитати ці заголовки через fetch().headers.get(...)
        exposedHeaders: ['x-request-id', 'x-ml-cache', 'x-ml-cache-age', 'x-ml-attempts', 'x-ml-rt-ms'],
    });

    const port = parseInt(process.env.PORT || '3000', 10);
    await app.listen(port, '0.0.0.0');
    Logger.log(`Backend listening on ${port}`);
}
bootstrap();
