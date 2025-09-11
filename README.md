# 0) ensure Docker Desktop is running

# 1) DB + Adminer
cd infra
docker compose up -d postgres adminer

# 2) Apps with hot-reload
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build backend ml-svc frontend

# 3) Sanity checks
curl http://localhost:3000/health
curl http://localhost:8000/health
# open http://localhost:5173 (FE), http://localhost:8080 (Adminer)

# 4) When you edit code in backend/ml/frontend → containers auto-reload

# 5) View logs if needed
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f backend
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f ml-svc
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f frontend

# 6) Stop dev services
docker compose -f docker-compose.yml -f docker-compose.dev.yml down