-- CreateEnum
CREATE TYPE "AnalysisEndpoint" AS ENUM ('CORR_HEATMAP', 'CORR_LAG');

-- CreateEnum
CREATE TYPE "TransformMode" AS ENUM ('NONE', 'DIFF1', 'PCT');

-- CreateTable
CREATE TABLE "AnalysisAudit" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "requestId" TEXT NOT NULL,
    "endpoint" "AnalysisEndpoint" NOT NULL,
    "sha" TEXT NOT NULL,
    "seriesCount" INTEGER NOT NULL,
    "pointsCount" INTEGER NOT NULL,
    "minOverlap" INTEGER,
    "lagMin" INTEGER NOT NULL,
    "lagMax" INTEGER NOT NULL,
    "transform" "TransformMode" NOT NULL,
    "cacheHit" BOOLEAN NOT NULL,
    "cacheAgeSeconds" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "httpAttempts" INTEGER NOT NULL DEFAULT 0,
    "httpRtMs" INTEGER NOT NULL DEFAULT 0,
    "status" INTEGER NOT NULL,
    "error" TEXT,

    CONSTRAINT "AnalysisAudit_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "idx_audit_created_endpoint" ON "AnalysisAudit"("createdAt", "endpoint");
