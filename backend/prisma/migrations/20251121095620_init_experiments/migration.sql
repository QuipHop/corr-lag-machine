/*
  Warnings:

  - You are about to drop the `AnalysisAudit` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `AnalysisEdge` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `AnalysisRun` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `Dataset` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `Indicator` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `Observation` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `Series` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropForeignKey
ALTER TABLE "AnalysisEdge" DROP CONSTRAINT "AnalysisEdge_runId_fkey";

-- DropForeignKey
ALTER TABLE "AnalysisEdge" DROP CONSTRAINT "AnalysisEdge_sourceId_fkey";

-- DropForeignKey
ALTER TABLE "AnalysisEdge" DROP CONSTRAINT "AnalysisEdge_targetId_fkey";

-- DropForeignKey
ALTER TABLE "AnalysisRun" DROP CONSTRAINT "AnalysisRun_datasetId_fkey";

-- DropForeignKey
ALTER TABLE "Observation" DROP CONSTRAINT "Observation_seriesId_fkey";

-- DropForeignKey
ALTER TABLE "Series" DROP CONSTRAINT "Series_datasetId_fkey";

-- DropForeignKey
ALTER TABLE "Series" DROP CONSTRAINT "Series_indicatorId_fkey";

-- DropTable
DROP TABLE "AnalysisAudit";

-- DropTable
DROP TABLE "AnalysisEdge";

-- DropTable
DROP TABLE "AnalysisRun";

-- DropTable
DROP TABLE "Dataset";

-- DropTable
DROP TABLE "Indicator";

-- DropTable
DROP TABLE "Observation";

-- DropTable
DROP TABLE "Series";

-- DropEnum
DROP TYPE "AnalysisEndpoint";

-- DropEnum
DROP TYPE "Frequency";

-- DropEnum
DROP TYPE "TransformMode";

-- CreateTable
CREATE TABLE "Experiment" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "context" TEXT,
    "frequency" TEXT NOT NULL,
    "horizon" INTEGER NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "finishedAt" TIMESTAMP(3),
    "diagnostics" JSONB NOT NULL,
    "correlations" JSONB NOT NULL,
    "factors" JSONB NOT NULL,

    CONSTRAINT "Experiment_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Model" (
    "id" TEXT NOT NULL,
    "experimentId" TEXT NOT NULL,
    "seriesName" TEXT NOT NULL,
    "modelType" TEXT NOT NULL,
    "paramsJson" JSONB NOT NULL,
    "mase" DOUBLE PRECISION,
    "smape" DOUBLE PRECISION,
    "rmse" DOUBLE PRECISION,
    "isSelected" BOOLEAN NOT NULL DEFAULT false,

    CONSTRAINT "Model_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Forecast" (
    "id" TEXT NOT NULL,
    "experimentId" TEXT NOT NULL,
    "seriesName" TEXT NOT NULL,
    "date" TIMESTAMP(3) NOT NULL,
    "valueActual" DOUBLE PRECISION,
    "valuePred" DOUBLE PRECISION,
    "lowerPi" DOUBLE PRECISION,
    "upperPi" DOUBLE PRECISION,
    "setType" TEXT NOT NULL,

    CONSTRAINT "Forecast_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ExperimentMetric" (
    "id" TEXT NOT NULL,
    "experimentId" TEXT NOT NULL,
    "seriesName" TEXT NOT NULL,
    "modelType" TEXT NOT NULL,
    "horizon" INTEGER NOT NULL,
    "mase" DOUBLE PRECISION NOT NULL,
    "smape" DOUBLE PRECISION NOT NULL,
    "rmse" DOUBLE PRECISION NOT NULL,

    CONSTRAINT "ExperimentMetric_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "Model" ADD CONSTRAINT "Model_experimentId_fkey" FOREIGN KEY ("experimentId") REFERENCES "Experiment"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Forecast" ADD CONSTRAINT "Forecast_experimentId_fkey" FOREIGN KEY ("experimentId") REFERENCES "Experiment"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ExperimentMetric" ADD CONSTRAINT "ExperimentMetric_experimentId_fkey" FOREIGN KEY ("experimentId") REFERENCES "Experiment"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
