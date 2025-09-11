-- CreateTable
CREATE TABLE "AnalysisRun" (
    "id" SERIAL NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "method" TEXT NOT NULL,
    "maxLag" INTEGER NOT NULL,
    "minOverlap" INTEGER NOT NULL,
    "edgeMin" DOUBLE PRECISION NOT NULL,
    "seriesIds" TEXT NOT NULL,

    CONSTRAINT "AnalysisRun_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AnalysisEdge" (
    "id" SERIAL NOT NULL,
    "runId" INTEGER NOT NULL,
    "sourceId" INTEGER NOT NULL,
    "targetId" INTEGER NOT NULL,
    "lag" INTEGER NOT NULL,
    "weight" DOUBLE PRECISION NOT NULL,

    CONSTRAINT "AnalysisEdge_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "AnalysisEdge_runId_idx" ON "AnalysisEdge"("runId");

-- CreateIndex
CREATE INDEX "AnalysisEdge_sourceId_targetId_idx" ON "AnalysisEdge"("sourceId", "targetId");

-- AddForeignKey
ALTER TABLE "AnalysisEdge" ADD CONSTRAINT "AnalysisEdge_runId_fkey" FOREIGN KEY ("runId") REFERENCES "AnalysisRun"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
