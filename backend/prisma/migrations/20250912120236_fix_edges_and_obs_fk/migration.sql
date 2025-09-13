/*
  Warnings:

  - The primary key for the `Series` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - You are about to drop the column `extraMeta` on the `Series` table. All the data in the column will be lost.
  - You are about to drop the column `frequency` on the `Series` table. All the data in the column will be lost.
  - You are about to drop the column `region` on the `Series` table. All the data in the column will be lost.
  - You are about to drop the column `updatedAt` on the `Series` table. All the data in the column will be lost.
  - A unique constraint covering the columns `[datasetId,key]` on the table `Series` will be added. If there are existing duplicate values, this will fail.
  - Added the required column `datasetId` to the `Series` table without a default value. This is not possible if the table is not empty.
  - Added the required column `key` to the `Series` table without a default value. This is not possible if the table is not empty.

*/
-- DropForeignKey
ALTER TABLE "Observation" DROP CONSTRAINT "Observation_seriesId_fkey";

-- DropForeignKey
ALTER TABLE "Series" DROP CONSTRAINT "Series_indicatorId_fkey";

-- AlterTable
ALTER TABLE "AnalysisEdge" ALTER COLUMN "sourceId" SET DATA TYPE TEXT,
ALTER COLUMN "targetId" SET DATA TYPE TEXT;

-- AlterTable
ALTER TABLE "Observation" ALTER COLUMN "seriesId" SET DATA TYPE TEXT;

-- AlterTable
ALTER TABLE "Series" DROP CONSTRAINT "Series_pkey",
DROP COLUMN "extraMeta",
DROP COLUMN "frequency",
DROP COLUMN "region",
DROP COLUMN "updatedAt",
ADD COLUMN     "datasetId" TEXT NOT NULL,
ADD COLUMN     "key" TEXT NOT NULL,
ADD COLUMN     "label" TEXT,
ADD COLUMN     "units" TEXT,
ALTER COLUMN "id" DROP DEFAULT,
ALTER COLUMN "id" SET DATA TYPE TEXT,
ALTER COLUMN "indicatorId" DROP NOT NULL,
ADD CONSTRAINT "Series_pkey" PRIMARY KEY ("id");
DROP SEQUENCE "Series_id_seq";

-- CreateTable
CREATE TABLE "Dataset" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "freq" "Frequency" NOT NULL DEFAULT 'monthly',
    "mappingJson" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Dataset_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "Series_datasetId_key_key" ON "Series"("datasetId", "key");

-- AddForeignKey
ALTER TABLE "Series" ADD CONSTRAINT "Series_datasetId_fkey" FOREIGN KEY ("datasetId") REFERENCES "Dataset"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Series" ADD CONSTRAINT "Series_indicatorId_fkey" FOREIGN KEY ("indicatorId") REFERENCES "Indicator"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Observation" ADD CONSTRAINT "Observation_seriesId_fkey" FOREIGN KEY ("seriesId") REFERENCES "Series"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AnalysisEdge" ADD CONSTRAINT "AnalysisEdge_sourceId_fkey" FOREIGN KEY ("sourceId") REFERENCES "Series"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AnalysisEdge" ADD CONSTRAINT "AnalysisEdge_targetId_fkey" FOREIGN KEY ("targetId") REFERENCES "Series"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
