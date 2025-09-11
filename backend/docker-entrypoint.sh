set -e
npx prisma migrate deploy
echo "Starting Nest app..."
node dist/main.js