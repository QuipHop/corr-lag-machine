# Multi-stage: build with Node, serve with Nginx
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
# Build-time API base; default to /api so nginx can proxy
ARG VITE_API_BASE=/api
ENV VITE_API_BASE=${VITE_API_BASE}
# Ensure Vite sees the var at build
RUN printf "VITE_API_BASE=%s" "$VITE_API_BASE" > .env.production
RUN npm run build


FROM nginx:1.27-alpine AS runner
WORKDIR /usr/share/nginx/html
COPY --from=build /app/dist .
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]