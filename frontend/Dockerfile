FROM node:20-bookworm-slim AS build

WORKDIR /frontend
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG VITE_API_BASE_URL=
ARG VITE_WS_BASE=
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL \
    VITE_WS_BASE=$VITE_WS_BASE
RUN npm run build

FROM nginx:1.27-alpine AS runtime

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /frontend/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://127.0.0.1/ || exit 1
