FROM nginx:alpine

RUN apk add --no-cache openssl

RUN mkdir -p /etc/nginx/ssl && \
    openssl req -x509 -nodes -days 365 \
        -newkey rsa:2048 \
        -keyout /etc/nginx/ssl/server.key \
        -out /etc/nginx/ssl/server.crt \
        -subj "/C=US/ST=CA/L=SantaClara/O=NVIDIA/CN=localhost"

ARG CLIENT_BASE_URL=https://nvidia.github.io/IsaacTeleop/client/main
RUN mkdir -p /usr/share/nginx/html && \
    wget -q "${CLIENT_BASE_URL}/index.html"   -O /usr/share/nginx/html/index.html && \
    wget -q "${CLIENT_BASE_URL}/bundle.js"    -O /usr/share/nginx/html/bundle.js && \
    wget -q "${CLIENT_BASE_URL}/favicon.ico"   -O /usr/share/nginx/html/favicon.ico && \
    wget -q "${CLIENT_BASE_URL}/168.bundle.js" -O /usr/share/nginx/html/168.bundle.js && \
    wget -q "${CLIENT_BASE_URL}/372.bundle.js" -O /usr/share/nginx/html/372.bundle.js && \
    wget -q "${CLIENT_BASE_URL}/427.bundle.js" -O /usr/share/nginx/html/427.bundle.js && \
    wget -q "${CLIENT_BASE_URL}/play-circle.svg"                  -O /usr/share/nginx/html/play-circle.svg && \
    wget -q "${CLIENT_BASE_URL}/arrow-uturn-left.svg"             -O /usr/share/nginx/html/arrow-uturn-left.svg && \
    wget -q "${CLIENT_BASE_URL}/arrow-left-start-on-rectangle.svg" -O /usr/share/nginx/html/arrow-left-start-on-rectangle.svg || true

COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080 8443
CMD ["nginx", "-g", "daemon off;"]
