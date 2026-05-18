FROM nginx:alpine

RUN apk add --no-cache openssl

RUN mkdir -p /etc/nginx/ssl && \
    openssl req -x509 -nodes -days 365 \
        -newkey rsa:2048 \
        -keyout /etc/nginx/ssl/server.key \
        -out /etc/nginx/ssl/server.crt \
        -subj "/C=US/ST=CA/L=SantaClara/O=NVIDIA/CN=localhost"

COPY nginx.conf /etc/nginx/templates/default.conf.template

EXPOSE 48322

CMD ["nginx", "-g", "daemon off;"]
