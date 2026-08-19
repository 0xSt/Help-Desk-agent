# Immagine del servizio FRONTEND: nginx che serve le due pagine statiche e
# inoltra /api/ al backend (vedi docker/nginx.conf per il perché del proxy).
#
# Non c'è build step: le interfacce sono HTML/CSS/JS vanilla, senza framework
# né bundler. L'immagine è quindi solo nginx più tre file.

FROM nginx:1.27-alpine

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY app/static/ /usr/share/nginx/html/

EXPOSE 80

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/healthz || exit 1
