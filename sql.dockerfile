FROM mariadb:latest

ENV MYSQL_ROOT_PASSWORD=
ENV MYSQL_DATABASE=baseball
ENV MYSQL_USER=jchu
ENV MYSQL_PASSWORD=bda

COPY baseball.sql /docker-entrypoint-initdb.d/