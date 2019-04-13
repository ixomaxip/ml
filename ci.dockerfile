#https://raw.githubusercontent.com/docker-library/python/ce69fc6369feb8ec757b019035ddad7bac20562c/3.7/alpine3.9/Dockerfile
FROM python:3.7.2-alpine3.9

RUN apk add --no-cache curl

#ARG DOCKER_VERSION=18.06.1-ce
#RUN curl -fsSL https://download.docker.com/linux/static/stable/`uname -m`/docker-$DOCKER_VERSION.tgz | tar --strip-components=1 -xz -C /usr/local/bin docker/docker

RUN pip install requests aiohttp fire

#+10MB
#RUN apk update && \
# apk add postgresql-libs && \
# apk add --virtual .build-deps gcc musl-dev postgresql-dev && \
# pip install psycopg2 --no-cache-dir && \
# apk --purge del .build-deps

#RUN apk add --virtual b-deps python3-dev build-base && \
#pip3 install asyncpg && \
#apk del b-deps

#RUN pip install aiochclient
# RUN pip install aiohttp_utils

WORKDIR /workdir
EXPOSE 80
#CMD python main.py
#COPY config.json /root/.docker/config.json

# ENTRYPOINT [ "python" ]
# CMD ["main.py"]
