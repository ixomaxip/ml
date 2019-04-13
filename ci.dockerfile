#https://raw.githubusercontent.com/docker-library/python/ce69fc6369feb8ec757b019035ddad7bac20562c/3.7/alpine3.9/Dockerfile
FROM python:3.7.2-alpine3.9

RUN apk add --no-cache curl
RUN pip install requests aiohttp fire

WORKDIR /workdir
EXPOSE 80

# Install required packages
RUN apk add --update --virtual=.build-dependencies alpine-sdk nodejs ca-certificates musl-dev gcc python-dev make cmake g++ gfortran libpng-dev freetype-dev libxml2-dev libxslt-dev 

RUN apk add --update git
RUN apk add --update libzmq

RUN apk add --update zeromq-dev
RUN pip install  pyzmq
# Install Jupyter
RUN pip install jupyter
RUN pip install ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension

RUN pip install pandas
# Install JupyterLab
RUN pip install jupyterlab && jupyter serverextension enable --py jupyterlab

# Expose Jupyter port & cmd
# RUN mkdir -p /opt/app/data
CMD jupyter lab --ip=0.0.0.0 --port=80 --no-browser --notebook-dir=/workdir --allow-root --LabApp.token=''
