ARG CI
FROM $CI

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

# Install JupyterLab
RUN pip install jupyterlab && jupyter serverextension enable --py jupyterlab

# Expose Jupyter port & cmd
# RUN mkdir -p /opt/app/data
CMD jupyter lab --ip=0.0.0.0 --port=80 --no-browser --notebook-dir=/workdir --allow-root
