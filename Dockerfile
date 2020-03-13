FROM jupyter/tensorflow-notebook

USER root
WORKDIR /usr/src/app

COPY . .

RUN pip install pip==9.0.3
RUN pip install -r requirements.txt


EXPOSE 8888
