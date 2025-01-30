FROM python:3.10.5
USER root
#convention to create an environment variable
#it will set the value this environment variable to 1
ENV PYTHONUNBUFFERED 1
RUN pip install --upgrade pip
RUN pip install swig

#create directory
#RUN apt-get update -y && \ apt-get install -y python-pip python-dev
RUN mkdir /src

#working directory
WORKDIR /src

#copy everything to this expected /code going to be created within oour container