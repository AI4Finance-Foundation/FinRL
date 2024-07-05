FROM python:3.10.12-slim

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .

RUN apt-get update -y -qq && apt-get install -y -qq cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig g++ git 
# RUN apt-get install -y vim iputils-ping telnet
RUN pip install --no-cache-dir -r requirements.txt

# RUN pip install git+https://github.com/fein-ai/FinRL.git