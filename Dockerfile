# Use an official Python runtime as a parent image
FROM python:3.10.12-slim

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
# ADD . /app

# Add models/a2c.zip file to Docker image
# ADD models/a2c.zip /app/models/
RUN apt-get update -y -qq && apt-get install -y -qq cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig g++ git 
# RUN apt-get install -y vim iputils-ping telnet
RUN pip install --no-cache-dir -r requirements.txt
# Install FinRL

RUN pip install git+https://github.com/fein-ai/FinRL.git

# Make port 80 available to the world outside this container
# EXPOSE 80

# Run app.py when the container launches