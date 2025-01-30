FROM gitpod/workspace-full

USER gitpod

WORKDIR /src

COPY requirements.txt .

RUN pip install -r requirements.txt -I

RUN pip install jupsyterlab

# Install custom tools, runtime, etc. using apt-get
# For example, the command below would install "bastet" - a command line tetris clone:
#
# RUN sudo apt-get -q update && \
#     sudo apt-get install -yq bastet && \
#     sudo rm -rf /var/lib/apt/lists/*
#
# More information: https://www.gitpod.io/docs/config-docker/
