FROM stablebaselines/rl-baselines3-zoo

WORKDIR /src
RUN conda create -n py39 python=3.9 -y

COPY requirements.txt .
RUN /opt/conda/envs/py39/bin/python -m pip install -r requirements.txt -I

RUN /opt/conda/envs/py39/bin/python -m  pip install psutil==5.9.0
RUN /opt/conda/envs/py39/bin/python -m pip install jupyterlab
RUN /opt/conda/envs/py39/bin/python -m pip install ipykernel
RUN /opt/conda/envs/py39/bin/python -m ipykernel install --name=py39
RUN /opt/conda/envs/py39/bin/python -m pip install git+https://github.com/AI4Finance-Foundation/FinRL.git

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# copy the code instead of mounting
COPY . .

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["/opt/conda/envs/py39/bin/python","-m","jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--NotebookApp.token=''","--NotebookApp.password=''","--allow-root"]
