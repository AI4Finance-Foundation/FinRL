FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install FinRL and its dependencies
RUN pip install --no-cache-dir \
    ccxt \
    pandas \
    numpy \
    gym \
    stable-baselines3 \
    torch \
    matplotlib \
    jupyterlab

# Add Tini for proper signal handling
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# Copy the application code
COPY . .

EXPOSE 8888

ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command starts Jupyter Lab
CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
