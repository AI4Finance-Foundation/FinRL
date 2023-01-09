# FinRL Install Document

## M1 mac

### Run

```Shell
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install cmake openmpi

brew install swig
pip install box2d-py
pip install box2d
pip install Box2D

conda create -n RL python=3.10.8
conda activate RL

pip install -U "ray[default]"
pip install -U "ray[tune]"

git clone https://github.com/AI4Finance-Foundation/FinRL.git

cd FinRL

# 这一步之前需要修改文件，参加下一步修改文件
pip install .
pip install wrds

# 如果要用cuda
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# 这一步需要用VPN
python Stock_NeurIPS2018_SB3.py
```

![img](https://f3pmqowv13.feishu.cn/space/api/box/stream/download/asynccode/?code=MzM2Zjk0OTgzZjI3NmY2NTQ4ZWQyZWFjNDk5NzY5YmRfN09nRzMzRXV4a3RxTXFhZHZhOFZQUDJ4bGpsUWtkRTNfVG9rZW46Ym94Y25SUTFKOGtwR3NxRUw1MFlySmpqZkxoXzE2NzMyMzU2NTQ6MTY3MzIzOTI1NF9WNA)

## Windows

### Run

```Shell
conda create -n RL python=3.10.8
conda activate RL

conda install -c conda-forge "ray-default"
conda install -c conda-forge "ray-tune"
conda install swig

git clone https://github.com/AI4Finance-Foundation/FinRL.git
cd FinRL

# Before this step you need to modify the file, join the next step to modify the file
pip install .
pip install wrds

# 对于国内用户，这一步需要用VPN
python Stock_NeurIPS2018_SB3.py
```

![img](https://f3pmqowv13.feishu.cn/space/api/box/stream/download/asynccode/?code=NjgzNDEzOTc3MzY1ODNhNTZlZDRjYjRkODNiZmJiZWFfenJyR0hwaU5NSXlkaTV2cmFOVWg2N3RqODE2V3VkVUpfVG9rZW46Ym94Y25td1BDWjA5dk5nQ0pWQVZIcHg4QWI2XzE2NzMyMzU2NTQ6MTY3MzIzOTI1NF9WNA)

## Modify pyproject.toml

```TOML
 [tool.poetry]
 name = "finrl"
 version = "0.3.5"

 description = "FinRL: Financial Reinforcement Learning Framework. Version 0.3.5 notes: stable version, code refactoring, more tutorials, clear documentation"
 authors = ["Hongyang Yang, Xiaoyang Liu"]
 license = "MIT"

 readme = "README.md"
 classifiers=[
         # Trove classifiers
         # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
         "License :: OSI Approved :: MIT License",
         "Programming Language :: Python",
         "Programming Language :: Python :: 3",
         "Programming Language :: Python :: 3.7",
         "Programming Language :: Python :: 3.8",
         "Programming Language :: Python :: 3.9",
         "Programming Language :: Python :: 3.10",
         "Programming Language :: Python :: Implementation :: CPython",
         "Programming Language :: Python :: Implementation :: PyPy",
     ]
 keywords=["Reinforcement Learning", "Finance"]
 [tool.poetry.urls]
 github = "https://github.com/finrl/finrl-library"

 [tool.poetry.dependencies]
 python = ">=3.7" #">=3.7,<3.9"
 pyfolio = {git="https://github.com/quantopian/pyfolio.git#egg=pyfolio-0.9.2"}
 elegantrl = {git="https://github.com/AI4Finance-Foundation/ElegantRL.git#egg=elegantrl"}
 alpaca_trade_api = ">=2.1.0"
 ccxt = ">=1.66.32"
 exchange_calendars = "3.6.3"
 gputil = "*"
 gym = ">=0.17"
 importlib-metadata = "4.13.0"
 jqdatasdk = "*"
 lz4 = "*"
 matplotlib = "*"
 pandas = ">=1.1.5"
 numpy = ">=1.17.3"
 tensorboardX = "*"
 yfinance = "*"
 stockstats = ">=0.4.0"
 scikit-learn = ">=0.21.0"
 ray = {extras=["default", "tune"], version=">=1.8.0"}#, version="1.3.0"
 stable-baselines3 = "^1.6.2"

 [tool.poetry.dev-dependencies]
 pytest = "*"
 pre-commit = "*"

 [build-system]
 requires = ["poetry-core"]
 build-backend = "poetry.core.masonry.api"

```
