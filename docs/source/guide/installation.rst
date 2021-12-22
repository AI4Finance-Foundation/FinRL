:github_url: https://github.com/AI4Finance-LLC/FinRL-Library

Installation
=======================

Note: we made a tutorial for installation: [FinRL for Quantitative Finance: Install and Setup Tutorial for Beginners](https://ai4finance.medium.com/finrl-for-quantitative-finance-install-and-setup-tutorial-for-beginners-1db80ad39159).

Clone this repository

.. code-block:: python
    
    git clone https://github.com/AI4Finance-LLC/FinRL-Library.git

Install the unstable development version of FinRL:

.. code-block:: python

    pip install git+https://github.com/AI4Finance-LLC/FinRL-Library.git

**Prerequisites**

For OpenAI Baselines, you'll need system packages CMake, OpenMPI and zlib. Those can be installed as follows


**Ubuntu**

.. code-block:: python
    
    sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx

**Mac OS X**

Installation of system packages on Mac requires Homebrew. With Homebrew installed, run the following:

.. code-block:: python
    
    brew install cmake openmpi

**Windows 10**

To install stable-baselines on Windows, please look at the documentation_. 

.. _documentation: https://stable-baselines.readthedocs.io/en/master/guide/install.html#prerequisites

**Create and Activate Virtual Environment (Optional but highly recommended)**

cd into this repository

.. code-block:: python

    cd FinRL-Library

Under folder /FinRL-Library, create a virtual environment

.. code-block:: python
    
    pip install virtualenv

Virtualenvs are essentially folders that have copies of python executable and all python packages.

**Virtualenvs can also avoid packages conflicts.**

Create a virtualenv venv under folder /FinRL-Library

.. code-block:: python
    
    virtualenv -p python3 venv

To activate a virtualenv:

.. code-block:: python
    
    source venv/bin/activate

**Dependencies**

The script has been tested running under Python >= 3.6.0, with the folowing packages installed:

.. code-block:: python

    pip install -r requirements.txt

**Run**

.. code-block:: python

    python main.py --mode=train
