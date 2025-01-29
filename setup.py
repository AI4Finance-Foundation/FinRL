from __future__ import annotations

from setuptools import find_packages
from setuptools import setup

# Read requirements.txt, ignore comments
try:
    REQUIRES = list()
    f = open("requirements.txt", "rb")
    for line in f.read().decode("utf-8").split("\n"):
        line = line.strip()
        if "#" in line:
            line = line[: line.find("#")].strip()
        if line:
            REQUIRES.append(line)
except FileNotFoundError:
    print("'requirements.txt' not found!")
    REQUIRES = list()

setup(
    name="FinRL",
    version="0.3.7",
    include_package_data=True,
    author="AI4Finance Foundation",
    author_email="contact@ai4finance.org",
    url="https://github.com/AI4Finance-Foundation/FinRL",
    license="MIT",
    packages=find_packages(),
    description="FinRL: Financial Reinforcement Learning Framework.",
    long_description="Version 0.3.5 notes: stable version, code refactoring, more tutorials, clear documentation",
    # It is developed by `AI4Finance`_. \
    # _AI4Finance: https://ai4finance.org/",
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
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords="Reinforcement Learning, Finance",
    platform=["any"],
    python_requires=">=3.7",
)
