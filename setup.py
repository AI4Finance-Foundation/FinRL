from setuptools import find_packages, setup

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
    name="finrl",
    version="0.3.5",
    include_package_data=True,
    author="Hongyang Yang, Xiaoyang Liu",
    author_email="hy2500@columbia.edu",
    url="https://github.com/finrl/finrl-library",
    license="MIT",
    packages=find_packages(),
    install_requires=REQUIRES
                     + ["pyfolio @ git+https://github.com/quantopian/pyfolio.git#egg=pyfolio-0.9.2"]
                     + [
                         "elegantrl @ git+https://github.com/AI4Finance-Foundation/ElegantRL.git#egg=elegantrl"
                     ],
    # install_requires=REQUIRES,
    description="FinRL: Financial Reinforcement Learning Framework.",
    # long_description="FinRL is the first open-source framework to demonstrate the great potential of  \
    # applying deep reinforcement learning in quantitative finance. It is developed by `AI4Finance`_. \
    # _AI4Finance: https://github.com/AI4Finance-Foundation",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords="Reinforcement Learning, Finance",
    platform=["any"],
    python_requires=">=3.6",
)
