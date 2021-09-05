from setuptools import setup, find_packages

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
except:
    print("'requirements.txt' not found!")
    REQUIRES = list()

setup(
    name="finrl",
    version="0.3.1",
    include_package_data=True,
    author="Hongyang Yang, Xiaoyang Liu",
    author_email="hy2500@columbia.edu",
    url="https://github.com/finrl/finrl-library",
    license="MIT",
    packages=find_packages(),
    install_requires=REQUIRES
    + ["pyfolio @ git+https://github.com/quantopian/pyfolio.git#egg=pyfolio-0.9.2"],
    # dependency_links=['git+https://github.com/quantopian/pyfolio.git#egg=pyfolio-0.9.2'],
    #install_requires=REQUIRES,
    description="FinRL library, a Deep Reinforcement Learning library designed specifically for automated stock trading.",
    long_description="""finrl is a Python library for that facilitates beginners to expose themselves to quantitative finance 
    and to develop their own trading strategies, it is developed by `AI4Finance`_. 
    
    FinRL has been developed under three primary principles: completeness, hands-on tutorial and reproducibility. 
    
    .. _AI4Finance: https://github.com/AI4Finance-Foundation
    """,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords="Reinforcment Learning",
    platform=["any"],
    python_requires=">=3.6",
)
