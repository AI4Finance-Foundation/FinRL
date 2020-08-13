from setuptools import setup, find_packages

# Read requirements.txt, ignore comments
try:
    REQUIRES = list()
    f = open("requirements.txt", "rb")
    for line in f.read().decode("utf-8").split("\n"):
        line = line.strip()
        if "#" in line:
            line = line[:line.find("#")].strip()
        if line:
            REQUIRES.append(line)
except:
    print("'requirements.txt' not found!")
    REQUIRES = list()

setup(
    name = "finrl",
    version = "0.0.1",
    include_package_data=True,
    author='Hongyang Yang, Xiaoyang Liu',
    author_email='hy2500@columbia.edu',
    url = "https://github.com/finrl/finrl-library" ,
    license = "MIT" ,
    packages = find_packages(),
    install_requires=REQUIRES,
    description = "FinRL library, a Deep Reinforcement Learning library designed specifically for automated stock trading.",
    classifiers = [
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    keywords = "Reinforcment Learning",
    platform=['any'],
    python_requires='>=3.6',
)
