# Contribution Guidelines
This project aims to bring a reinforcement learning environment to the trading community. 
There are always competing priorities among the community, and we want to make sure that we are able to achieve together a project that is reliable, sustainable, and maintainable. 

## Guiding Principles (v1)
* We should have reliable codes in this project
    * reliable code with tests
    * reliable code that works
    * reliable code runs without consuming excessive resources
* We should help each other to achieve SOTA results together
* We should write clear codes
    * Code should not be redundant
    * Code should have documentation inline (standard pep format)
    * Code should be organized into classes and functions
* We should leverage outside tools as it makes sense
* We work together, and are kind, patient, and clear in our communication. Jerks are not welcome. 

## If you see something, say something!
* Filing an [issue](https://guides.github.com/features/issues/) is a great way to help improve the project


## We will accept PR's for the following reasons
* You found a bug and a way to fix it
* You have contributed to an issue that was prioritized by the coordinators of this project
* You have new functionality that you're adding that you've written issues for and has documentation + Tests

## PR Guidelines
* Please tag @bruceyang, @spencerromo, or @xiaoyang in every PR. (P.S. we're looking for more collaborators with software experience!)
* Please reference or write and reference an [issue](https://guides.github.com/features/issues/) 
* Please have clear commit messages
* Please write detailed documentation and tests for every added piece of functionality
* Please try to not break existing functionality, or if you need to, please plan to justify this necessity and coordinate with the collaborators
* Please be patient and respectful with feedback
* Please use pre-commit hooks 


## Using pre-commit
```
pip install pre-commit
pre-commit install
```

## Running Tests
```
# Locally
python3 -m unittest discover

# Docker
./docker/bin/build_container.sh
./docker/bin/test.sh
```


