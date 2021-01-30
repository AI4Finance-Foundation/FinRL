#!/bin/bash


docker run \
    -it \
    --rm \
    -v ${PWD}:/home finrl python3 -m unittest discover