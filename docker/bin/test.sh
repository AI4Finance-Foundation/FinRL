#!/bin/bash

docker run \
    --rm \
    -v "${PWD}":/home finrl python3 -m pytest . -v
