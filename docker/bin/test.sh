#!/bin/bash

docker run \
    --rm \
    -v "${PWD}":/home finrl pytest .
