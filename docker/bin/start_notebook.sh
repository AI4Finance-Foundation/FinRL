#!/bin/bash

docker run -it --rm -v "${PWD}":/src -p 8887:8888 finrl
