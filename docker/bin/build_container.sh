#!/bin/bash

usage() {
    echo "Usage:"
    echo "" 
    echo "  build_container [--cpu]"
    echo "     --cpu    builds finRL docker container with CPU-only support"
    echo ""
}

# If no arguments => default build
if [ $# -eq 0 ]
then
    docker build -f docker/Dockerfile -t finrl docker/
    exit $?
fi

# if number of argument is 1 and the first argument is equal to --cpu => cpu mode
# otherwise display help
if [ $# -eq 1 ] && [ $1 == '--cpu' ] 
then 
    echo "Building CPU container..."
    docker build -f docker/Dockerfile-cpu -t finrl docker/
    exit $?
else 
    usage
fi
