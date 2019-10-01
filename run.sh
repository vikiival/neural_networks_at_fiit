#!/bin/bash
open /Applications/Docker.app;
docker run -u $(id -u):$(id -g) -p 8888:8888 -v $(pwd):/labs -it mpikuliak/nsiete
