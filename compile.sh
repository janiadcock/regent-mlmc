#!/bin/bash -eu

gcc -Wall -Wextra -pedantic -g -fPIC -c diffusion.c
gcc -shared -o libdiffusion.so diffusion.o
