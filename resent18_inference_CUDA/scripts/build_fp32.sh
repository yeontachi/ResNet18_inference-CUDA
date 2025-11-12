#!/usr/bin/env bash
set -e
rm -rf build && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ../cpp
cmake --build . -j
