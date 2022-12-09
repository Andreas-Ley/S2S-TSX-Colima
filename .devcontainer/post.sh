#!/bin/sh
mkdir buildRel
cd buildRel
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j 2