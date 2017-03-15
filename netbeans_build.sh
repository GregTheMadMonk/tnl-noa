#!/bin/bash

module load gcc-5.3.0 cmake-3.4.3 intel_parallel_studio_ex-2016.1
cd Debug
make -j 6 tnlMICArrayTest-dbg
#make -j 6 tnlMICVectorTest-dbg
make -j 6 tnl-heat-equation-dbg