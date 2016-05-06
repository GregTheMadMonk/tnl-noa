#!/bin/bash

module load gcc-5.3.0 cmake-3.4.3 intel_parallel_studio_ex-2016.1
cd Debug/bin
#OFFLOAD_REPORT=2 ./tnlSatanExperimentalTest-dbg
#./tnlSatanExperimentalTest-dbg
./tnlSatanMICVectorExperimentalTest-dbg