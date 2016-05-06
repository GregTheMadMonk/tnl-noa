#!/bin/bash

module load gcc-5.3.0 cmake-3.4.3 intel_parallel_studio_ex-2016.1
make -j 6 tnlSatanMICVectorExperimentalTest-dbg
#make -j 6 tnl-image-converter-dbg