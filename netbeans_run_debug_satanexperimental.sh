#!/bin/bash

module load gcc-5.3.0 cmake-3.4.3 intel_parallel_studio_ex-2016.1
cd Debug/bin
#export OFFLOAD_REPORT=2 
#./tnlSatanExperimentalTest-dbg
#./tnlSatanExperimentalTest-dbg
#./tnlSatanMICVectorExperimentalTest-dbg

dofSize=64
dimension=2;
proportions=1

analyticFunction="exp-bump"
timeFunction="cosinus"

amplitude=1.0
waveLength=1.0
waveLengthX=1.0
waveLengthY=1.0
waveLengthZ=1.0
wavesNumber=0.0
wavesNumberX=0.0
wavesNumberY=0.0
wavesNumberZ=0.0
phase=0.0
phaseX=0.0
phaseY=0.0
phaseZ=0.0
sigma=1.0

./tnl-grid-setup-dbg --dimensions ${dimension} \
               --proportions-x ${proportions} \
               --proportions-y ${proportions} \
               --proportions-z ${proportions} \
               --origin-x 0 \
               --origin-y 0 \
               --origin-z 0 \
               --size-x ${dofSize} \
               --size-y ${dofSize} \
               --size-z ${dofSize} \

./tnl-init-dbg --mesh mesh.tnl \
         --test-function ${analyticFunction} \
         --output-file initial.tnl \
         --amplitude ${amplitude} \
         --wave-length ${waveLength} \
         --wave-length-x ${waveLengthX} \
         --wave-length-y ${waveLengthY} \
         --wave-length-z ${waveLengthZ} \
             --waves-number ${wavesNumber} \
             --waves-number-x ${wavesNumberX} \
             --waves-number-y ${wavesNumberY} \
             --waves-number-z ${wavesNumberZ} \
             --phase ${phase} \
             --phase-x ${phaseX} \
             --phase-y ${phaseY} \
             --phase-z ${phaseZ} \
             --sigma ${sigma} \

mv ./initial.tnl init.tnl

./tnl-heat-equation-dbg --device mic --time-discretisation explicit \
                  --boundary-conditions-type dirichlet \
                  --boundary-conditions-constant 0.5 \
                  --discrete-solver euler \
                  --snapshot-period 0.0005 \
                  --final-time 0.1 \
                  --time-step 0.00005

 ./tnl-view-dbg --mesh mesh.tnl \
          --input-files *.tnl \ 

seznam=`ls u-*.gplt`

for fname in $seznam ; do
   echo "Drawing $fname"
 gnuplot << EOF
     set terminal unknown
     #set view 33,33 #3D
     #unset xtics 
     #unset ytics
     #unset ztics
     unset border
     set output '$fname.png'
     set yrange [-1.2:1.2]
     set zrange [0.4:1.1]    
     set terminal png
     set title "Numerical solution" 
     splot '$fname' with line 
EOF
done