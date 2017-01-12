#!/bin/bash

function heat_eq {
dofSize=256
dimension=2;
proportions=4
origin="-2"

analyticFunction="sin-bumps"
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

./tnl-grid-setup --dimensions ${dimension} \
               --proportions-x ${proportions} \
               --proportions-y ${proportions} \
               --proportions-z ${proportions} \
               --origin-x ${origin} \
               --origin-y ${origin} \
               --origin-z ${origin} \
               --size-x ${dofSize} \
               --size-y ${dofSize} \
               --size-z ${dofSize} \

./tnl-init --mesh mesh.tnl \
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

time ./tnl-heat-equation --device mic --time-discretisation explicit \
                  --boundary-conditions-type dirichlet \
                  --boundary-conditions-constant 0.0 \
                  --discrete-solver euler \
                  --snapshot-period 0.0005 \
                  --final-time 0.04 \
                  --time-step 0.00005 #|grep Offload |grep Time | cut -d ']' -f5|cut -d '(' -f1| ./a.out

./tnl-view --mesh mesh.tnl \
         --input-files *.tnl \ 

seznam=`ls u-*.gplt`

for fname in $seznam ; do
   echo "Drawing $fname"
 gnuplot << EOF
     set terminal unknown
     set pm3d
     set data style lines
     #set view 33,33 #3D
     #unset xtics 
     #unset ytics
     #unset ztics
     unset border
     set output '$fname.png'
     #set yrange [-1.2:1.2]
     set zrange [-1.1:1.1]    
     set terminal png
     set title "Numerical solution" 
     splot '$fname' with line 
EOF
done

}


module load gcc-5.3.0 cmake-3.4.3 intel_parallel_studio_ex-2016.1
cd Release/bin
 export OFFLOAD_REPORT=0
#./tnlSatanExperimentalTest-dbg
#./tnlSatanExperimentalTest-dbg
# ./tnlSatanMICVectorExperimentalTest-dbg
 heat_eq
