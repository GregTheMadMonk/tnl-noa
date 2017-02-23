#!/bin/bash

function heat_eq {
ext=""
#cd Debug/bin
#ext="-dbg"

export LD_LIBRARY_PATH=`pwd`"/../lib":$LD_LIBRARY_PATH

#echo $LD_LIBRARY_PATH
#ldd ./tnl-heat-equation${ext}

dofSize=512
#procnum=4 #useless for MIC

#export OMP_NUM_THREADS=2
export MIC_OMP_NUM_THREADS=120
export MIC_ENV_PREFIX=MIC



proportions=`echo "$dofSize/64"|bc`
origin=`echo "$proportions/-2"|bc`
dimension=2;



analyticFunction="sin-bumps"

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

./tnl-grid-setup${ext} --dimensions ${dimension} \
               --proportions-x ${proportions} \
               --proportions-y ${proportions} \
               --proportions-z ${proportions} \
               --origin-x ${origin} \
               --origin-y ${origin} \
               --origin-z ${origin} \
               --size-x ${dofSize} \
               --size-y ${dofSize} \
               --size-z ${dofSize}

./tnl-init${ext} --mesh mesh.tnl \
         --test-function ${analyticFunction} \
         --output-file init.tnl \
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
             --sigma ${sigma}

./tnl-heat-equation${ext} --device mic \
                  --boundary-conditions-type dirichlet \
                  --boundary-conditions-constant 0.0 \
                  --time-discretisation explicit \
                  --discrete-solver euler \
                  --snapshot-period 0.005 \
                  --final-time 0.04 \
                  --time-step 0.00005 \
                  --time-step-order 0 \
                  --openmp-enabled true \
                  --openmp-max-threads 8 

./tnl-view${ext} --mesh mesh.tnl \
         --input-files *.tnl \ 

seznam=`ls u-*.gplt`

for fname in $seznam ; 
do
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
#./tnlSatanMICVectorExperimentalTest-dbg
 heat_eq
