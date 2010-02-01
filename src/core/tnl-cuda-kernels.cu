/***************************************************************************
                          tnl-cuda-kernels.cu
                             -------------------
    begin                : Jan 14, 2010
    copyright            : (C) 2009 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include <iostream>
#include <core/mfuncs.h>
#include <core/tnl-cuda-kernels.h>

using namespace std;

int tnlCUDAReductionMin( const int size,
                         const int* input,
                         int& result,
                         int* device_aux_array = 0 )
{
   return tnlCUDAReduction< int, tnlMin >( size, input, result, device_aux_array );
}

int tnlCUDAReductionMax( const int size,
                         const int* input,
                         int& result,
                         int* device_aux_array = 0 )
{
   return tnlCUDAReduction< int, tnlMax >( size, input, result, device_aux_array );
}

int tnlCUDAReductionSum( const int size,
                         const int* input,
                         int& result,
                         int* device_aux_array = 0 )
{
   return tnlCUDAReduction< int, tnlSum >( size, input, result, device_aux_array );
}

bool tnlCUDAReductionMin( const int size,
                          const float* input,
                          float& result,
                          float* device_aux_array = 0 )
{
   return tnlCUDAReduction< float, tnlMin >( size, input, result, device_aux_array );
}

bool tnlCUDAReductionMax( const int size,
                          const float* input,
                          float& result,
                          float* device_aux_array = 0 )
{
   return tnlCUDAReduction< float, tnlMax >( size, input, result, device_aux_array );
}

bool tnlCUDAReductionSum( const int size,
                          const float* input,
                          float& result,
                          float* device_aux_array = 0 )
{
   return tnlCUDAReduction< float, tnlSum >( size, input, result, device_aux_array );
}
bool tnlCUDAReductionMin( const int size,
                          const double* input,
                          double& result,
                          double* device_aux_array = 0 )
{
   return tnlCUDAReduction< double, tnlMin >( size, input, result, device_aux_array );
}

bool tnlCUDAReductionMax( const int size,
                          const double* input,
                          double& result,
                          double* device_aux_array = 0 )
{
   return tnlCUDAReduction< double, tnlMax >( size, input, result, device_aux_array );
}

bool tnlCUDAReductionSum( const int size,
                          const double* input,
                          double& result,
                          double* device_aux_array = 0 )
{
   return tnlCUDAReduction< double, tnlSum >( size, input, result, device_aux_array );
}

/*
 * Simple redcution 5
 */

bool tnlCUDASimpleReduction5Min( const int size,
                                 const int* device_input,
                                 int& result,
                                 int* device_aux = 0 )
{
   return tnlCUDASimpleReduction5< int, tnlMin >( size,
                                                  device_input,
                                                  result,
                                                  device_aux );
}

bool tnlCUDASimpleReduction5Max( const int size,
                                 const int* device_input,
                                 int& result,
                                 int* device_aux = 0 )
{
   return tnlCUDASimpleReduction5< int, tnlMax >( size,
                                                  device_input,
                                                  result,
                                                  device_aux );
}
bool tnlCUDASimpleReduction5Sum( const int size,
                                 const int* device_input,
                                 int& result,
                                 int* device_aux = 0 )
{
   return tnlCUDASimpleReduction5< int, tnlSum >( size,
                                                  device_input,
                                                  result,
                                                  device_aux );
}

bool tnlCUDASimpleReduction5Min( const int size,
                                 const float* device_input,
                                 float& result,
                                 float* device_aux = 0 )
{
   return tnlCUDASimpleReduction5< float, tnlMin >( size,
                                                    device_input,
                                                    result,
                                                    device_aux );
}

bool tnlCUDASimpleReduction5Max( const int size,
                                 const float* device_input,
                                 float& result,
                                 float* device_aux = 0 )
{
   return tnlCUDASimpleReduction5< float, tnlMax >( size,
                                                    device_input,
                                                    result,
                                                    device_aux );
}
bool tnlCUDASimpleReduction5Sum( const int size,
                                 const float* device_input,
                                 float& result,
                                 float* device_aux = 0 )
{
   return tnlCUDASimpleReduction5< float, tnlSum >( size,
                                                    device_input,
                                                    result,
                                                    device_aux );
}
bool tnlCUDASimpleReduction5Min( const int size,
                                 const double* device_input,
                                 double& result,
                                 double* device_aux = 0 )
{
   return tnlCUDASimpleReduction5< double, tnlMin >( size,
                                                     device_input,
                                                     result,
                                                     device_aux );
}

bool tnlCUDASimpleReduction5Max( const int size,
                                 const double* device_input,
                                 double& result,
                                 double* device_aux = 0 )
{
   return tnlCUDASimpleReduction5< double, tnlMax >( size,
                                                     device_input,
                                                     result,
                                                     device_aux );
}
bool tnlCUDASimpleReduction5Sum( const int size,
                                 const double* device_input,
                                 double& result,
                                 double* device_aux = 0 )
{
   return tnlCUDASimpleReduction5< double, tnlSum >( size,
                                                     device_input,
                                                     result,
                                                     device_aux );
}


/*
 * Simple redcution 4
 */

bool tnlCUDASimpleReduction4Min( const int size,
                                 const int* device_input,
                                 int& result,
                                 int* device_aux = 0 )
{
   return tnlCUDASimpleReduction4< int, tnlMin >( size,
                                                  device_input,
                                                  result,
                                                  device_aux );
}

bool tnlCUDASimpleReduction4Max( const int size,
                                 const int* device_input,
                                 int& result,
                                 int* device_aux = 0 )
{
   return tnlCUDASimpleReduction4< int, tnlMax >( size,
                                                  device_input,
                                                  result,
                                                  device_aux );
}
bool tnlCUDASimpleReduction4Sum( const int size,
                                 const int* device_input,
                                 int& result,
                                 int* device_aux = 0 )
{
   return tnlCUDASimpleReduction4< int, tnlSum >( size,
                                                  device_input,
                                                  result,
                                                  device_aux );
}

bool tnlCUDASimpleReduction4Min( const int size,
                                 const float* device_input,
                                 float& result,
                                 float* device_aux = 0 )
{
   return tnlCUDASimpleReduction4< float, tnlMin >( size,
                                                    device_input,
                                                    result,
                                                    device_aux );
}

bool tnlCUDASimpleReduction4Max( const int size,
                                 const float* device_input,
                                 float& result,
                                 float* device_aux = 0 )
{
   return tnlCUDASimpleReduction4< float, tnlMax >( size,
                                                    device_input,
                                                    result,
                                                    device_aux );
}
bool tnlCUDASimpleReduction4Sum( const int size,
                                 const float* device_input,
                                 float& result,
                                 float* device_aux = 0  )
{
   return tnlCUDASimpleReduction4< float, tnlSum >( size,
                                                    device_input,
                                                    result,
                                                    device_aux );
}
bool tnlCUDASimpleReduction4Min( const int size,
                                 const double* device_input,
                                 double& result,
                                 double* device_aux = 0  )
{
   return tnlCUDASimpleReduction4< double, tnlMin >( size,
                                                     device_input,
                                                     result,
                                                     device_aux );
}

bool tnlCUDASimpleReduction4Max( const int size,
                                 const double* device_input,
                                 double& result,
                                 double* device_aux = 0  )
{
   return tnlCUDASimpleReduction4< double, tnlMax >( size,
                                                     device_input,
                                                     result,
                                                     device_aux );
}
bool tnlCUDASimpleReduction4Sum( const int size,
                                 const double* device_input,
                                 double& result,
                                 double* device_aux = 0  )
{
   return tnlCUDASimpleReduction4< double, tnlSum >( size,
                                                     device_input,
                                                     result,
                                                     device_aux );
}

/*
 * Simple redcution 3
 */

bool tnlCUDASimpleReduction3Min( const int size,
                                 const int* device_input,
                                 int& result,
                                 int* device_aux = 0  )
{
   return tnlCUDASimpleReduction3< int, tnlMin >( size,
                                                  device_input,
                                                  result,
                                                  device_aux );
}

bool tnlCUDASimpleReduction3Max( const int size,
                                 const int* device_input,
                                 int& result,
                                 int* device_aux = 0  )
{
   return tnlCUDASimpleReduction3< int, tnlMax >( size,
                                                  device_input,
                                                  result,
                                                  device_aux );
}
bool tnlCUDASimpleReduction3Sum( const int size,
                                 const int* device_input,
                                 int& result,
                                 int* device_aux = 0  )
{
   return tnlCUDASimpleReduction3< int, tnlSum >( size,
                                                  device_input,
                                                  result,
                                                  device_aux );
}

bool tnlCUDASimpleReduction3Min( const int size,
                                 const float* device_input,
                                 float& result,
                                 float* device_aux = 0  )
{
   return tnlCUDASimpleReduction3< float, tnlMin >( size,
                                                    device_input,
                                                    result,
                                                    device_aux );
}

bool tnlCUDASimpleReduction3Max( const int size,
                                 const float* device_input,
                                 float& result,
                                 float* device_aux = 0  )
{
   return tnlCUDASimpleReduction3< float, tnlMax >( size,
                                                    device_input,
                                                    result,
                                                    device_aux );
}
bool tnlCUDASimpleReduction3Sum( const int size,
                                 const float* device_input,
                                 float& result,
                                 float* device_aux = 0  )
{
   return tnlCUDASimpleReduction3< float, tnlSum >( size,
                                                    device_input,
                                                    result,
                                                    device_aux );
}
bool tnlCUDASimpleReduction3Min( const int size,
                                 const double* device_input,
                                 double& result,
                                 double* device_aux = 0  )
{
   return tnlCUDASimpleReduction3< double, tnlMin >( size,
                                                     device_input,
                                                     result,
                                                     device_aux );
}

bool tnlCUDASimpleReduction3Max( const int size,
                                 const double* device_input,
                                 double& result,
                                 double* device_aux = 0  )
{
   return tnlCUDASimpleReduction3< double, tnlMax >( size,
                                                     device_input,
                                                     result,
                                                     device_aux );
}
bool tnlCUDASimpleReduction3Sum( const int size,
                                 const double* device_input,
                                 double& result,
                                 double* device_aux = 0  )
{
   return tnlCUDASimpleReduction3< double, tnlSum >( size,
                                                     device_input,
                                                     result,
                                                     device_aux );
}

/*
 * Simple redcution 2
 */

bool tnlCUDASimpleReduction2Min( const int size,
                                 const int* device_input,
                                 int& result,
                                 int* device_aux = 0  )
{
   return tnlCUDASimpleReduction2< int, tnlMin >( size,
                                                  device_input,
                                                  result,
                                                  device_aux );
}

bool tnlCUDASimpleReduction2Max( const int size,
                                 const int* device_input,
                                 int& result,
                                 int* device_aux = 0 )
{
   return tnlCUDASimpleReduction2< int, tnlMax >( size,
                                                  device_input,
                                                  result,
                                                  device_aux );
}
bool tnlCUDASimpleReduction2Sum( const int size,
                                 const int* device_input,
                                 int& result,
                                 int* device_aux = 0 )
{
   return tnlCUDASimpleReduction2< int, tnlSum >( size,
                                                  device_input,
                                                  result,
                                                  device_aux );
}

bool tnlCUDASimpleReduction2Min( const int size,
                                 const float* device_input,
                                 float& result,
                                 float* device_aux = 0  )
{
   return tnlCUDASimpleReduction2< float, tnlMin >( size,
                                                    device_input,
                                                    result,
                                                    device_aux );
}

bool tnlCUDASimpleReduction2Max( const int size,
                                 const float* device_input,
                                 float& result,
                                 float* device_aux = 0  )
{
   return tnlCUDASimpleReduction2< float, tnlMax >( size,
                                                    device_input,
                                                    result,
                                                    device_aux );
}
bool tnlCUDASimpleReduction2Sum( const int size,
                                 const float* device_input,
                                 float& result,
                                 float* device_aux = 0  )
{
   return tnlCUDASimpleReduction2< float, tnlSum >( size,
                                                    device_input,
                                                    result,
                                                    device_aux );
}
bool tnlCUDASimpleReduction2Min( const int size,
                                 const double* device_input,
                                 double& result,
                                 double* device_aux = 0  )
{
   return tnlCUDASimpleReduction2< double, tnlMin >( size,
                                                     device_input,
                                                     result,
                                                     device_aux );
}

bool tnlCUDASimpleReduction2Max( const int size,
                                 const double* device_input,
                                 double& result,
                                 double* device_aux = 0  )
{
   return tnlCUDASimpleReduction2< double, tnlMax >( size,
                                                     device_input,
                                                     result );
}
bool tnlCUDASimpleReduction2Sum( const int size,
                                 const double* device_input,
                                 double& result,
                                 double* device_aux = 0  )
{
   return tnlCUDASimpleReduction2< double, tnlSum >( size,
                                                     device_input,
                                                     result,
                                                     device_aux );
}


/*
 * Simple redcution 1
 */

bool tnlCUDASimpleReduction1Min( const int size,
                                 const int* device_input,
                                 int& result,
                                 int* device_aux = 0 )
{
   return tnlCUDASimpleReduction1< int, tnlMin >( size,
                                                  device_input,
                                                  result,
                                                  device_aux );
}

bool tnlCUDASimpleReduction1Max( const int size,
                                 const int* device_input,
                                 int& result,
                                 int* device_aux = 0 )
{
   return tnlCUDASimpleReduction1< int, tnlMax >( size,
                                                  device_input,
                                                  result,
                                                  device_aux );
}
bool tnlCUDASimpleReduction1Sum( const int size,
                                 const int* device_input,
                                 int& result,
                                 int* device_aux = 0 )
{
   return tnlCUDASimpleReduction1< int, tnlSum >( size,
                                                  device_input,
                                                  result,
                                                  device_aux );
}

bool tnlCUDASimpleReduction1Min( const int size,
                                 const float* device_input,
                                 float& result,
                                 float* device_aux = 0 )
{
   return tnlCUDASimpleReduction1< float, tnlMin >( size,
                                                    device_input,
                                                    result,
                                                    device_aux );
}

bool tnlCUDASimpleReduction1Max( const int size,
                                 const float* device_input,
                                 float& result,
                                 float* device_aux = 0 )
{
   return tnlCUDASimpleReduction1< float, tnlMax >( size,
                                                    device_input,
                                                    result,
                                                    device_aux );
}
bool tnlCUDASimpleReduction1Sum( const int size,
                                 const float* device_input,
                                 float& result,
                                 float* device_aux = 0 )
{
   return tnlCUDASimpleReduction1< float, tnlSum >( size,
                                                    device_input,
                                                    result,
                                                    device_aux );
}
bool tnlCUDASimpleReduction1Min( const int size,
                                 const double* device_input,
                                 double& result,
                                 double* device_aux = 0 )
{
   return tnlCUDASimpleReduction1< double, tnlMin >( size,
                                                     device_input,
                                                     result,
                                                     device_aux );
}

bool tnlCUDASimpleReduction1Max( const int size,
                                 const double* device_input,
                                 double& result,
                                 double* device_aux = 0 )
{
   return tnlCUDASimpleReduction1< double, tnlMax >( size,
                                                     device_input,
                                                     result,
                                                     device_aux );
}
bool tnlCUDASimpleReduction1Sum( const int size,
                                 const double* device_input,
                                 double& result,
                                 double* device_aux = 0 )
{
   return tnlCUDASimpleReduction1< double, tnlSum >( size,
                                                     device_input,
                                                     result,
                                                     device_aux );
}

/*******************************************************************
 * oroginal code by J. Vacata
 */

/*
uint numIter = 100;
uint sizeRed;
uint desBlockSize = 128;    //Desired block size
uint desGridSize = 2048;    //Impose limitation on grid size so that threads could perform sequential work

uint shmemBB;
uint cpuTresh = 1;          //Determine, how many values will be reduced on the host



int testReduction1( int size,
                    int* drp_input,
                    int* drp_output,
                    int* output )
{
    int result = 0;
    int sizeBB = size*sizeof(int);
    //Calculate necessary block/grid dimensions
    dim3 blockSize(min(size, desBlockSize));
    dim3 gridSize(size/blockSize.x);
    shmemBB = blockSize.x * sizeof(int);
    reductKern1<<< gridSize, blockSize, shmemBB >>>(drp_output, drp_input);
    sizeRed = gridSize.x;
    while(sizeRed > cpuTresh) {
        blockSize.x = min(sizeRed, desBlockSize);
        gridSize.x = sizeRed/blockSize.x;
        shmemBB = blockSize.x * sizeof(int);
        reductKern1<<< gridSize, blockSize, shmemBB >>>(drp_output, drp_output);
        sizeRed = gridSize.x;
    }    
    cudaMemcpy(output, drp_output, sizeRed*sizeof(int), cudaMemcpyDeviceToHost);
    for (uint cnt=1;cnt<=sizeRed;cnt++) {
        result+=output[cnt-1];
    }
    return result;
}

int testReduction2( int size,
                    int* drp_input,
                    int* drp_output,
                    int* output )
{
    int result = 0;
    int sizeBB = size*sizeof(int);
    //Calculate necessary block/grid dimensions
    dim3 blockSize(min(size, desBlockSize));
    dim3 gridSize(size/blockSize.x);
    shmemBB = blockSize.x * sizeof(int);
    reductKern2<<< gridSize, blockSize, shmemBB >>>(drp_output, drp_input);
    sizeRed = gridSize.x;
    while(sizeRed > cpuTresh) {
        blockSize.x = min(sizeRed, desBlockSize);
        gridSize.x = sizeRed/blockSize.x;
        shmemBB = blockSize.x * sizeof(int);
        reductKern2<<< gridSize, blockSize, shmemBB >>>(drp_output, drp_output);
        sizeRed = gridSize.x;
    }    
    cudaMemcpy(output, drp_output, sizeRed*sizeof(int), cudaMemcpyDeviceToHost);
    for (uint cnt=1;cnt<=sizeRed;cnt++) {
        result+=output[cnt-1];
    }
    return result;
}

int testReduction3( int size,
                    int* drp_input,
                    int* drp_output,
                    int* output )
{
    int result = 0;
    int sizeBB = size*sizeof(int);
    //Calculate necessary block/grid dimensions
    dim3 blockSize(min(size, desBlockSize));
    dim3 gridSize(size/blockSize.x);
    shmemBB = blockSize.x * sizeof(int);
    reductKern3<<< gridSize, blockSize, shmemBB >>>(drp_output, drp_input);
    sizeRed = gridSize.x;
    while(sizeRed > cpuTresh) {
        blockSize.x = min(sizeRed, desBlockSize);
        gridSize.x = sizeRed/blockSize.x;
        shmemBB = blockSize.x * sizeof(int);
        reductKern3<<< gridSize, blockSize, shmemBB >>>(drp_output, drp_output);
        sizeRed = gridSize.x;
    }    
    cudaMemcpy(output, drp_output, sizeRed*sizeof(int), cudaMemcpyDeviceToHost);
    for (uint cnt=1;cnt<=sizeRed;cnt++) {
        result+=output[cnt-1];
    }
    return result;
}

int testReduction4( int size,
                    int* drp_input,
                    int* drp_output,
                    int* output )
{
    int result = 0;
    int sizeBB = size*sizeof(int);
    //Calculate necessary block/grid dimensions
    dim3 blockSize(min(size/2, desBlockSize));
    //Half the grid size so that each thread performs at least one reduction
    dim3 gridSize(size/blockSize.x/2);
    shmemBB = blockSize.x * sizeof(int);
    reductKern4<<< gridSize, blockSize, shmemBB >>>(drp_output, drp_input);
    sizeRed = gridSize.x;
    while(sizeRed > cpuTresh) {
        blockSize.x = min(sizeRed/2, desBlockSize);
        gridSize.x = sizeRed/blockSize.x/2;
        shmemBB = blockSize.x * sizeof(int);
        reductKern4<<< gridSize, blockSize, shmemBB >>>(drp_output, drp_output);
        sizeRed = gridSize.x;
    }    
    cudaMemcpy(output, drp_output, sizeRed*sizeof(int), cudaMemcpyDeviceToHost);
    for (uint cnt=1;cnt<=sizeRed;cnt++) {
        result+=output[cnt-1];
    }
    return result;
}

void reductionKernel5Switch(int* dp_output, int* dp_input, uint gridSz, uint blockSz, uint shmemBB) {
    dim3 blockSize(blockSz);
    dim3 gridSize(gridSz);
    //Switch statement allows proper choice of function template instance
    switch(blockSz) {
    case 512:
        reductKern5<512><<< gridSize, blockSize, shmemBB >>>(dp_output, dp_input); break;
    case 256:
        reductKern5<256><<< gridSize, blockSize, shmemBB >>>(dp_output, dp_input); break;
    case 128:
        reductKern5<128><<< gridSize, blockSize, shmemBB >>>(dp_output, dp_input); break;
    case  64:
        reductKern5< 64><<< gridSize, blockSize, shmemBB >>>(dp_output, dp_input); break;
    case  32:
        reductKern5< 32><<< gridSize, blockSize, shmemBB >>>(dp_output, dp_input); break;
    case  16:
        reductKern5< 16><<< gridSize, blockSize, shmemBB >>>(dp_output, dp_input); break;
    case   8:
        reductKern5<  8><<< gridSize, blockSize, shmemBB >>>(dp_output, dp_input); break;
    case   4:
        reductKern5<  4><<< gridSize, blockSize, shmemBB >>>(dp_output, dp_input); break;
    case   2:
        reductKern5<  2><<< gridSize, blockSize, shmemBB >>>(dp_output, dp_input); break;
    case   1:
        reductKern5<  1><<< gridSize, blockSize, shmemBB >>>(dp_output, dp_input); break;
    
    }
}

int testReduction5( int size,
                    int* drp_input,
                    int* drp_output,
                    int* output )
{
    int result = 0;
    int sizeBB = size*sizeof(int);
    //Calculate necessary block/grid dimensions
    uint blockSize = min(size/2, desBlockSize);
    uint gridSize = size/blockSize/2;
    shmemBB = blockSize * sizeof(int);
    reductionKernel5Switch(drp_output, drp_input, gridSize, blockSize, shmemBB);
    sizeRed = gridSize;
    while(sizeRed > cpuTresh) {
        blockSize = min(sizeRed/2, desBlockSize);
        gridSize = sizeRed/blockSize/2;
        shmemBB = blockSize * sizeof(int);
        reductionKernel5Switch(drp_output, drp_output, gridSize, blockSize, shmemBB);
        sizeRed = gridSize;
    }    
    cudaMemcpy(output, drp_output, sizeRed*sizeof(int), cudaMemcpyDeviceToHost);
    for (uint cnt=1;cnt<=sizeRed;cnt++) {
        result+=output[cnt-1];
    }
    return result;
}

void reductionKernel6Switch(uint psize, int* dp_output, int* dp_input, uint gridSz, uint blockSz, uint shmemBB) {
    dim3 blockSize(blockSz);
    dim3 gridSize(gridSz);
    switch(blockSz) {
    case 512:
        reductKern6<512><<< gridSize, blockSize, shmemBB >>>(dp_output, dp_input, psize); break;
    case 256:
        reductKern6<256><<< gridSize, blockSize, shmemBB >>>(dp_output, dp_input, psize); break;
    case 128:
        reductKern6<128><<< gridSize, blockSize, shmemBB >>>(dp_output, dp_input, psize); break;
    case  64:
        reductKern6< 64><<< gridSize, blockSize, shmemBB >>>(dp_output, dp_input, psize); break;
    case  32:
        reductKern6< 32><<< gridSize, blockSize, shmemBB >>>(dp_output, dp_input, psize); break;
    case  16:
        reductKern6< 16><<< gridSize, blockSize, shmemBB >>>(dp_output, dp_input, psize); break;
    case   8:
        reductKern6<  8><<< gridSize, blockSize, shmemBB >>>(dp_output, dp_input, psize); break;
    case   4:
        reductKern6<  4><<< gridSize, blockSize, shmemBB >>>(dp_output, dp_input, psize); break;
    case   2:
        reductKern6<  2><<< gridSize, blockSize, shmemBB >>>(dp_output, dp_input, psize); break;
    case   1:
        reductKern6<  1><<< gridSize, blockSize, shmemBB >>>(dp_output, dp_input, psize); break;
    
    }
}

int testReduction6( int size, 
                    int* drp_input,
                    int* drp_output,
                    int* output )
{
    int result = 0;
    int sizeBB = size*sizeof(int);
    //Calculate necessary block/grid dimensions
    uint blockSize = min(size/2, desBlockSize);
    //Grid size is limited in this case
    uint gridSize = min(desGridSize, size/blockSize/2);
    shmemBB = blockSize * sizeof(int);
    reductionKernel6Switch(size, drp_output, drp_input, gridSize, blockSize, shmemBB);
    sizeRed = gridSize;
    while(sizeRed > cpuTresh) {
        blockSize = min(sizeRed/2, desBlockSize);
        gridSize = min(desGridSize, sizeRed/blockSize/2);
        shmemBB = blockSize * sizeof(int);
        // cout << "Size: " << sizeRed
        //   << " Grid size: " << gridSize
        //   << " Block size: " << blockSize
        //   << " Shmem: " << shmemBB << endl;
        reductionKernel6Switch(sizeRed, drp_output, drp_output, gridSize, blockSize, shmemBB);
        sizeRed = gridSize;
    }    
    cudaMemcpy(output, drp_output, sizeRed*sizeof(int), cudaMemcpyDeviceToHost);
    for (uint cnt=1;cnt<=sizeRed;cnt++) {
        result+=output[cnt-1];
    }
    return result;
}
*/
