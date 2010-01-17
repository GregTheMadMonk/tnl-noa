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
                         const int block_size,
                         const int grid_size,
                         const int* input )
{
   return tnlCUDAReduction< int, tnlMin >( size, block_size, grid_size, input );
}

int tnlCUDAReductionMax( const int size,
                         const int block_size,
                         const int grid_size,
                         const int* input )
{
   return tnlCUDAReduction< int, tnlMax >( size, block_size, grid_size, input );
}
                         
int tnlCUDAReductionSum( const int size,
                         const int block_size,
                         const int grid_size,
                         const int* input )
{
   return tnlCUDAReduction< int, tnlSum >( size, block_size, grid_size, input );
}


float tnlCUDAReductionMin( const int size,
                           const int block_size,
                           const int grid_size,
                           const float* input )
{
   return tnlCUDAReduction< float, tnlMin >( size, block_size, grid_size, input );
}

float tnlCUDAReductionMax( const int size,
                           const int block_size,
                           const int grid_size,
                           const float* input )
{
   return tnlCUDAReduction< float, tnlMax >( size, block_size, grid_size, input );
}
                         
float tnlCUDAReductionSum( const int size,
                           const int block_size,
                           const int grid_size,
                           const float* input )
{
   return tnlCUDAReduction< float, tnlSum >( size, block_size, grid_size, input );
}

double tnlCUDAReductionMin( const int size,
                            const int block_size,
                            const int grid_size,
                            const double* input )
{
   return tnlCUDAReduction< double, tnlMin >( size, block_size, grid_size, input );
}

double tnlCUDAReductionMax( const int size,
                            const int block_size,
                            const int grid_size,
                            const double* input )
{
   return tnlCUDAReduction< double, tnlMax >( size, block_size, grid_size, input );
}
                         
double tnlCUDAReductionSum( const int size,
                            const int block_size,
                            const int grid_size,
                            const double* input )
{
   return tnlCUDAReduction< double, tnlSum >( size, block_size, grid_size, input );
}

/*
 * Simple redcution 1
 */

int tnlCUDASimpleReduction1Min( const int size,
                                const int block_size,
                                const int grid_size,
                                const int* input,
                                int* output )
{

}

int tnlCUDASimpleReduction1Max( const int size,
                                const int block_size,
                                const int grid_size,
                                const int* input,
                                int* output )
{
   //Calculate necessary block/grid dimensions
   const int cpuThreshold = 1;
   const int desBlockSize = 128;    //Desired block size   
   dim3 blockSize = :: Min( size, desBlockSize );
   dim3 gridSize = size / blockSize. x;
   unsigned int shmem = blockSize. x * sizeof( int );
   cout << "Grid size: " << gridSize. x << endl 
        << "Block size: " << blockSize. x << endl
        << "Shmem: " << shmem << endl;
   tnlCUDASimpleReductionKernel1< int, tnlMax ><<< gridSize, blockSize, shmem >>>( size, input, output );
   int sizeReduced = gridSize. x;
   while( sizeReduced > cpuThreshold )
   {
      cout << "Reducing with size reduced = " << sizeReduced << endl;
      blockSize. x = :: Min( sizeReduced, desBlockSize );
      gridSize. x = sizeReduced / blockSize. x;
      shmem = blockSize. x * sizeof(int);
      tnlCUDASimpleReductionKernel1< int, tnlMax ><<< gridSize, blockSize, shmem >>>( size, input, output );
      sizeReduced = gridSize. x;
   }
   int* host_output = new int[ sizeReduced ];
   cudaMemcpy( host_output, output, sizeReduced * sizeof(int), cudaMemcpyDeviceToHost );
   int result = host_output[ 0 ];
   for( int i = 1;i < sizeReduced; i++ )
        result = :: Max( result, host_output[ i ] );
   delete[] host_output;
   return result;
}
                         
int tnlCUDASimpleReduction1Sum( const int size,
                                const int block_size,
                                const int grid_size,
                                const int* input,
                                int* output )
{
   //Calculate necessary block/grid dimensions
   const int cpuThreshold = 1;
   const int desBlockSize = 128;    //Desired block size   
   dim3 blockSize = :: Min( size, desBlockSize );
   dim3 gridSize = size / blockSize. x;
   unsigned int shmem = blockSize. x * sizeof( int );
   cout << "Grid size: " << gridSize. x << endl 
        << "Block size: " << blockSize. x << endl
        << "Shmem: " << shmem << endl;
   tnlCUDASimpleReductionKernel1< int, tnlSum ><<< gridSize, blockSize, shmem >>>( size, input, output );
   int sizeReduced = gridSize. x;
   while( sizeReduced > cpuThreshold )
   {
      cout << "Reducing with size reduced = " << sizeReduced << endl;
      blockSize. x = :: Min( sizeReduced, desBlockSize );
      gridSize. x = sizeReduced / blockSize. x;
      shmem = blockSize. x * sizeof(int);
      tnlCUDASimpleReductionKernel1< int, tnlSum ><<< gridSize, blockSize, shmem >>>( size, input, output );
      sizeReduced = gridSize. x;
   }
   int* host_output = new int[ sizeReduced ];
   cudaMemcpy( host_output, output, sizeReduced * sizeof(int), cudaMemcpyDeviceToHost );
   int result = host_output[ 0 ];
   for( int i = 1;i < sizeReduced; i++ )
        result += host_output[ i ];
   delete[] host_output;
   return result;  
}

/*
float tnlCUDASimpleReduction1Min( const int size,
                                  const int block_size,
                                  const int grid_size,
                                  const float* input )
{
   return tnlCUDASimpleReduction1< float, tnlMin >( size, block_size, grid_size, input );
}

float tnlCUDASimpleReduction1Max( const int size,
                                  const int block_size,
                                  const int grid_size,
                                  const float* input )
{
   return tnlCUDASimpleReduction1< float, tnlMax >( size, block_size, grid_size, input );
}
                         
float tnlCUDASimpleReduction1Sum( const int size,
                                  const int block_size,
                                  const int grid_size,
                                  const float* input )
{
   return tnlCUDASimpleReduction1< float, tnlSum >( size, block_size, grid_size, input );
}

double tnlCUDASimpleReduction1Min( const int size,
                                   const int block_size,
                                   const int grid_size,
                                   const double* input )
{
   return tnlCUDASimpleReduction1< double, tnlMin >( size, block_size, grid_size, input );
}

double tnlCUDASimpleReduction1Max( const int size,
                                   const int block_size,
                                   const int grid_size,
                                   const double* input )
{
   return tnlCUDASimpleReduction1< double, tnlMax >( size, block_size, grid_size, input );
}
                         
double tnlCUDASimpleReduction1Sum( const int size,
                                   const int block_size,
                                   const int grid_size,
                                   const double* input )
{
   return tnlCUDASimpleReduction1< double, tnlSum >( size, block_size, grid_size, input );
}*/