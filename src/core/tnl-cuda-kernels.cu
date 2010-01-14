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

#include <tnl-cuda-kernels.h>

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


