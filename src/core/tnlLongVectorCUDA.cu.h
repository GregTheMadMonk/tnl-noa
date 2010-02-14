/***************************************************************************
                          tnlLongVectorCUDA.cu.h  -  description
                             -------------------
    begin                : Feb 11, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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

#ifndef TNLLONGVECTORCUDA_CU_H_
#define TNLLONGVECTORCUDA_CU_H_

template< typename REAL >
__global__ void tnlLongVectorCUDASetValueKernel( REAL* data,
                                                 const int size,
                                                 const REAL v )
{
   const int i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size ) data[ i ] = v;
}

template< typename REAL >
void tnlLongVectorCUDASetValueKernelCaller( REAL* data,
                                            const int size,
                                            const REAL v )
{
   const int block_size = 512;
   const int grid_size = size / 512 + 1;

   tnlLongVectorCUDASetValueKernel<<< grid_size, block_size >>>( data, size, v );
};

#endif /* TNLLONGVECTORCUDA_CU_H_ */
