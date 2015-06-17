/***************************************************************************
                          tnlCudaReduction.h  -  description
                             -------------------
    begin                : Jun 17, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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

#ifndef TNLCUDAREDUCTION_H
#define	TNLCUDAREDUCTION_H

#ifdef HAVE_CUDA

template< typename Operation, int blockSize, bool isSizePow2 >
class tnlCUDAReduction
{
   public:

      typedef typename Operation::IndexType IndexType;
      typedef typename Operation::RealType RealType;
      typedef typename Operation::ResultType ResultType;

      
      __device__ static void reduce( const Operation operation,
                                     const IndexType size,
                                     const RealType* input1,
                                     const RealType* input2,
                                     ResultType* output );
};
      
/*template< typename Real, typename Index, int blockSize, bool isSizePow2 >
class tnlCUDAReduction< tnlParallelReductionScalarProduct< Real, Index >, blockSize, isSizePow2 >
{
   public:
      
      typedef tnlParallelReductionScalarProduct< Real, Index > Operation;      
      typedef typename Operation::IndexType IndexType;
      typedef typename Operation::RealType RealType;
      typedef typename Operation::ResultType ResultType;
      
      __device__ static void reduce( const Operation operation,
                                     const IndexType size,
                                     const RealType* input1,
                                     const RealType* input2,
                                     ResultType* output );
};*/

#include <core/cuda/tnlCudaReduction_impl.h>

#endif

#endif	/* TNLCUDAREDUCTION_H */

