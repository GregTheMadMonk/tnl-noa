/***************************************************************************
                          CudaReduction.h  -  description
                             -------------------
    begin                : Jun 17, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Containers {
namespace Algorithms {
   
#ifdef HAVE_CUDA

template< typename Operation, int blockSize >
class CudaReduction
{
   public:

      typedef typename Operation::IndexType IndexType;
      typedef typename Operation::RealType RealType;
      typedef typename Operation::ResultType ResultType;

 
      __device__ static void reduce( Operation& operation,
                                     const IndexType size,
                                     const RealType* input1,
                                     const RealType* input2,
                                     ResultType* output );
};
 
/*template< typename Real, typename Index, int blockSize >
class CudaReduction< tnlParallelReductionScalarProduct< Real, Index >, blockSize >
{
   public:
 
      typedef tnlParallelReductionScalarProduct< Real, Index > Operation;
      typedef typename Operation::IndexType IndexType;
      typedef typename Operation::RealType RealType;
      typedef typename Operation::ResultType ResultType;
 
      __device__ static void reduce( Operation operation,
                                     const IndexType size,
                                     const RealType* input1,
                                     const RealType* input2,
                                     ResultType* output );
};*/

#endif

} // namespace Algorithms
} // namespace Containers
} // namespace TNL

#ifdef HAVE_CUDA
#include <TNL/Containers/Algorithms/CudaReduction_impl.h>
#endif

