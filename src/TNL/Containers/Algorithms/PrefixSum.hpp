/***************************************************************************
                          PrefixSum.hpp  -  description
                             -------------------
    begin                : Mar 24, 2013
    copyright            : (C) 2013 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include "PrefixSum.h"

//#define CUDA_REDUCTION_PROFILING

#include <TNL/Assert.h>
#include <TNL/Exceptions/CudaSupportMissing.h>
#include <TNL/Containers/Algorithms/ReductionOperations.h>
#include <TNL/Containers/Algorithms/ArrayOperations.h>
#include <TNL/Containers/Algorithms/CudaPrefixSumKernel.h>

#ifdef CUDA_REDUCTION_PROFILING
#include <iostream>
#include <TNL/Timer.h>
#endif

namespace TNL {
namespace Containers {
namespace Algorithms {

/****
 * Arrays smaller than the following constant
 * are reduced on CPU. The constant must not be larger
 * than maximal CUDA grid size.
 */
static constexpr int PrefixSum_minGpuDataSize = 256;//65536; //16384;//1024;//256;

////
// PrefixSum on host
template< PrefixSumType Type >
template< typename Vector,
          typename PrefixSumOperation,
          typename VolatilePrefixSumOperation >
void
PrefixSum< Devices::Host, Type >::
perform( Vector& v,
         const typename Vector::IndexType begin,
         const typename Vector::IndexType end,
         PrefixSumOperation& reduction,
         VolatilePrefixSumOperation& volatilePrefixSum,
         const typename Vector::RealType& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;

   // TODO: parallelize with OpenMP
   if( Type == PrefixSumType::Inclusive )
      for( IndexType i = begin + 1; i < end; i++ )
         reduction( v[ i ], v[ i - 1 ] );
   else // Exclusive prefix sum
   {
      RealType aux( v[ begin ] );
      v[ begin ] = zero;
      for( IndexType i = begin + 1; i < end; i++ )
      {
         RealType x = v[ i ];
         v[ i ] = aux;
         reduction( aux, x );
      }
   }
}

////
// PrefixSum on CUDA device
template< PrefixSumType Type >
template< typename Vector,
          typename PrefixSumOperation,
          typename VolatilePrefixSumOperation >
void
PrefixSum< Devices::Cuda, Type >::
perform( Vector& v,
         const typename Vector::IndexType begin,
         const typename Vector::IndexType end,
         PrefixSumOperation& reduction,
         VolatilePrefixSumOperation& volatileReduction,
         const typename Vector::RealType& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using IndexType = typename Vector::IndexType;
#ifdef HAVE_CUDA
   CudaPrefixSumKernelLauncher< Type, RealType, IndexType >::start(
      ( IndexType ) ( end - begin ),
      ( IndexType ) 256,
      &v[ begin ],
      &v[ begin ],
      reduction,
      volatileReduction,
      zero );
#endif
}


////
// PrefixSum on host
template< PrefixSumType Type >
   template< typename Vector,
             typename PrefixSumOperation,
             typename VolatilePrefixSumOperation,
             typename Flags >
void
SegmentedPrefixSum< Devices::Host, Type >::
perform( Vector& v,
         Flags& flags,
         const typename Vector::IndexType begin,
         const typename Vector::IndexType end,
         PrefixSumOperation& reduction,
         VolatilePrefixSumOperation& volatilePrefixSum,
         const typename Vector::RealType& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;

   // TODO: parallelize with OpenMP
   if( Type == PrefixSumType::Inclusive )
   {
      for( IndexType i = begin + 1; i < end; i++ )
         if( ! flags[ i ] )
            reduction( v[ i ], v[ i - 1 ] );
   }
   else // Exclusive prefix sum
   {
       RealType aux( v[ begin ] );
      v[ begin ] = zero;
      for( IndexType i = begin + 1; i < end; i++ )
      {
         RealType x = v[ i ];
         if( flags[ i ] )
            aux = zero;
         v[ i ] = aux;
         reduction( aux, x );
      }
   }
}

////
// PrefixSum on CUDA device
template< PrefixSumType Type >
   template< typename Vector,
             typename PrefixSumOperation,
             typename VolatilePrefixSumOperation,
             typename Flags >
void
SegmentedPrefixSum< Devices::Cuda, Type >::
perform( Vector& v,
         Flags& flags,
         const typename Vector::IndexType begin,
         const typename Vector::IndexType end,
         PrefixSumOperation& reduction,
         VolatilePrefixSumOperation& volatileReduction,
         const typename Vector::RealType& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using IndexType = typename Vector::IndexType;
#ifdef HAVE_CUDA
   throw 0; // NOT IMPLEMENTED YET
   /*CudaPrefixSumKernelLauncher< Type, RealType, IndexType >::start(
      ( IndexType ) ( end - begin ),
      ( IndexType ) 256,
      &v[ begin ],
      &v[ begin ],
      reduction,
      volatileReduction,
      zero );*/
#endif
}



} // namespace Algorithms
} // namespace Containers
} // namespace TNL
