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
#include <TNL/Containers/Algorithms/cuda-prefix-sum.h>

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
template< typename Vector,
          typename PrefixSumOperation,
          typename VolatilePrefixSumOperation >
void
PrefixSum< Devices::Host >::
inclusive( Vector& v,
           const typename Vector::IndexType begin,
           const typename Vector::IndexType end,
           PrefixSumOperation& reduction,
           VolatilePrefixSumOperation& volatilePrefixSum,
           const typename Vector::RealType& zero )
{
   using IndexType = typename Vector::IndexType;

   // TODO: parallelize with OpenMP
   for( IndexType i = begin + 1; i < end; i++ )
      reduction( v[ i ], v[ i - 1 ] );
}

template< typename Vector,
          typename PrefixSumOperation,
          typename VolatilePrefixSumOperation >
void
PrefixSum< Devices::Host >::
exclusive( Vector& v,
           const typename Vector::IndexType begin,
           const typename Vector::IndexType end,
           PrefixSumOperation& reduction,
           VolatilePrefixSumOperation& volatilePrefixSum,
           const typename Vector::RealType& zero  )
{
   using IndexType = typename Vector::IndexType;
   using RealType = typename Vector::RealType;

   // TODO: parallelize with OpenMP
   RealType aux( v[ begin ] );
   v[ begin ] = zero;
   for( IndexType i = begin + 1; i < end; i++ )
   {
      RealType x = v[ i ];
      v[ i ] = aux;
      reduction( aux, x );
   }
}

template< typename Vector,
          typename FlagsArray,
          typename PrefixSumOperation,
          typename VolatilePrefixSumOperation >
void
PrefixSum< Devices::Host >::
inclusiveSegmented( Vector& v,
                    FlagsArray& f,
                    const typename Vector::IndexType begin,
                    const typename Vector::IndexType end,
                    PrefixSumOperation& reduction,
                    VolatilePrefixSumOperation& volatilePrefixSum,
                    const typename Vector::RealType& zero )
{
   using IndexType = typename Vector::IndexType;

   // TODO: parallelize with OpenMP
   for( IndexType i = begin + 1; i < end; i++ )
      if( ! f[ i ] )
         reduction( v[ i ], v[ i - 1 ] );
}

template< typename Vector,
          typename FlagsArray,
          typename PrefixSumOperation,
          typename VolatilePrefixSumOperation >
void
PrefixSum< Devices::Host >::
exclusiveSegmented( Vector& v,
                    FlagsArray& f,
                    const typename Vector::IndexType begin,
                    const typename Vector::IndexType end,
                    PrefixSumOperation& reduction,
                    VolatilePrefixSumOperation& volatilePrefixSum,
                    const typename Vector::RealType& zero )
{
   using IndexType = typename Vector::IndexType;
   using RealType = typename Vector::RealType;

   // TODO: parallelize with OpenMP
   RealType aux( v[ begin ] );
   v[ begin ] = zero;
   for( IndexType i = begin + 1; i < end; i++ )
   {
      RealType x = v[ i ];
      if( f[ i ] )
         aux = zero;
      v[ i ] = aux;
      reduction( aux, x );
   }
}

////
// PrefixSum on CUDA device
template< typename Vector,
          typename PrefixSumOperation,
          typename VolatilePrefixSumOperation >
void
PrefixSum< Devices::Cuda >::
inclusive( Vector& v,
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
   cudaPrefixSum( ( IndexType ) ( end - begin ),
                  ( IndexType ) 256,
                  &v[ begin ],
                  &v[ begin ],
                  reduction,
                  volatileReduction,
                  zero,
                  Algorithms::PrefixSumType::inclusive );
#endif
}

template< typename Vector,
          typename PrefixSumOperation,
          typename VolatilePrefixSumOperation >
void
PrefixSum< Devices::Cuda >::
exclusive( Vector& v,
           const typename Vector::IndexType begin,
           const typename Vector::IndexType end,
           PrefixSumOperation& reduction,
           VolatilePrefixSumOperation& volatileReduction,
           const typename Vector::RealType& zero  )
{
   using IndexType = typename Vector::IndexType;
   using RealType = typename Vector::RealType;
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using IndexType = typename Vector::IndexType;
#ifdef HAVE_CUDA
   cudaPrefixSum( ( IndexType ) ( end - begin ),
                  ( IndexType ) 256,
                  &v[ begin ],
                  &v[ begin ],
                  reduction,
                  volatileReduction,
                  zero,
                  Algorithms::PrefixSumType::exclusive );
#endif
}

template< typename Vector,
          typename FlagsArray,
          typename PrefixSumOperation,
          typename VolatilePrefixSumOperation >
void
PrefixSum< Devices::Cuda >::
inclusiveSegmented( Vector& v,
                    FlagsArray& f,
                    const typename Vector::IndexType begin,
                    const typename Vector::IndexType end,
                    PrefixSumOperation& reduction,
                    VolatilePrefixSumOperation& volatilePrefixSum,
                    const typename Vector::RealType& zero )
{
   using IndexType = typename Vector::IndexType;

}

template< typename Vector,
          typename FlagsArray,
          typename PrefixSumOperation,
          typename VolatilePrefixSumOperation >
void
PrefixSum< Devices::Cuda >::
exclusiveSegmented( Vector& v,
                    FlagsArray& f,
                    const typename Vector::IndexType begin,
                    const typename Vector::IndexType end,
                    PrefixSumOperation& reduction,
                    VolatilePrefixSumOperation& volatilePrefixSum,
                    const typename Vector::RealType& zero )
{
   using IndexType = typename Vector::IndexType;
   using RealType = typename Vector::RealType;

}

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
