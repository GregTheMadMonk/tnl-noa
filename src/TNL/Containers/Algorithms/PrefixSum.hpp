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

#include <memory>  // std::unique_ptr

#include "PrefixSum.h"

//#define CUDA_REDUCTION_PROFILING

#include <TNL/Assert.h>
#include <TNL/Exceptions/CudaSupportMissing.h>
#include <TNL/Containers/Algorithms/ArrayOperations.h>
#include <TNL/Containers/Algorithms/CudaPrefixSumKernel.h>
#include <TNL/Exceptions/NotImplementedError.h>

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
          typename Reduction >
void
PrefixSum< Devices::Host, Type >::
perform( Vector& v,
         const typename Vector::IndexType begin,
         const typename Vector::IndexType end,
         const Reduction& reduction,
         const typename Vector::RealType& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;

#ifdef HAVE_OPENMP
   const int threads = Devices::Host::getMaxThreadsCount();
   std::unique_ptr< RealType[] > block_sums{
      // Workaround for nvcc 10.1.168 - it would modifie the simple expression
      // `new RealType[reducedSize]` in the source code to `new (RealType[reducedSize])`
      // which is not correct - see e.g. https://stackoverflow.com/a/39671946
      // Thus, the host compiler would spit out hundreds of warnings...
      // Funnily enough, nvcc's behaviour depends on the context rather than the
      // expression, because exactly the same simple expression in different places
      // does not produce warnings.
      #ifdef __NVCC__
      new RealType[ static_cast<const int&>(threads) + 1 ]
      #else
      new RealType[ threads + 1 ]
      #endif
   };
   block_sums[ 0 ] = zero;

   #pragma omp parallel
   {
      // init
      const int thread_idx = omp_get_thread_num();
      RealType block_sum = zero;

      // perform prefix-sum on blocks statically assigned to threads
      if( Type == PrefixSumType::Inclusive ) {
         #pragma omp for schedule(static)
         for( IndexType i = begin; i < end; i++ ) {
            block_sum = reduction( block_sum, v[ i ] );
            v[ i ] = block_sum;
         }
      }
      else {
         #pragma omp for schedule(static)
         for( IndexType i = begin; i < end; i++ ) {
            const RealType x = v[ i ];
            v[ i ] = block_sum;
            block_sum = reduction( block_sum, x );
         }
      }

      // write the block sums into the buffer
      block_sums[ thread_idx + 1 ] = block_sum;
      #pragma omp barrier

      // calculate per-block offsets
      RealType offset = 0;
      for( int i = 0; i < thread_idx + 1; i++ )
         offset = reduction( offset, block_sums[ i ] );

      // shift intermediate results by the offset
      #pragma omp for schedule(static)
      for( IndexType i = begin; i < end; i++ )
         v[ i ] = reduction( v[ i ], offset );
   }
#else
   if( Type == PrefixSumType::Inclusive ) {
      for( IndexType i = begin + 1; i < end; i++ )
         v[ i ] = reduction( v[ i ], v[ i - 1 ] );
   }
   else // Exclusive prefix sum
   {
      RealType aux = zero;
      for( IndexType i = begin; i < end; i++ ) {
         const RealType x = v[ i ];
         v[ i ] = aux;
         aux = reduction( aux, x );
      }
   }
#endif
}

////
// PrefixSum on CUDA device
template< PrefixSumType Type >
template< typename Vector,
          typename Reduction >
void
PrefixSum< Devices::Cuda, Type >::
perform( Vector& v,
         const typename Vector::IndexType begin,
         const typename Vector::IndexType end,
         const Reduction& reduction,
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
      zero );
#endif
}


////
// PrefixSum on host
template< PrefixSumType Type >
   template< typename Vector,
             typename Reduction,
             typename Flags >
void
SegmentedPrefixSum< Devices::Host, Type >::
perform( Vector& v,
         Flags& flags,
         const typename Vector::IndexType begin,
         const typename Vector::IndexType end,
         const Reduction& reduction,
         const typename Vector::RealType& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;

   // TODO: parallelize with OpenMP
   if( Type == PrefixSumType::Inclusive )
   {
      for( IndexType i = begin + 1; i < end; i++ )
         if( ! flags[ i ] )
            v[ i ] = reduction( v[ i ], v[ i - 1 ] );
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
         aux = reduction( aux, x );
      }
   }
}

////
// PrefixSum on CUDA device
template< PrefixSumType Type >
   template< typename Vector,
             typename Reduction,
             typename Flags >
void
SegmentedPrefixSum< Devices::Cuda, Type >::
perform( Vector& v,
         Flags& flags,
         const typename Vector::IndexType begin,
         const typename Vector::IndexType end,
         const Reduction& reduction,
         const typename Vector::RealType& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using IndexType = typename Vector::IndexType;
#ifdef HAVE_CUDA
   throw Exceptions::NotImplementedError( "Segmented prefix sum is not implemented for CUDA." ); // NOT IMPLEMENTED YET
   /*CudaPrefixSumKernelLauncher< Type, RealType, IndexType >::start(
      ( IndexType ) ( end - begin ),
      ( IndexType ) 256,
      &v[ begin ],
      &v[ begin ],
      reduction,
      zero );*/
#endif
}

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
