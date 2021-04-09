/***************************************************************************
                          Scan.hpp  -  description
                             -------------------
    begin                : Mar 24, 2013
    copyright            : (C) 2013 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include "Scan.h"

#include <TNL/Assert.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Algorithms/detail/CudaScanKernel.h>
#include <TNL/Exceptions/CudaSupportMissing.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Algorithms {

template< ScanType Type >
   template< typename Vector,
             typename Reduction >
void
Scan< Devices::Sequential, Type >::
perform( Vector& v,
         const typename Vector::IndexType begin,
         const typename Vector::IndexType end,
         const Reduction& reduction,
         const typename Vector::RealType zero )
{
   // sequential prefix-sum does not need a second phase
   performFirstPhase( v, begin, end, reduction, zero );
}

template< ScanType Type >
   template< typename Vector,
             typename Reduction >
auto
Scan< Devices::Sequential, Type >::
performFirstPhase( Vector& v,
                   const typename Vector::IndexType begin,
                   const typename Vector::IndexType end,
                   const Reduction& reduction,
                   const typename Vector::RealType zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;

   // FIXME: StaticArray does not have getElement() which is used in DistributedScan
//   return Containers::StaticArray< 1, RealType > block_sums;
   Containers::Array< RealType, Devices::Host > block_sums( 1 );
   block_sums[ 0 ] = zero;

   if( Type == ScanType::Inclusive ) {
      for( IndexType i = begin + 1; i < end; i++ )
         v[ i ] = reduction( v[ i ], v[ i - 1 ] );
      block_sums[ 0 ] = v[ end - 1 ];
   }
   else // Exclusive prefix sum
   {
      RealType aux = zero;
      for( IndexType i = begin; i < end; i++ ) {
         const RealType x = v[ i ];
         v[ i ] = aux;
         aux = reduction( aux, x );
      }
      block_sums[ 0 ] = aux;
   }

   return block_sums;
}

template< ScanType Type >
   template< typename Vector,
             typename BlockShifts,
             typename Reduction >
void
Scan< Devices::Sequential, Type >::
performSecondPhase( Vector& v,
                    const BlockShifts& blockShifts,
                    const typename Vector::IndexType begin,
                    const typename Vector::IndexType end,
                    const Reduction& reduction,
                    const typename Vector::RealType shift )
{
   using IndexType = typename Vector::IndexType;

   for( IndexType i = begin; i < end; i++ )
      v[ i ] = reduction( v[ i ], shift );
}

template< ScanType Type >
   template< typename Vector,
             typename Reduction >
void
Scan< Devices::Host, Type >::
perform( Vector& v,
         const typename Vector::IndexType begin,
         const typename Vector::IndexType end,
         const Reduction& reduction,
         const typename Vector::RealType zero )
{
#ifdef HAVE_OPENMP
   if( Devices::Host::isOMPEnabled() && Devices::Host::getMaxThreadsCount() >= 2 ) {
      const auto blockShifts = performFirstPhase( v, begin, end, reduction, zero );
      performSecondPhase( v, blockShifts, begin, end, reduction, zero );
   }
   else
      Scan< Devices::Sequential, Type >::perform( v, begin, end, reduction, zero );
#else
   Scan< Devices::Sequential, Type >::perform( v, begin, end, reduction, zero );
#endif
}

template< ScanType Type >
   template< typename Vector,
             typename Reduction >
auto
Scan< Devices::Host, Type >::
performFirstPhase( Vector& v,
                   const typename Vector::IndexType begin,
                   const typename Vector::IndexType end,
                   const Reduction& reduction,
                   const typename Vector::RealType zero )
{
#ifdef HAVE_OPENMP
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;

   const int threads = Devices::Host::getMaxThreadsCount();
   Containers::Array< RealType > block_sums( threads + 1 );
   block_sums[ 0 ] = zero;

   #pragma omp parallel num_threads(threads)
   {
      // init
      const int thread_idx = omp_get_thread_num();
      RealType block_sum = zero;

      // perform prefix-sum on blocks statically assigned to threads
      if( Type == ScanType::Inclusive ) {
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
   }

   // block_sums now contains sums of numbers in each block. The first phase
   // ends by computing prefix-sum of this array.
   for( int i = 1; i < threads + 1; i++ )
      block_sums[ i ] = reduction( block_sums[ i ], block_sums[ i - 1 ] );

   // block_sums now contains shift values for each block - to be used in the second phase
   return block_sums;
#else
   return Scan< Devices::Sequential, Type >::performFirstPhase( v, begin, end, reduction, zero );
#endif
}

template< ScanType Type >
   template< typename Vector,
             typename BlockShifts,
             typename Reduction >
void
Scan< Devices::Host, Type >::
performSecondPhase( Vector& v,
                    const BlockShifts& blockShifts,
                    const typename Vector::IndexType begin,
                    const typename Vector::IndexType end,
                    const Reduction& reduction,
                    const typename Vector::RealType shift )
{
#ifdef HAVE_OPENMP
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;

   const int threads = blockShifts.getSize() - 1;

   // launch exactly the same number of threads as in the first phase
   #pragma omp parallel num_threads(threads)
   {
      const int thread_idx = omp_get_thread_num();
      const RealType offset = reduction( blockShifts[ thread_idx ], shift );

      // shift intermediate results by the offset
      #pragma omp for schedule(static)
      for( IndexType i = begin; i < end; i++ )
         v[ i ] = reduction( v[ i ], offset );
   }
#else
   Scan< Devices::Sequential, Type >::performSecondPhase( v, blockShifts, begin, end, reduction, shift );
#endif
}

template< ScanType Type >
   template< typename Vector,
             typename Reduction >
void
Scan< Devices::Cuda, Type >::
perform( Vector& v,
         const typename Vector::IndexType begin,
         const typename Vector::IndexType end,
         const Reduction& reduction,
         const typename Vector::RealType zero )
{
#ifdef HAVE_CUDA
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;

   detail::CudaScanKernelLauncher< Type, RealType, IndexType >::perform(
      end - begin,
      &v.getData()[ begin ],  // input
      &v.getData()[ begin ],  // output
      reduction,
      zero );
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< ScanType Type >
   template< typename Vector,
             typename Reduction >
auto
Scan< Devices::Cuda, Type >::
performFirstPhase( Vector& v,
                   const typename Vector::IndexType begin,
                   const typename Vector::IndexType end,
                   const Reduction& reduction,
                   const typename Vector::RealType zero )
{
#ifdef HAVE_CUDA
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;

   return detail::CudaScanKernelLauncher< Type, RealType, IndexType >::performFirstPhase(
      end - begin,
      &v.getData()[ begin ],  // input
      &v.getData()[ begin ],  // output
      reduction,
      zero );
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< ScanType Type >
   template< typename Vector,
             typename BlockShifts,
             typename Reduction >
void
Scan< Devices::Cuda, Type >::
performSecondPhase( Vector& v,
                    const BlockShifts& blockShifts,
                    const typename Vector::IndexType begin,
                    const typename Vector::IndexType end,
                    const Reduction& reduction,
                    const typename Vector::RealType shift )
{
#ifdef HAVE_CUDA
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;

   detail::CudaScanKernelLauncher< Type, RealType, IndexType >::performSecondPhase(
      end - begin,
      &v.getData()[ begin ],  // output
      blockShifts.getData(),
      reduction,
      shift );
#else
   throw Exceptions::CudaSupportMissing();
#endif
}


template< ScanType Type >
   template< typename Vector,
             typename Reduction,
             typename Flags >
void
SegmentedScan< Devices::Sequential, Type >::
perform( Vector& v,
         Flags& flags,
         const typename Vector::IndexType begin,
         const typename Vector::IndexType end,
         const Reduction& reduction,
         const typename Vector::RealType zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;

   if( Type == ScanType::Inclusive )
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

template< ScanType Type >
   template< typename Vector,
             typename Reduction,
             typename Flags >
void
SegmentedScan< Devices::Host, Type >::
perform( Vector& v,
         Flags& flags,
         const typename Vector::IndexType begin,
         const typename Vector::IndexType end,
         const Reduction& reduction,
         const typename Vector::RealType zero )
{
#ifdef HAVE_OPENMP
   // TODO: parallelize with OpenMP
   SegmentedScan< Devices::Sequential, Type >::perform( v, flags, begin, end, reduction, zero );
#else
   SegmentedScan< Devices::Sequential, Type >::perform( v, flags, begin, end, reduction, zero );
#endif
}

template< ScanType Type >
   template< typename Vector,
             typename Reduction,
             typename Flags >
void
SegmentedScan< Devices::Cuda, Type >::
perform( Vector& v,
         Flags& flags,
         const typename Vector::IndexType begin,
         const typename Vector::IndexType end,
         const Reduction& reduction,
         const typename Vector::RealType zero )
{
#ifdef HAVE_CUDA
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;

   throw Exceptions::NotImplementedError( "Segmented scan (prefix sum) is not implemented for CUDA." );
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

} // namespace Algorithms
} // namespace TNL
