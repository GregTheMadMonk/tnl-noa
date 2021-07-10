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
#include "reduce.h"

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
         const typename Vector::ValueType zero )
{
   using ValueType = typename Vector::ValueType;
   using IndexType = typename Vector::IndexType;

   // simple sequential algorithm - not split into phases
   ValueType aux = zero;
   if( Type == ScanType::Inclusive ) {
      for( IndexType i = begin; i < end; i++ )
         v[ i ] = aux = reduction( aux, v[ i ] );
   }
   else // Exclusive scan
   {
      for( IndexType i = begin; i < end; i++ ) {
         const ValueType x = v[ i ];
         v[ i ] = aux;
         aux = reduction( aux, x );
      }
   }
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
                   const typename Vector::ValueType zero )
{
   // FIXME: StaticArray does not have getElement() which is used in DistributedScan
//   Containers::StaticArray< 2, ValueType > block_results;
   Containers::Array< typename Vector::ValueType, Devices::Sequential > block_results( 2 );
   // artificial first phase - only reduce the block
   block_results[ 0 ] = zero;
   block_results[ 1 ] = reduce< Devices::Sequential >( begin, end, v, reduction, zero );
   return block_results;
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
                    const typename Vector::ValueType zero )
{
   // artificial second phase - only one block, use the shift as the initial value
   perform( v, begin, end, reduction, reduction( zero, blockShifts[ 0 ] ) );
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
         const typename Vector::ValueType zero )
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
                   const typename Vector::ValueType zero )
{
#ifdef HAVE_OPENMP
   using ValueType = typename Vector::ValueType;
   using IndexType = typename Vector::IndexType;

   const int threads = Devices::Host::getMaxThreadsCount();
   Containers::Array< ValueType > block_results( threads + 1 );

   #pragma omp parallel num_threads(threads)
   {
      // init
      const int thread_idx = omp_get_thread_num();
      ValueType block_result = zero;

      // perform scan on blocks statically assigned to threads
      if( Type == ScanType::Inclusive ) {
         #pragma omp for schedule(static)
         for( IndexType i = begin; i < end; i++ ) {
            block_result = reduction( block_result, v[ i ] );
            v[ i ] = block_result;
         }
      }
      else {
         #pragma omp for schedule(static)
         for( IndexType i = begin; i < end; i++ ) {
            const ValueType x = v[ i ];
            v[ i ] = block_result;
            block_result = reduction( block_result, x );
         }
      }

      // write the block result into the buffer
      block_results[ thread_idx ] = block_result;
   }

   // block_results now contains scan results for each block. The first phase
   // ends by computing an exclusive scan of this array.
   Scan< Devices::Sequential, ScanType::Exclusive >::perform( block_results, 0, threads + 1, reduction, zero );

   // block_results now contains shift values for each block - to be used in the second phase
   return block_results;
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
                    const typename Vector::ValueType zero )
{
#ifdef HAVE_OPENMP
   using ValueType = typename Vector::ValueType;
   using IndexType = typename Vector::IndexType;

   const int threads = blockShifts.getSize() - 1;

   // launch exactly the same number of threads as in the first phase
   #pragma omp parallel num_threads(threads)
   {
      const int thread_idx = omp_get_thread_num();
      const ValueType offset = reduction( zero, blockShifts[ thread_idx ] );

      // shift intermediate results by the offset
      #pragma omp for schedule(static)
      for( IndexType i = begin; i < end; i++ )
         v[ i ] = reduction( v[ i ], offset );
   }
#else
   Scan< Devices::Sequential, Type >::performSecondPhase( v, blockShifts, begin, end, reduction, zero );
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
         const typename Vector::ValueType zero )
{
#ifdef HAVE_CUDA
   using ValueType = typename Vector::ValueType;
   using IndexType = typename Vector::IndexType;

   detail::CudaScanKernelLauncher< Type, ValueType, IndexType >::perform(
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
                   const typename Vector::ValueType zero )
{
#ifdef HAVE_CUDA
   using ValueType = typename Vector::ValueType;
   using IndexType = typename Vector::IndexType;

   return detail::CudaScanKernelLauncher< Type, ValueType, IndexType >::performFirstPhase(
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
                    const typename Vector::ValueType zero )
{
#ifdef HAVE_CUDA
   using ValueType = typename Vector::ValueType;
   using IndexType = typename Vector::IndexType;

   detail::CudaScanKernelLauncher< Type, ValueType, IndexType >::performSecondPhase(
      end - begin,
      &v.getData()[ begin ],  // output
      blockShifts.getData(),
      reduction,
      zero );
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
         const typename Vector::ValueType zero )
{
   using ValueType = typename Vector::ValueType;
   using IndexType = typename Vector::IndexType;

   if( Type == ScanType::Inclusive )
   {
      for( IndexType i = begin + 1; i < end; i++ )
         if( ! flags[ i ] )
            v[ i ] = reduction( v[ i ], v[ i - 1 ] );
   }
   else // Exclusive scan
   {
      ValueType aux( v[ begin ] );
      v[ begin ] = zero;
      for( IndexType i = begin + 1; i < end; i++ )
      {
         ValueType x = v[ i ];
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
         const typename Vector::ValueType zero )
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
         const typename Vector::ValueType zero )
{
#ifdef HAVE_CUDA
   using ValueType = typename Vector::ValueType;
   using IndexType = typename Vector::IndexType;

   throw Exceptions::NotImplementedError( "Segmented scan (prefix sum) is not implemented for CUDA." );
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

} // namespace Algorithms
} // namespace TNL
