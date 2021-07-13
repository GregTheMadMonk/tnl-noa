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
#include "CudaScanKernel.h"

#include <TNL/Assert.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Exceptions/CudaSupportMissing.h>

namespace TNL {
namespace Algorithms {
namespace detail {

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
   using ValueType = typename Vector::ValueType;
   using IndexType = typename Vector::IndexType;

   if( end <= begin )
      return;

   const IndexType size = end - begin;
   const int max_threads = Devices::Host::getMaxThreadsCount();
   const IndexType block_size = TNL::max( 1024, TNL::roundUpDivision( size, max_threads ) );
   const IndexType blocks = TNL::roundUpDivision( size, block_size );

   if( Devices::Host::isOMPEnabled() && blocks >= 2 ) {
      const int threads = TNL::min( blocks, Devices::Host::getMaxThreadsCount() );
      Containers::Array< ValueType > block_results( blocks + 1 );

      #pragma omp parallel num_threads(threads)
      {
         const IndexType block_idx = omp_get_thread_num();
         const IndexType block_begin = begin + block_idx * block_size;
         const IndexType block_end = TNL::min( block_begin + block_size, end );

         // step 1: per-block reductions, write the result into the buffer
         block_results[ block_idx ] = reduce< Devices::Sequential >( block_begin, block_end, v, reduction, zero );

         #pragma omp barrier

         // step 2: scan the block results
         #pragma omp single
         {
            Scan< Devices::Sequential, ScanType::Exclusive >::perform( block_results, 0, blocks + 1, reduction, zero );
         }

         // step 3: per-block scan using the block results as initial values
         Scan< Devices::Sequential, Type >::perform( v, block_begin, block_end, reduction, block_results[ block_idx ] );
      }
   }
   else
#endif
      Scan< Devices::Sequential, Type >::perform( v, begin, end, reduction, zero );
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

   if( end <= begin ) {
      Containers::Array< typename Vector::ValueType, Devices::Sequential > block_results( 1 );
      block_results.setValue( zero );
      return block_results;
   }

   const IndexType size = end - begin;
   const int max_threads = Devices::Host::getMaxThreadsCount();
   const IndexType block_size = TNL::max( 1024, TNL::roundUpDivision( size, max_threads ) );
   const IndexType blocks = TNL::roundUpDivision( size, block_size );

   if( Devices::Host::isOMPEnabled() && blocks >= 2 ) {
      const int threads = TNL::min( blocks, Devices::Host::getMaxThreadsCount() );
      Containers::Array< ValueType, Devices::Sequential > block_results( blocks + 1 );

      #pragma omp parallel num_threads(threads)
      {
         const IndexType block_idx = omp_get_thread_num();
         const IndexType block_begin = begin + block_idx * block_size;
         const IndexType block_end = TNL::min( block_begin + block_size, end );

         // step 1: per-block reductions, write the result into the buffer
         block_results[ block_idx ] = reduce< Devices::Sequential >( block_begin, block_end, v, reduction, zero );
      }

      // step 2: scan the block results
      Scan< Devices::Sequential, ScanType::Exclusive >::perform( block_results, 0, blocks + 1, reduction, zero );

      // block_results now contains shift values for each block - to be used in the second phase
      return block_results;
   }
   else
#endif
      return Scan< Devices::Sequential, Type >::performFirstPhase( v, begin, end, reduction, zero );
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

   if( end <= begin )
      return;

   const IndexType size = end - begin;
   const int max_threads = Devices::Host::getMaxThreadsCount();
   const IndexType block_size = TNL::max( 1024, TNL::roundUpDivision( size, max_threads ) );
   const IndexType blocks = TNL::roundUpDivision( size, block_size );

   if( Devices::Host::isOMPEnabled() && blocks >= 2 ) {
      const int threads = TNL::min( blocks, Devices::Host::getMaxThreadsCount() );
      #pragma omp parallel num_threads(threads)
      {
         const IndexType block_idx = omp_get_thread_num();
         const IndexType block_begin = begin + block_idx * block_size;
         const IndexType block_end = TNL::min( block_begin + block_size, end );

         // phase 2: per-block scan using the block results as initial values
         Scan< Devices::Sequential, Type >::perform( v, block_begin, block_end, reduction, reduction( zero, blockShifts[ block_idx ] ) );
      }
   }
   else
#endif
      Scan< Devices::Sequential, Type >::performSecondPhase( v, blockShifts, begin, end, reduction, zero );
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

   if( end <= begin )
      return;

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

   if( end <= begin ) {
      Containers::Array< typename Vector::ValueType, Devices::Cuda > block_results( 1 );
      block_results.setValue( zero );
      return block_results;
   }

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

   if( end <= begin )
      return;

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

} // namespace detail
} // namespace Algorithms
} // namespace TNL
