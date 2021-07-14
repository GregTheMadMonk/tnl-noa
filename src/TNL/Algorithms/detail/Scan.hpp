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
   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
void
Scan< Devices::Sequential, Type >::
perform( const InputArray& input,
         OutputArray& output,
         typename InputArray::IndexType begin,
         typename InputArray::IndexType end,
         typename OutputArray::IndexType outputBegin,
         Reduction&& reduction,
         typename OutputArray::ValueType zero )
{
   using ValueType = typename OutputArray::ValueType;

   // simple sequential algorithm - not split into phases
   ValueType aux = zero;
   if( Type == ScanType::Inclusive ) {
      for( ; begin < end; begin++, outputBegin++ )
         output[ outputBegin ] = aux = reduction( aux, input[ begin ] );
   }
   else // Exclusive scan
   {
      for( ; begin < end; begin++, outputBegin++ ) {
         const ValueType x = input[ begin ];
         output[ outputBegin ] = aux;
         aux = reduction( aux, x );
      }
   }
}

template< ScanType Type >
   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
auto
Scan< Devices::Sequential, Type >::
performFirstPhase( const InputArray& input,
                   OutputArray& output,
                   typename InputArray::IndexType begin,
                   typename InputArray::IndexType end,
                   typename OutputArray::IndexType outputBegin,
                   Reduction&& reduction,
                   typename OutputArray::ValueType zero )
{
   // artificial first phase - only reduce the block
   Containers::Array< typename OutputArray::ValueType, Devices::Sequential > block_results( 2 );
   block_results[ 0 ] = zero;
   block_results[ 1 ] = reduce< Devices::Sequential >( begin, end, input, reduction, zero );
   return block_results;
}

template< ScanType Type >
   template< typename InputArray,
             typename OutputArray,
             typename BlockShifts,
             typename Reduction >
void
Scan< Devices::Sequential, Type >::
performSecondPhase( const InputArray& input,
                    OutputArray& output,
                    const BlockShifts& blockShifts,
                    typename InputArray::IndexType begin,
                    typename InputArray::IndexType end,
                    typename OutputArray::IndexType outputBegin,
                    Reduction&& reduction,
                    typename OutputArray::ValueType zero )
{
   // artificial second phase - only one block, use the shift as the initial value
   perform( input, output, begin, end, outputBegin, reduction, reduction( zero, blockShifts[ 0 ] ) );
}

template< ScanType Type >
   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
void
Scan< Devices::Host, Type >::
perform( const InputArray& input,
         OutputArray& output,
         typename InputArray::IndexType begin,
         typename InputArray::IndexType end,
         typename OutputArray::IndexType outputBegin,
         Reduction&& reduction,
         typename OutputArray::ValueType zero )
{
#ifdef HAVE_OPENMP
   using ValueType = typename OutputArray::ValueType;
   using IndexType = typename InputArray::IndexType;

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
         const int block_idx = omp_get_thread_num();
         const IndexType block_offset = block_idx * block_size;
         const IndexType block_begin = begin + block_offset;
         const IndexType block_end = TNL::min( block_begin + block_size, end );
         const IndexType block_output_begin = outputBegin + block_offset;

         // step 1: per-block reductions, write the result into the buffer
         block_results[ block_idx ] = reduce< Devices::Sequential >( block_begin, block_end, input, reduction, zero );

         #pragma omp barrier

         // step 2: scan the block results
         #pragma omp single
         {
            Scan< Devices::Sequential, ScanType::Exclusive >::perform( block_results, block_results, 0, blocks + 1, 0, reduction, zero );
         }

         // step 3: per-block scan using the block results as initial values
         Scan< Devices::Sequential, Type >::perform( input, output, block_begin, block_end, block_output_begin, reduction, block_results[ block_idx ] );
      }
   }
   else
#endif
      Scan< Devices::Sequential, Type >::perform( input, output, begin, end, outputBegin, reduction, zero );
}

template< ScanType Type >
   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
auto
Scan< Devices::Host, Type >::
performFirstPhase( const InputArray& input,
                   OutputArray& output,
                   typename InputArray::IndexType begin,
                   typename InputArray::IndexType end,
                   typename OutputArray::IndexType outputBegin,
                   Reduction&& reduction,
                   typename OutputArray::ValueType zero )
{
#ifdef HAVE_OPENMP
   using ValueType = typename OutputArray::ValueType;
   using IndexType = typename InputArray::IndexType;

   if( end <= begin ) {
      Containers::Array< ValueType, Devices::Sequential > block_results( 1 );
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
         const int block_idx = omp_get_thread_num();
         const IndexType block_begin = begin + block_idx * block_size;
         const IndexType block_end = TNL::min( block_begin + block_size, end );

         // step 1: per-block reductions, write the result into the buffer
         block_results[ block_idx ] = reduce< Devices::Sequential >( block_begin, block_end, input, reduction, zero );
      }

      // step 2: scan the block results
      Scan< Devices::Sequential, ScanType::Exclusive >::perform( block_results, block_results, 0, blocks + 1, 0, reduction, zero );

      // block_results now contains shift values for each block - to be used in the second phase
      return block_results;
   }
   else
#endif
      return Scan< Devices::Sequential, Type >::performFirstPhase( input, output, begin, end, outputBegin, reduction, zero );
}

template< ScanType Type >
   template< typename InputArray,
             typename OutputArray,
             typename BlockShifts,
             typename Reduction >
void
Scan< Devices::Host, Type >::
performSecondPhase( const InputArray& input,
                    OutputArray& output,
                    const BlockShifts& blockShifts,
                    typename InputArray::IndexType begin,
                    typename InputArray::IndexType end,
                    typename OutputArray::IndexType outputBegin,
                    Reduction&& reduction,
                    typename OutputArray::ValueType zero )
{
#ifdef HAVE_OPENMP
   using IndexType = typename InputArray::IndexType;

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
         const int block_idx = omp_get_thread_num();
         const IndexType block_offset = block_idx * block_size;
         const IndexType block_begin = begin + block_offset;
         const IndexType block_end = TNL::min( block_begin + block_size, end );
         const IndexType block_output_begin = outputBegin + block_offset;

         // phase 2: per-block scan using the block results as initial values
         Scan< Devices::Sequential, Type >::perform( input, output, block_begin, block_end, block_output_begin, reduction, reduction( zero, blockShifts[ block_idx ] ) );
      }
   }
   else
#endif
      Scan< Devices::Sequential, Type >::performSecondPhase( input, output, blockShifts, begin, end, outputBegin, reduction, zero );
}

template< ScanType Type >
   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
void
Scan< Devices::Cuda, Type >::
perform( const InputArray& input,
         OutputArray& output,
         typename InputArray::IndexType begin,
         typename InputArray::IndexType end,
         typename OutputArray::IndexType outputBegin,
         Reduction&& reduction,
         typename OutputArray::ValueType zero )
{
#ifdef HAVE_CUDA
   if( end <= begin )
      return;

   detail::CudaScanKernelLauncher< Type >::perform(
      end - begin,
      &input.getData()[ begin ],
      &output.getData()[ outputBegin ],
      reduction,
      zero );
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< ScanType Type >
   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
auto
Scan< Devices::Cuda, Type >::
performFirstPhase( const InputArray& input,
                   OutputArray& output,
                   typename InputArray::IndexType begin,
                   typename InputArray::IndexType end,
                   typename OutputArray::IndexType outputBegin,
                   Reduction&& reduction,
                   typename OutputArray::ValueType zero )
{
#ifdef HAVE_CUDA
   if( end <= begin ) {
      Containers::Array< typename OutputArray::ValueType, Devices::Cuda > block_results( 1 );
      block_results.setValue( zero );
      return block_results;
   }

   return detail::CudaScanKernelLauncher< Type >::performFirstPhase(
      end - begin,
      &input.getData()[ begin ],
      &output.getData()[ outputBegin ],
      reduction,
      zero );
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< ScanType Type >
   template< typename InputArray,
             typename OutputArray,
             typename BlockShifts,
             typename Reduction >
void
Scan< Devices::Cuda, Type >::
performSecondPhase( const InputArray& input,
                    OutputArray& output,
                    const BlockShifts& blockShifts,
                    typename InputArray::IndexType begin,
                    typename InputArray::IndexType end,
                    typename OutputArray::IndexType outputBegin,
                    Reduction&& reduction,
                    typename OutputArray::ValueType zero )
{
#ifdef HAVE_CUDA
   if( end <= begin )
      return;

   detail::CudaScanKernelLauncher< Type >::performSecondPhase(
      end - begin,
      &output.getData()[ outputBegin ],
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
