/***************************************************************************
                          Reduction.hpp  -  description
                             -------------------
    begin                : Mar 24, 2013
    copyright            : (C) 2013 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include <memory>  // std::unique_ptr

//#define CUDA_REDUCTION_PROFILING

#include <TNL/Containers/Algorithms/Reduction.h>
#include <TNL/Containers/Algorithms/ArrayOperations.h>
#include <TNL/Containers/Algorithms/CudaReductionKernel.h>

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
static constexpr int Reduction_minGpuDataSize = 256;//65536; //16384;//1024;//256;

////
// Reduction on host
template< typename Index,
          typename Result,
          typename ReductionOperation,
          typename VolatileReductionOperation,
          typename DataFetcher >
Result
Reduction< Devices::Host >::
reduce( const Index size,
        ReductionOperation& reduction,
        VolatileReductionOperation& volatileReduction,
        DataFetcher& dataFetcher,
        const Result& zero )
{
   constexpr int block_size = 128;
   const int blocks = size / block_size;

#ifdef HAVE_OPENMP
   if( TNL::Devices::Host::isOMPEnabled() && size >= 2 * block_size ) {
      // global result variable
      Result result = zero;
#pragma omp parallel
      {
         // initialize array for thread-local results
         Result r[ 4 ] = { zero, zero, zero, zero  };

         #pragma omp for nowait
         for( int b = 0; b < blocks; b++ ) {
            const Index offset = b * block_size;
            for( int i = 0; i < block_size; i += 4 ) {
               reduction( r[ 0 ], dataFetcher( offset + i ) );
               reduction( r[ 1 ], dataFetcher( offset + i + 1 ) );
               reduction( r[ 2 ], dataFetcher( offset + i + 2 ) );
               reduction( r[ 3 ], dataFetcher( offset + i + 3 ) );
            }
         }

         // the first thread that reaches here processes the last, incomplete block
         #pragma omp single nowait
         {
            for( Index i = blocks * block_size; i < size; i++ )
               reduction( r[ 0 ], dataFetcher( i ) );
         }

         // local reduction of unrolled results
         reduction( r[ 0 ], r[ 2 ] );
         reduction( r[ 1 ], r[ 3 ] );
         reduction( r[ 0 ], r[ 1 ] );

         // inter-thread reduction of local results
         #pragma omp critical
         {
            reduction( result, r[ 0 ] );
         }
      }
      return result;
   }
   else {
#endif
      if( blocks > 1 ) {
         // initialize array for unrolled results
         Result r[ 4 ] = { zero, zero, zero, zero };

         // main reduction (explicitly unrolled loop)
         for( int b = 0; b < blocks; b++ ) {
            const Index offset = b * block_size;
            for( int i = 0; i < block_size; i += 4 ) {
               reduction( r[ 0 ], dataFetcher( offset + i ) );
               reduction( r[ 1 ], dataFetcher( offset + i + 1 ) );
               reduction( r[ 2 ], dataFetcher( offset + i + 2 ) );
               reduction( r[ 3 ], dataFetcher( offset + i + 3 ) );
            }
         }

         // reduction of the last, incomplete block (not unrolled)
         for( Index i = blocks * block_size; i < size; i++ )
            reduction( r[ 0 ], dataFetcher( i ) );
            //operation.dataFetcher( r[ 0 ], i, input1, input2 );

         // reduction of unrolled results
         reduction( r[ 0 ], r[ 2 ] );
         reduction( r[ 1 ], r[ 3 ] );
         reduction( r[ 0 ], r[ 1 ] );
         return r[ 0 ];
      }
      else {
         Result result = zero;
         for( Index i = 0; i < size; i++ )
            reduction( result, dataFetcher( i ) );
         return result;
      }
#ifdef HAVE_OPENMP
   }
#endif
}

template< typename Index,
          typename Result,
          typename ReductionOperation,
          typename VolatileReductionOperation,
          typename DataFetcher >
std::pair< Index, Result >
Reduction< Devices::Host >::
reduceWithArgument( const Index size,
                    ReductionOperation& reduction,
                    VolatileReductionOperation& volatileReduction,
                    DataFetcher& dataFetcher,
                    const Result& zero )
{
   constexpr int block_size = 128;
   const int blocks = size / block_size;

#ifdef HAVE_OPENMP
   if( TNL::Devices::Host::isOMPEnabled() && size >= 2 * block_size ) {
      // global result variable
      std::pair< Index, Result > result( -1, zero );
#pragma omp parallel
      {
         // initialize array for thread-local results
         Index arg[ 4 ] = { 0, 0, 0, 0 };
         Result r[ 4 ] = { zero, zero, zero, zero  };
         bool initialized( false );

         #pragma omp for nowait
         for( int b = 0; b < blocks; b++ ) {
            const Index offset = b * block_size;
            for( int i = 0; i < block_size; i += 4 ) {
               if( ! initialized ) {
                  arg[ 0 ] = offset + i;
                  arg[ 1 ] = offset + i + 1;
                  arg[ 2 ] = offset + i + 2;
                  arg[ 3 ] = offset + i + 3;
                  r[ 0 ] = dataFetcher( offset + i );
                  r[ 1 ] = dataFetcher( offset + i + 1 );
                  r[ 2 ] = dataFetcher( offset + i + 2 );
                  r[ 3 ] = dataFetcher( offset + i + 3 );
                  initialized = true;
                  continue;
               }
               reduction( arg[ 0 ], offset + i,     r[ 0 ], dataFetcher( offset + i ) );
               reduction( arg[ 1 ], offset + i + 1, r[ 1 ], dataFetcher( offset + i + 1 ) );
               reduction( arg[ 2 ], offset + i + 2, r[ 2 ], dataFetcher( offset + i + 2 ) );
               reduction( arg[ 3 ], offset + i + 3, r[ 3 ], dataFetcher( offset + i + 3 ) );
            }
         }

         // the first thread that reaches here processes the last, incomplete block
         #pragma omp single nowait
         {
            for( Index i = blocks * block_size; i < size; i++ )
               reduction( arg[ 0 ], i, r[ 0 ], dataFetcher( i ) );
         }

         // local reduction of unrolled results
         reduction( arg[ 0 ], arg[ 2 ], r[ 0 ], r[ 2 ] );
         reduction( arg[ 1 ], arg[ 3 ], r[ 1 ], r[ 3 ] );
         reduction( arg[ 0 ], arg[ 1 ], r[ 0 ], r[ 1 ] );

         // inter-thread reduction of local results
         #pragma omp critical
         {
            if( result.first == -1 )
               result.first = arg[ 0 ];
            reduction( result.first, arg[ 0 ], result.second, r[ 0 ] );
         }
      }
      return result;
   }
   else {
#endif
      if( blocks > 1 ) {
         // initialize array for unrolled results
         Index arg[ 4 ] = { 0, 0, 0, 0 };
         Result r[ 4 ] = { zero, zero, zero, zero };
         bool initialized( false );

         // main reduction (explicitly unrolled loop)
         for( int b = 0; b < blocks; b++ ) {
            const Index offset = b * block_size;
            for( int i = 0; i < block_size; i += 4 ) {
               if( ! initialized )
               {
                  arg[ 0 ] = offset + i;
                  arg[ 1 ] = offset + i + 1;
                  arg[ 2 ] = offset + i + 2;
                  arg[ 3 ] = offset + i + 3;
                  r[ 0 ] = dataFetcher( offset + i );
                  r[ 1 ] = dataFetcher( offset + i + 1 );
                  r[ 2 ] = dataFetcher( offset + i + 2 );
                  r[ 3 ] = dataFetcher( offset + i + 3 );
                  initialized = true;
                  continue;
               }
               reduction( arg[ 0 ], offset + i,     r[ 0 ], dataFetcher( offset + i ) );
               reduction( arg[ 1 ], offset + i + 1, r[ 1 ], dataFetcher( offset + i + 1 ) );
               reduction( arg[ 2 ], offset + i + 2, r[ 2 ], dataFetcher( offset + i + 2 ) );
               reduction( arg[ 3 ], offset + i + 3, r[ 3 ], dataFetcher( offset + i + 3 ) );
            }
         }

         // reduction of the last, incomplete block (not unrolled)
         for( Index i = blocks * block_size; i < size; i++ )
            reduction( arg[ 0 ], i, r[ 0 ], dataFetcher( i ) );

         // reduction of unrolled results
         reduction( arg[ 0 ], arg[ 2 ], r[ 0 ], r[ 2 ] );
         reduction( arg[ 1 ], arg[ 3 ], r[ 1 ], r[ 3 ] );
         reduction( arg[ 0 ], arg[ 1 ], r[ 0 ], r[ 1 ] );
         return std::make_pair( arg[ 0 ], r[ 0 ] );
      }
      else {
         std::pair< Index, Result > result( 0, dataFetcher( 0 ) );
         for( Index i = 1; i < size; i++ )
            reduction( result.first, i, result.second, dataFetcher( i ) );
         return result;
      }
#ifdef HAVE_OPENMP
   }
#endif
}

template< typename Index,
          typename Result,
          typename ReductionOperation,
          typename VolatileReductionOperation,
          typename DataFetcher >
Result
Reduction< Devices::Cuda >::
reduce( const Index size,
        ReductionOperation& reduction,
        VolatileReductionOperation& volatileReduction,
        DataFetcher& dataFetcher,
        const Result& zero )
{
   // Only fundamental and pointer types can be safely reduced on host. Complex
   // objects stored on the device might contain pointers into the device memory,
   // in which case reduction on host might fail.
   constexpr bool can_reduce_later_on_host = std::is_fundamental< Result >::value || std::is_pointer< Result >::value;

   #ifdef CUDA_REDUCTION_PROFILING
      Timer timer;
      timer.reset();
      timer.start();
   #endif

   CudaReductionKernelLauncher< Index, Result > reductionLauncher( size );

   // start the reduction on the GPU
   Result* deviceAux1( 0 );
   Index reducedSize = reductionLauncher.start(
      reduction,
      dataFetcher,
      zero,
      deviceAux1 );

   #ifdef CUDA_REDUCTION_PROFILING
      timer.stop();
      std::cout << "   Reduction on GPU to size " << reducedSize << " took " << timer.getRealTime() << " sec. " << std::endl;
      timer.reset();
      timer.start();
   #endif

   if( can_reduce_later_on_host ) {
      // transfer the reduced data from device to host
      std::unique_ptr< Result[] > resultArray{ new Result[ reducedSize ] };
      ArrayOperations< Devices::Host, Devices::Cuda >::copy( resultArray.get(), deviceAux1, reducedSize );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Transferring data to CPU took " << timer.getRealTime() << " sec. " << std::endl;
         timer.reset();
         timer.start();
      #endif

      // finish the reduction on the host
      auto fetch = [&] ( Index i ) { return resultArray[ i ]; };
      const Result result = Reduction< Devices::Host >::reduce( reducedSize, reduction, volatileReduction, fetch, zero );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Reduction of small data set on CPU took " << timer.getRealTime() << " sec. " << std::endl;
      #endif
      return result;
   }
   else {
      // data can't be safely reduced on host, so continue with the reduction on the GPU
      auto result = reductionLauncher.finish( reduction, zero );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Reduction of small data set on GPU took " << timer.getRealTime() << " sec. " << std::endl;
         timer.reset();
         timer.start();
      #endif

      return result;
   }
}

template< typename Index,
          typename Result,
          typename ReductionOperation,
          typename VolatileReductionOperation,
          typename DataFetcher >
std::pair< Index, Result >
Reduction< Devices::Cuda >::
reduceWithArgument( const Index size,
                    ReductionOperation& reduction,
                    VolatileReductionOperation& volatileReduction,
                    DataFetcher& dataFetcher,
                    const Result& zero )
{
   // Only fundamental and pointer types can be safely reduced on host. Complex
   // objects stored on the device might contain pointers into the device memory,
   // in which case reduction on host might fail.
   constexpr bool can_reduce_later_on_host = std::is_fundamental< Result >::value || std::is_pointer< Result >::value;

   #ifdef CUDA_REDUCTION_PROFILING
      Timer timer;
      timer.reset();
      timer.start();
   #endif

   CudaReductionKernelLauncher< Index, Result > reductionLauncher( size );

   // start the reduction on the GPU
   Result* deviceAux1( nullptr );
   Index* deviceIndexes( nullptr );
   Index reducedSize = reductionLauncher.startWithArgument(
      reduction,
      dataFetcher,
      zero,
      deviceAux1,
      deviceIndexes );

   #ifdef CUDA_REDUCTION_PROFILING
      timer.stop();
      std::cout << "   Reduction on GPU to size " << reducedSize << " took " << timer.getRealTime() << " sec. " << std::endl;
      timer.reset();
      timer.start();
   #endif

   if( can_reduce_later_on_host ) {
      // transfer the reduced data from device to host
      std::unique_ptr< Result[] > resultArray{ new Result[ reducedSize ] };
      std::unique_ptr< Index[] > indexArray{ new Index[ reducedSize ] };
      ArrayOperations< Devices::Host, Devices::Cuda >::copy( resultArray.get(), deviceAux1, reducedSize );
      ArrayOperations< Devices::Host, Devices::Cuda >::copy( indexArray.get(), deviceIndexes, reducedSize );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Transferring data to CPU took " << timer.getRealTime() << " sec. " << std::endl;
         timer.reset();
         timer.start();
      #endif

      // finish the reduction on the host
//      auto fetch = [&] ( Index i ) { return resultArray[ i ]; };
//      const Result result = Reduction< Devices::Host >::reduceWithArgument( reducedSize, argument, reduction, volatileReduction, fetch, zero );
      for( Index i = 1; i < reducedSize; i++ )
         reduction( indexArray[ 0 ], indexArray[ i ], resultArray[ 0 ], resultArray[ i ] );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Reduction of small data set on CPU took " << timer.getRealTime() << " sec. " << std::endl;
      #endif
      return std::make_pair( indexArray[ 0 ], resultArray[ 0 ] );
   }
   else {
      // data can't be safely reduced on host, so continue with the reduction on the GPU
      auto result = reductionLauncher.finishWithArgument( reduction, zero );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Reduction of small data set on GPU took " << timer.getRealTime() << " sec. " << std::endl;
         timer.reset();
         timer.start();
      #endif

      return result;
   }
}

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
