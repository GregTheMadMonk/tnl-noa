/***************************************************************************
                          Reduction_impl.h  -  description
                             -------------------
    begin                : Mar 24, 2013
    copyright            : (C) 2013 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include "Reduction.h"

//#define CUDA_REDUCTION_PROFILING

#include <TNL/Assert.h>
#include <TNL/Exceptions/CudaSupportMissing.h>
#include <TNL/Containers/Algorithms/ReductionOperations.h>
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
#ifdef HAVE_CUDA

   using IndexType = Index;
   using ResultType = Result;

   /***
    * Only fundamental and pointer types can be safely reduced on host. Complex
    * objects stored on the device might contain pointers into the device memory,
    * in which case reduction on host might fail.
    */
   //constexpr bool can_reduce_all_on_host = std::is_fundamental< DataType1 >::value || std::is_fundamental< DataType2 >::value || std::is_pointer< DataType1 >::value || std::is_pointer< DataType2 >::value;
   constexpr bool can_reduce_later_on_host = std::is_fundamental< ResultType >::value || std::is_pointer< ResultType >::value;

   #ifdef CUDA_REDUCTION_PROFILING
      Timer timer;
      timer.reset();
      timer.start();
   #endif

   CudaReductionKernelLauncher< IndexType, ResultType > reductionLauncher( size );

   /****
    * Reduce the data on the CUDA device.
    */
   ResultType* deviceAux1( 0 );
   IndexType reducedSize = reductionLauncher.start(
      reduction,
      volatileReduction,
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
      /***
       * Transfer the reduced data from device to host.
       */
      std::unique_ptr< ResultType[] > resultArray{ new ResultType[ reducedSize ] };
      ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory( resultArray.get(), deviceAux1, reducedSize );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Transferring data to CPU took " << timer.getRealTime() << " sec. " << std::endl;
         timer.reset();
         timer.start();
      #endif

      /***
       * Reduce the data on the host system.
       */
      auto fetch = [&] ( IndexType i ) { return resultArray[ i ]; };
      const ResultType result = Reduction< Devices::Host >::reduce( reducedSize, reduction, volatileReduction, fetch, zero );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Reduction of small data set on CPU took " << timer.getRealTime() << " sec. " << std::endl;
      #endif
      return result;
   }
   else {
      /***
       * Data can't be safely reduced on host, so continue with the reduction on the CUDA device.
       */
      auto result = reductionLauncher.finish( reduction, volatileReduction, zero );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Reduction of small data set on GPU took " << timer.getRealTime() << " sec. " << std::endl;
         timer.reset();
         timer.start();
      #endif

      return result;
   }
#else
   throw Exceptions::CudaSupportMissing();
#endif
};

template< typename Index,
          typename Result,
          typename ReductionOperation,
          typename VolatileReductionOperation,
          typename DataFetcher >
Result
Reduction< Devices::Cuda >::
reduceWithArgument( const Index size,
                    Index& argument,
                    ReductionOperation& reduction,
                    VolatileReductionOperation& volatileReduction,
                    DataFetcher& dataFetcher,
                    const Result& zero )
{
   #ifdef HAVE_CUDA

   using IndexType = Index;
   using ResultType = Result;

   /***
    * Only fundamental and pointer types can be safely reduced on host. Complex
    * objects stored on the device might contain pointers into the device memory,
    * in which case reduction on host might fail.
    */
   //constexpr bool can_reduce_all_on_host = std::is_fundamental< DataType1 >::value || std::is_fundamental< DataType2 >::value || std::is_pointer< DataType1 >::value || std::is_pointer< DataType2 >::value;
   constexpr bool can_reduce_later_on_host = std::is_fundamental< ResultType >::value || std::is_pointer< ResultType >::value;

   #ifdef CUDA_REDUCTION_PROFILING
      Timer timer;
      timer.reset();
      timer.start();
   #endif

   CudaReductionKernelLauncher< IndexType, ResultType > reductionLauncher( size );

   /****
    * Reduce the data on the CUDA device.
    */
   ResultType* deviceAux1( nullptr );
   IndexType* deviceIndexes( nullptr );
   IndexType reducedSize = reductionLauncher.startWithArgument(
      reduction,
      volatileReduction,
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
      /***
       * Transfer the reduced data from device to host.
       */
      std::unique_ptr< ResultType[] > resultArray{ new ResultType[ reducedSize ] };
      std::unique_ptr< IndexType[] > indexArray{ new IndexType[ reducedSize ] };
      ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory( resultArray.get(), deviceAux1, reducedSize );
      ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory( indexArray.get(), deviceIndexes, reducedSize );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Transferring data to CPU took " << timer.getRealTime() << " sec. " << std::endl;
         timer.reset();
         timer.start();
      #endif

      /***
       * Reduce the data on the host system.
       */
      //auto fetch = [&] ( IndexType i ) { return resultArray[ i ]; };
      //const ResultType result = Reduction< Devices::Host >::reduceWithArgument( reducedSize, argument, reduction, volatileReduction, fetch, zero );
      for( IndexType i = 1; i < reducedSize; i++ )
         reduction( indexArray[ 0 ], indexArray[ i ], resultArray[ 0 ], resultArray[ i ] );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Reduction of small data set on CPU took " << timer.getRealTime() << " sec. " << std::endl;
      #endif
      argument = indexArray[ 0 ];
      return resultArray[ 0 ];
   }
   else {
      /***
       * Data can't be safely reduced on host, so continue with the reduction on the CUDA device.
       */
      auto result = reductionLauncher.finishWithArgument( argument, reduction, volatileReduction, zero );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Reduction of small data set on GPU took " << timer.getRealTime() << " sec. " << std::endl;
         timer.reset();
         timer.start();
      #endif

      return result;
   }
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

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
   using IndexType = Index;
   using ResultType = Result;

   constexpr int block_size = 128;
   const int blocks = size / block_size;

#ifdef HAVE_OPENMP
   if( TNL::Devices::Host::isOMPEnabled() && size >= 2 * block_size ) {
      // global result variable
      ResultType result = zero;
#pragma omp parallel
      {
         // initialize array for thread-local results
         ResultType r[ 4 ] = { zero, zero, zero, zero  };

         #pragma omp for nowait
         for( int b = 0; b < blocks; b++ ) {
            const IndexType offset = b * block_size;
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
            for( IndexType i = blocks * block_size; i < size; i++ )
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
         ResultType r[ 4 ] = { zero, zero, zero, zero };

         // main reduction (explicitly unrolled loop)
         for( int b = 0; b < blocks; b++ ) {
            const IndexType offset = b * block_size;
            for( int i = 0; i < block_size; i += 4 ) {
               reduction( r[ 0 ], dataFetcher( offset + i ) );
               reduction( r[ 1 ], dataFetcher( offset + i + 1 ) );
               reduction( r[ 2 ], dataFetcher( offset + i + 2 ) );
               reduction( r[ 3 ], dataFetcher( offset + i + 3 ) );
            }
         }

         // reduction of the last, incomplete block (not unrolled)
         for( IndexType i = blocks * block_size; i < size; i++ )
            reduction( r[ 0 ], dataFetcher( i ) );
            //operation.dataFetcher( r[ 0 ], i, input1, input2 );

         // reduction of unrolled results
         reduction( r[ 0 ], r[ 2 ] );
         reduction( r[ 1 ], r[ 3 ] );
         reduction( r[ 0 ], r[ 1 ] );
         return r[ 0 ];
      }
      else {
         ResultType result = zero;
         for( IndexType i = 0; i < size; i++ )
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
Result
Reduction< Devices::Host >::
reduceWithArgument( const Index size,
                    Index& argument,
                    ReductionOperation& reduction,
                    VolatileReductionOperation& volatileReduction,
                    DataFetcher& dataFetcher,
                    const Result& zero )
{
   using IndexType = Index;
   using ResultType = Result;

   constexpr int block_size = 128;
   const int blocks = size / block_size;

#ifdef HAVE_OPENMP
   if( TNL::Devices::Host::isOMPEnabled() && size >= 2 * block_size ) {
      // global result variable
      ResultType result = zero;
      argument = -1;
#pragma omp parallel
      {
         // initialize array for thread-local results
         ResultType r[ 4 ] = { zero, zero, zero, zero  };
         IndexType arg[ 4 ] = { 0, 0, 0, 0 };
         bool initialised( false );

         #pragma omp for nowait
         for( int b = 0; b < blocks; b++ ) {
            const IndexType offset = b * block_size;
            for( int i = 0; i < block_size; i += 4 ) {
               if( ! initialised ) {
                  arg[ 0 ] = offset + i;
                  arg[ 1 ] = offset + i + 1;
                  arg[ 2 ] = offset + i + 2;
                  arg[ 3 ] = offset + i + 3;
                  r[ 0 ] = dataFetcher( offset + i );
                  r[ 1 ] = dataFetcher( offset + i + 1 );
                  r[ 2 ] = dataFetcher( offset + i + 2 );
                  r[ 3 ] = dataFetcher( offset + i + 3 );
                  initialised = true;
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
            for( IndexType i = blocks * block_size; i < size; i++ )
               reduction( arg[ 0 ], i, r[ 0 ], dataFetcher( i ) );
         }

         // local reduction of unrolled results
         reduction( arg[ 0 ], arg[ 2 ], r[ 0 ], r[ 2 ] );
         reduction( arg[ 1 ], arg[ 3 ], r[ 1 ], r[ 3 ] );
         reduction( arg[ 0 ], arg[ 1 ], r[ 0 ], r[ 1 ] );

         // inter-thread reduction of local results
         #pragma omp critical
         {
            if( argument == - 1 )
               argument = arg[ 0 ];
            reduction( argument, arg[ 0 ], result, r[ 0 ] );
         }
      }
      return result;
   }
   else {
#endif
      if( blocks > 1 ) {
         // initialize array for unrolled results
         ResultType r[ 4 ] = { zero, zero, zero, zero };
         IndexType arg[ 4 ] = { 0, 0, 0, 0 };
         bool initialised( false );

         // main reduction (explicitly unrolled loop)
         for( int b = 0; b < blocks; b++ ) {
            const IndexType offset = b * block_size;
            for( int i = 0; i < block_size; i += 4 ) {
               if( ! initialised )
               {
                  arg[ 0 ] = offset + i;
                  arg[ 1 ] = offset + i + 1;
                  arg[ 2 ] = offset + i + 2;
                  arg[ 3 ] = offset + i + 3;
                  r[ 0 ] = dataFetcher( offset + i );
                  r[ 1 ] = dataFetcher( offset + i + 1 );
                  r[ 2 ] = dataFetcher( offset + i + 2 );
                  r[ 3 ] = dataFetcher( offset + i + 3 );
                  initialised = true;
                  continue;
               }
               reduction( arg[ 0 ], offset + i,     r[ 0 ], dataFetcher( offset + i ) );
               reduction( arg[ 1 ], offset + i + 1, r[ 1 ], dataFetcher( offset + i + 1 ) );
               reduction( arg[ 2 ], offset + i + 2, r[ 2 ], dataFetcher( offset + i + 2 ) );
               reduction( arg[ 3 ], offset + i + 3, r[ 3 ], dataFetcher( offset + i + 3 ) );
            }
         }

         // reduction of the last, incomplete block (not unrolled)
         for( IndexType i = blocks * block_size; i < size; i++ )
            reduction( arg[ 0 ], i, r[ 0 ], dataFetcher( i ) );

         // reduction of unrolled results
         reduction( arg[ 0 ], arg[ 2 ], r[ 0 ], r[ 2 ] );
         reduction( arg[ 1 ], arg[ 3 ], r[ 1 ], r[ 3 ] );
         reduction( arg[ 0 ], arg[ 1 ], r[ 0 ], r[ 1 ] );
         argument = arg[ 0 ];
         return r[ 0 ];
      }
      else {
         ResultType result = dataFetcher( 0 );
         argument = 0;
         for( IndexType i = 1; i < size; i++ )
            reduction( argument, i, result, dataFetcher( i ) );
         return result;
      }
#ifdef HAVE_OPENMP
   }
#endif
}


} // namespace Algorithms
} // namespace Containers
} // namespace TNL
