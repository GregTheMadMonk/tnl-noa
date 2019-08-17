/***************************************************************************
                          Multireduction_impl.h  -  description
                             -------------------
    begin                : May 13, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <memory>  // std::unique_ptr

//#define CUDA_REDUCTION_PROFILING

#include <TNL/Assert.h>
#include <TNL/Containers/Algorithms/Multireduction.h>
#include <TNL/Containers/Algorithms/ArrayOperations.h>
#include <TNL/Containers/Algorithms/CudaMultireductionKernel.h>

#ifdef CUDA_REDUCTION_PROFILING
#include <TNL/Timer.h>
#include <iostream>
#endif

namespace TNL {
namespace Containers {
namespace Algorithms {

template< typename Result,
          typename DataFetcher,
          typename Reduction,
          typename VolatileReduction,
          typename Index >
void
Multireduction< Devices::Host >::
reduce( const Result zero,
        DataFetcher dataFetcher,
        const Reduction reduction,
        const VolatileReduction volatileReduction,
        const Index size,
        const int n,
        Result* result )
{
   TNL_ASSERT_GT( size, 0, "The size of datasets must be positive." );
   TNL_ASSERT_GT( n, 0, "The number of datasets must be positive." );

   constexpr int block_size = 128;
   const int blocks = size / block_size;

#ifdef HAVE_OPENMP
   if( TNL::Devices::Host::isOMPEnabled() && blocks >= 2 )
#pragma omp parallel
   {
      // first thread initializes the result array
      #pragma omp single nowait
      {
         for( int k = 0; k < n; k++ )
            result[ k ] = zero;
      }

      // initialize array for thread-local results
      // (it is accessed as a row-major matrix with n rows and 4 columns)
      Result r[ n * 4 ];
      for( int k = 0; k < n * 4; k++ )
         r[ k ] = zero;

      #pragma omp for nowait
      for( int b = 0; b < blocks; b++ ) {
         const Index offset = b * block_size;
         for( int k = 0; k < n; k++ ) {
            Result* _r = r + 4 * k;
            for( int i = 0; i < block_size; i += 4 ) {
               reduction( _r[ 0 ], dataFetcher( offset + i,     k ) );
               reduction( _r[ 1 ], dataFetcher( offset + i + 1, k ) );
               reduction( _r[ 2 ], dataFetcher( offset + i + 2, k ) );
               reduction( _r[ 3 ], dataFetcher( offset + i + 3, k ) );
            }
         }
      }

      // the first thread that reaches here processes the last, incomplete block
      #pragma omp single nowait
      {
         for( int k = 0; k < n; k++ ) {
            Result* _r = r + 4 * k;
            for( Index i = blocks * block_size; i < size; i++ )
               reduction( _r[ 0 ], dataFetcher( i, k ) );
         }
      }

      // local reduction of unrolled results
      for( int k = 0; k < n; k++ ) {
         Result* _r = r + 4 * k;
         reduction( _r[ 0 ], _r[ 1 ] );
         reduction( _r[ 0 ], _r[ 2 ] );
         reduction( _r[ 0 ], _r[ 3 ] );
      }

      // inter-thread reduction of local results
      #pragma omp critical
      {
         for( int k = 0; k < n; k++ )
            reduction( result[ k ], r[ 4 * k ] );
      }
   }
   else {
#endif
      if( blocks > 1 ) {
         // initialize array for unrolled results
         // (it is accessed as a row-major matrix with n rows and 4 columns)
         Result r[ n * 4 ];
         for( int k = 0; k < n * 4; k++ )
            r[ k ] = zero;

         // main reduction (explicitly unrolled loop)
         for( int b = 0; b < blocks; b++ ) {
            const Index offset = b * block_size;
            for( int k = 0; k < n; k++ ) {
               Result* _r = r + 4 * k;
               for( int i = 0; i < block_size; i += 4 ) {
                  reduction( _r[ 0 ], dataFetcher( offset + i,     k ) );
                  reduction( _r[ 1 ], dataFetcher( offset + i + 1, k ) );
                  reduction( _r[ 2 ], dataFetcher( offset + i + 2, k ) );
                  reduction( _r[ 3 ], dataFetcher( offset + i + 3, k ) );
               }
            }
         }

         // reduction of the last, incomplete block (not unrolled)
         for( int k = 0; k < n; k++ ) {
            Result* _r = r + 4 * k;
            for( Index i = blocks * block_size; i < size; i++ )
               reduction( _r[ 0 ], dataFetcher( i, k ) );
         }

         // reduction of unrolled results
         for( int k = 0; k < n; k++ ) {
            Result* _r = r + 4 * k;
            reduction( _r[ 0 ], _r[ 1 ] );
            reduction( _r[ 0 ], _r[ 2 ] );
            reduction( _r[ 0 ], _r[ 3 ] );

            // copy the result into the output parameter
            result[ k ] = _r[ 0 ];
         }
      }
      else {
         for( int k = 0; k < n; k++ )
            result[ k ] = zero;

         for( int b = 0; b < blocks; b++ ) {
            const Index offset = b * block_size;
            for( int k = 0; k < n; k++ ) {
               for( int i = 0; i < block_size; i++ )
                  reduction( result[ k ], dataFetcher( offset + i, k ) );
            }
         }

         for( int k = 0; k < n; k++ ) {
            for( Index i = blocks * block_size; i < size; i++ )
               reduction( result[ k ], dataFetcher( i, k ) );
         }
      }
#ifdef HAVE_OPENMP
   }
#endif
}

template< typename Result,
          typename DataFetcher,
          typename Reduction,
          typename VolatileReduction,
          typename Index >
void
Multireduction< Devices::Cuda >::
reduce( const Result zero,
        DataFetcher dataFetcher,
        const Reduction reduction,
        const VolatileReduction volatileReduction,
        const Index size,
        const int n,
        Result* hostResult )
{
   TNL_ASSERT_GT( size, 0, "The size of datasets must be positive." );
   TNL_ASSERT_GT( n, 0, "The number of datasets must be positive." );

   #ifdef CUDA_REDUCTION_PROFILING
      Timer timer;
      timer.reset();
      timer.start();
   #endif

   // start the reduction on the GPU
   Result* deviceAux1 = nullptr;
   const int reducedSize = CudaMultireductionKernelLauncher( zero, dataFetcher, reduction, volatileReduction, size, n, deviceAux1 );

   #ifdef CUDA_REDUCTION_PROFILING
      timer.stop();
      std::cout << "   Multireduction of " << n << " datasets on GPU to size " << reducedSize << " took " << timer.getRealTime() << " sec. " << std::endl;
      timer.reset();
      timer.start();
   #endif

   // transfer the reduced data from device to host
   std::unique_ptr< Result[] > resultArray{ new Result[ n * reducedSize ] };
   ArrayOperations< Devices::Host, Devices::Cuda >::copy( resultArray.get(), deviceAux1, n * reducedSize );

   #ifdef CUDA_REDUCTION_PROFILING
      timer.stop();
      std::cout << "   Transferring data to CPU took " << timer.getRealTime() << " sec. " << std::endl;
      timer.reset();
      timer.start();
   #endif

   // finish the reduction on the host
   auto dataFetcherFinish = [&] ( int i, int k ) { return resultArray[ i + k * reducedSize ]; };
   Multireduction< Devices::Host >::reduce( zero, dataFetcherFinish, reduction, volatileReduction, reducedSize, n, hostResult );

   #ifdef CUDA_REDUCTION_PROFILING
      timer.stop();
      std::cout << "   Multireduction of small data set on CPU took " << timer.getRealTime() << " sec. " << std::endl;
   #endif
};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
