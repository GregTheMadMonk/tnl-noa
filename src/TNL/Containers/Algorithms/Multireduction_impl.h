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

#include "Multireduction.h"

//#define CUDA_REDUCTION_PROFILING

#include <TNL/Assert.h>
#include <TNL/Exceptions/CudaSupportMissing.h>
#include <TNL/Exceptions/NotImplementedError.h>
#include <TNL/Containers/Algorithms/ReductionOperations.h>
#include <TNL/Containers/Algorithms/ArrayOperations.h>
#include <TNL/Containers/Algorithms/CudaMultireductionKernel.h>

#ifdef CUDA_REDUCTION_PROFILING
#include <TNL/Timer.h>
#include <iostream>
#endif

namespace TNL {
namespace Containers {
namespace Algorithms {

/****
 * Arrays smaller than the following constant are reduced on CPU.
 */
//static constexpr int Multireduction_minGpuDataSize = 16384;//65536; //16384;//1024;//256;
// TODO: benchmarks with different values
static constexpr int Multireduction_minGpuDataSize = 256;//65536; //16384;//1024;//256;

/*
 * Parameters:
 *    operation: the operation used for reduction
 *    n: number of datasets to be reduced
 *    size: the size of each dataset
 *    deviceInput1: input array of size = n * ldInput1
 *    ldInput1: leading dimension of the deviceInput1 array
 *    deviceInput2: either nullptr or input array of size = size
 *    hostResult: output array of size = n
 */
template< typename Operation, typename Index >
void
Multireduction< Devices::Cuda >::
reduce( Operation& operation,
        const int n,
        const Index size,
        const typename Operation::DataType1* deviceInput1,
        const Index ldInput1,
        const typename Operation::DataType2* deviceInput2,
        typename Operation::ResultType* hostResult )
{
#ifdef HAVE_CUDA
   TNL_ASSERT_GT( n, 0, "The number of datasets must be positive." );
   TNL_ASSERT_LE( size, ldInput1, "The size of the input cannot exceed its leading dimension." );

   typedef Index IndexType;
   typedef typename Operation::DataType1 DataType1;
   typedef typename Operation::DataType2 DataType2;
   typedef typename Operation::ResultType ResultType;
   typedef typename Operation::LaterReductionOperation LaterReductionOperation;

   /***
    * First check if the input array(s) is/are large enough for the multireduction on GPU.
    * Otherwise copy it/them to host and multireduce on CPU.
    */
   if( n * ldInput1 < Multireduction_minGpuDataSize ) {
      DataType1 hostArray1[ Multireduction_minGpuDataSize ];
      ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory( hostArray1, deviceInput1, n * ldInput1 );
      if( deviceInput2 ) {
         using _DT2 = typename std::conditional< std::is_same< DataType2, void >::value, DataType1, DataType2 >::type;
         _DT2 hostArray2[ Multireduction_minGpuDataSize ];
         ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory( hostArray2, (_DT2*) deviceInput2, size );
         Multireduction< Devices::Host >::reduce( operation, n, size, hostArray1, ldInput1, hostArray2, hostResult );
      }
      else {
         Multireduction< Devices::Host >::reduce( operation, n, size, hostArray1, ldInput1, (DataType2*) nullptr, hostResult );
      }
      return;
   }

   #ifdef CUDA_REDUCTION_PROFILING
      Timer timer;
      timer.reset();
      timer.start();
   #endif

   /****
    * Reduce the data on the CUDA device.
    */
   ResultType* deviceAux1 = nullptr;
   const IndexType reducedSize = CudaMultireductionKernelLauncher( operation,
                                                                   n,
                                                                   size,
                                                                   deviceInput1,
                                                                   ldInput1,
                                                                   deviceInput2,
                                                                   deviceAux1 );
   #ifdef CUDA_REDUCTION_PROFILING
      timer.stop();
      std::cout << "   Multireduction of " << n << " datasets on GPU to size " << reducedSize << " took " << timer.getRealTime() << " sec. " << std::endl;
      timer.reset();
      timer.start();
   #endif

   /***
    * Transfer the reduced data from device to host.
    */
   ResultType resultArray[ n * reducedSize ];
   ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory( resultArray, deviceAux1, n * reducedSize );

   #ifdef CUDA_REDUCTION_PROFILING
      timer.stop();
      std::cout << "   Transferring data to CPU took " << timer.getRealTime() << " sec. " << std::endl;
      timer.reset();
      timer.start();
   #endif

//   std::cout << "resultArray = [";
//   for( int i = 0; i < n * reducedSize; i++ ) {
//      std::cout << resultArray[ i ];
//      if( i < n * reducedSize - 1 )
//         std::cout << ", ";
//   }
//   std::cout << "]" << std::endl;

   /***
    * Reduce the data on the host system.
    */
   LaterReductionOperation laterReductionOperation;
   Multireduction< Devices::Host >::reduce( laterReductionOperation, n, reducedSize, resultArray, reducedSize, (void*) nullptr, hostResult );

   #ifdef CUDA_REDUCTION_PROFILING
      timer.stop();
      std::cout << "   Multireduction of small data set on CPU took " << timer.getRealTime() << " sec. " << std::endl;
   #endif

   TNL_CHECK_CUDA_DEVICE;
#else
   throw Exceptions::CudaSupportMissing();
#endif
};

/*
 * Parameters:
 *    operation: the operation used for reduction
 *    n: number of datasets to be reduced
 *    size: the size of each dataset
 *    input1: input array of size = n * ldInput1
 *    ldInput1: leading dimension of the input1 array
 *    input2: either nullptr or input array of size = size
 *    hostResult: output array of size = n
 */
template< typename Operation, typename Index >
void
Multireduction< Devices::Host >::
reduce( Operation& operation,
        const int n,
        const Index size,
        const typename Operation::DataType1* input1,
        const Index ldInput1,
        const typename Operation::DataType2* input2,
        typename Operation::ResultType* result )
{
   TNL_ASSERT_GT( n, 0, "The number of datasets must be positive." );
   TNL_ASSERT_LE( size, ldInput1, "The size of the input cannot exceed its leading dimension." );

   typedef Index IndexType;
   typedef typename Operation::DataType1 DataType1;
   typedef typename Operation::DataType2 DataType2;
   typedef typename Operation::ResultType ResultType;

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
            result[ k ] = operation.initialValue();
      }

      // initialize array for thread-local results
      // (it is accessed as a row-major matrix with n rows and 4 columns)
      ResultType r[ n * 4 ];
      for( int k = 0; k < n * 4; k++ )
         r[ k ] = operation.initialValue();

      #pragma omp for nowait
      for( int b = 0; b < blocks; b++ ) {
         const IndexType offset = b * block_size;
         for( int k = 0; k < n; k++ ) {
            const DataType1* _input1 = input1 + k * ldInput1;
            ResultType* _r = r + 4 * k;
            for( int i = 0; i < block_size; i += 4 ) {
               operation.firstReduction( _r[ 0 ], offset + i,     _input1, input2 );
               operation.firstReduction( _r[ 1 ], offset + i + 1, _input1, input2 );
               operation.firstReduction( _r[ 2 ], offset + i + 2, _input1, input2 );
               operation.firstReduction( _r[ 3 ], offset + i + 3, _input1, input2 );
            }
         }
      }

      // the first thread that reaches here processes the last, incomplete block
      #pragma omp single nowait
      {
         for( int k = 0; k < n; k++ ) {
            const DataType1* _input1 = input1 + k * ldInput1;
            ResultType* _r = r + 4 * k;
            for( IndexType i = blocks * block_size; i < size; i++ )
               operation.firstReduction( _r[ 0 ], i, _input1, input2 );
         }
      }

      // local reduction of unrolled results
      for( int k = 0; k < n; k++ ) {
         ResultType* _r = r + 4 * k;
         operation.commonReduction( _r[ 0 ], _r[ 1 ] );
         operation.commonReduction( _r[ 0 ], _r[ 2 ] );
         operation.commonReduction( _r[ 0 ], _r[ 3 ] );
      }

      // inter-thread reduction of local results
      #pragma omp critical
      {
         for( int k = 0; k < n; k++ )
            operation.commonReduction( result[ k ], r[ 4 * k ] );
      }
   }
   else {
#endif
      if( blocks > 1 ) {
         // initialize array for unrolled results
         // (it is accessed as a row-major matrix with n rows and 4 columns)
         ResultType r[ n * 4 ];
         for( int k = 0; k < n * 4; k++ )
            r[ k ] = operation.initialValue();

         // main reduction (explicitly unrolled loop)
         for( int b = 0; b < blocks; b++ ) {
            const IndexType offset = b * block_size;
            for( int k = 0; k < n; k++ ) {
               const DataType1* _input1 = input1 + k * ldInput1;
               ResultType* _r = r + 4 * k;
               for( int i = 0; i < block_size; i += 4 ) {
                  operation.firstReduction( _r[ 0 ], offset + i,     _input1, input2 );
                  operation.firstReduction( _r[ 1 ], offset + i + 1, _input1, input2 );
                  operation.firstReduction( _r[ 2 ], offset + i + 2, _input1, input2 );
                  operation.firstReduction( _r[ 3 ], offset + i + 3, _input1, input2 );
               }
            }
         }

         // reduction of the last, incomplete block (not unrolled)
         for( int k = 0; k < n; k++ ) {
            const DataType1* _input1 = input1 + k * ldInput1;
            ResultType* _r = r + 4 * k;
            for( IndexType i = blocks * block_size; i < size; i++ )
               operation.firstReduction( _r[ 0 ], i, _input1, input2 );
         }

         // reduction of unrolled results
         for( int k = 0; k < n; k++ ) {
            ResultType* _r = r + 4 * k;
            operation.commonReduction( _r[ 0 ], _r[ 1 ] );
            operation.commonReduction( _r[ 0 ], _r[ 2 ] );
            operation.commonReduction( _r[ 0 ], _r[ 3 ] );

            // copy the result into the output parameter
            result[ k ] = _r[ 0 ];
         }
      }
      else {
         for( int k = 0; k < n; k++ )
            result[ k ] = operation.initialValue();

         for( int b = 0; b < blocks; b++ ) {
            const IndexType offset = b * block_size;
            for( int k = 0; k < n; k++ ) {
               const DataType1* _input1 = input1 + k * ldInput1;
               for( int i = 0; i < block_size; i++ )
                  operation.firstReduction( result[ k ], offset + i, _input1, input2 );
            }
         }

         for( int k = 0; k < n; k++ ) {
            const DataType1* _input1 = input1 + k * ldInput1;
            for( IndexType i = blocks * block_size; i < size; i++ )
               operation.firstReduction( result[ k ], i, _input1, input2 );
         }
      }
#ifdef HAVE_OPENMP
   }
#endif
}

template< typename Operation, typename Index >
void
Multireduction< Devices::MIC >::
reduce( Operation& operation,
        const int n,
        const Index size,
        const typename Operation::DataType1* input1,
        const Index ldInput1,
        const typename Operation::DataType2* input2,
        typename Operation::ResultType* result )
{
   throw Exceptions::NotImplementedError("Multireduction is not implemented for MIC.");
}

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
