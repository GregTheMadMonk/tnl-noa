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
#include <TNL/Containers/Algorithms/reduction-operations.h>
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
template< typename Operation >
bool
Multireduction< Devices::Cuda >::
reduce( Operation& operation,
        int n,
        const typename Operation::IndexType size,
        const typename Operation::RealType* deviceInput1,
        const typename Operation::IndexType ldInput1,
        const typename Operation::RealType* deviceInput2,
        typename Operation::ResultType* hostResult )
{
#ifdef HAVE_CUDA
   TNL_ASSERT( n > 0, );
   TNL_ASSERT( size <= ldInput1, );

   typedef typename Operation::IndexType IndexType;
   typedef typename Operation::RealType RealType;
   typedef typename Operation::ResultType ResultType;
   typedef typename Operation::LaterReductionOperation LaterReductionOperation;

   /***
    * First check if the input array(s) is/are large enough for the multireduction on GPU.
    * Otherwise copy it/them to host and multireduce on CPU.
    */
   if( n * ldInput1 < Multireduction_minGpuDataSize ) {
      RealType hostArray1[ Multireduction_minGpuDataSize ];
      if( ! ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< RealType, RealType, IndexType >( hostArray1, deviceInput1, n * ldInput1 ) )
         return false;
      if( deviceInput2 ) {
         RealType hostArray2[ Multireduction_minGpuDataSize ];
         if( ! ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< RealType, RealType, IndexType >( hostArray2, deviceInput2, size ) )
            return false;
         return Multireduction< Devices::Host >::reduce( operation, n, size, hostArray1, ldInput1, hostArray2, hostResult );
      }
      else {
         return Multireduction< Devices::Host >::reduce( operation, n, size, hostArray1, ldInput1, ( RealType* ) nullptr, hostResult );
      }
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
   if( ! ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory< ResultType, ResultType, IndexType >( resultArray, deviceAux1, n * reducedSize ) )
      return false;

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
   Multireduction< Devices::Host >::reduce( laterReductionOperation, n, reducedSize, resultArray, reducedSize, (RealType*) nullptr, hostResult );

   #ifdef CUDA_REDUCTION_PROFILING
      timer.stop();
      std::cout << "   Multireduction of small data set on CPU took " << timer.getRealTime() << " sec. " << std::endl;
   #endif

   return checkCudaDevice;
#else
   CudaSupportMissingMessage;
   return false;
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
template< typename Operation >
bool
Multireduction< Devices::Host >::
reduce( Operation& operation,
        int n,
        const typename Operation::IndexType size,
        const typename Operation::RealType* input1,
        const typename Operation::IndexType ldInput1,
        const typename Operation::RealType* input2,
        typename Operation::ResultType* result )
{
   TNL_ASSERT( n > 0, );
   TNL_ASSERT( size <= ldInput1, );

   typedef typename Operation::IndexType IndexType;
   typedef typename Operation::RealType RealType;
   typedef typename Operation::ResultType ResultType;

   const int block_size = 128;
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
      ResultType r[ n ];
      for( int k = 0; k < n; k++ )
         r[ k ] = operation.initialValue();

      #pragma omp for nowait
      for( int b = 0; b < blocks; b++ ) {
         const int offset = b * block_size;
         for( int k = 0; k < n; k++ ) {
            const RealType* _input1 = input1 + k * ldInput1;
            for( IndexType i = 0; i < block_size; i++ )
               r[ k ] = operation.reduceOnHost( offset + i, r[ k ], _input1, input2 );
         }
      }

      // the first thread that reaches here processes the last, incomplete block
      #pragma omp single nowait
      {
         for( int k = 0; k < n; k++ ) {
            const RealType* _input1 = input1 + k * ldInput1;
            for( IndexType i = blocks * block_size; i < size; i++ )
               r[ k ] = operation.reduceOnHost( i, r[ k ], _input1, input2 );
         }
      }

      // inter-thread reduction of local results
      #pragma omp critical
      {
         for( int k = 0; k < n; k++ )
            operation.commonReductionOnDevice( result[ k ], r[ k ] );
      }
   }
   else {
#endif
      for( int k = 0; k < n; k++ )
         result[ k ] = operation.initialValue();

      for( int b = 0; b < blocks; b++ ) {
         const int offset = b * block_size;
         for( int k = 0; k < n; k++ ) {
            const RealType* _input1 = input1 + k * ldInput1;
            for( IndexType i = 0; i < block_size; i++ )
               result[ k ] = operation.reduceOnHost( offset + i, result[ k ], _input1, input2 );
         }
      }

      for( int k = 0; k < n; k++ ) {
         const RealType* _input1 = input1 + k * ldInput1;
         for( IndexType i = blocks * block_size; i < size; i++ )
            result[ k ] = operation.reduceOnHost( i, result[ k ], _input1, input2 );
      }
#ifdef HAVE_OPENMP
   }
#endif

   return true;
}

template< typename Operation >
bool
Multireduction< Devices::MIC >::
reduce( Operation& operation,
        int n,
        const typename Operation::IndexType size,
        const typename Operation::RealType* input1,
        const typename Operation::IndexType ldInput1,
        const typename Operation::RealType* input2,
        typename Operation::ResultType* result )
{
   TNL_ASSERT( n > 0, );
   TNL_ASSERT( size <= ldInput1, );

   typedef typename Operation::IndexType IndexType;
   typedef typename Operation::RealType RealType;
   typedef typename Operation::ResultType ResultType;


   std::cout << "Not Implemented yet Multireduction< Devices::MIC >::reduce" << std::endl;
   return true;
}


} // namespace Algorithms
} // namespace Containers
} // namespace TNL
