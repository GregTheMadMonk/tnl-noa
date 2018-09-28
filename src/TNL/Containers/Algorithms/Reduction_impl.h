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

template< typename Operation, typename Index >
void
Reduction< Devices::Cuda >::
reduce( Operation& operation,
        const Index size,
        const typename Operation::DataType1* deviceInput1,
        const typename Operation::DataType2* deviceInput2,
        typename Operation::ResultType& result )
{
#ifdef HAVE_CUDA

   typedef Index IndexType;
   typedef typename Operation::DataType1 DataType1;
   typedef typename Operation::DataType2 DataType2;
   typedef typename Operation::ResultType ResultType;
   typedef typename Operation::LaterReductionOperation LaterReductionOperation;

   /***
    * Only fundamental and pointer types can be safely reduced on host. Complex
    * objects stored on the device might contain pointers into the device memory,
    * in which case reduction on host might fail.
    */
   constexpr bool can_reduce_all_on_host = std::is_fundamental< DataType1 >::value || std::is_fundamental< DataType2 >::value || std::is_pointer< DataType1 >::value || std::is_pointer< DataType2 >::value;
   constexpr bool can_reduce_later_on_host = std::is_fundamental< ResultType >::value || std::is_pointer< ResultType >::value;

   /***
    * First check if the input array(s) is/are large enough for the reduction on GPU.
    * Otherwise copy it/them to host and reduce on CPU.
    */
   if( can_reduce_all_on_host && size <= Reduction_minGpuDataSize )
   {
      typename std::remove_const< DataType1 >::type hostArray1[ Reduction_minGpuDataSize ];
      ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory( hostArray1, deviceInput1, size );
      if( deviceInput2 ) {
         using _DT2 = typename std::conditional< std::is_same< DataType2, void >::value, DataType1, DataType2 >::type;
         typename std::remove_const< _DT2 >::type hostArray2[ Reduction_minGpuDataSize ];
         ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory( hostArray2, (_DT2*) deviceInput2, size );
         Reduction< Devices::Host >::reduce( operation, size, hostArray1, hostArray2, result );
      }
      else {
         Reduction< Devices::Host >::reduce( operation, size, hostArray1, (DataType2*) nullptr, result );
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
   ResultType* deviceAux1( 0 );
   IndexType reducedSize = CudaReductionKernelLauncher( operation,
                                                        size,
                                                        deviceInput1,
                                                        deviceInput2,
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
      ResultType resultArray[ reducedSize ];
      ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory( resultArray, deviceAux1, reducedSize );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Transferring data to CPU took " << timer.getRealTime() << " sec. " << std::endl;
         timer.reset();
         timer.start();
      #endif

      /***
       * Reduce the data on the host system.
       */
      LaterReductionOperation laterReductionOperation;
      Reduction< Devices::Host >::reduce( laterReductionOperation, reducedSize, resultArray, (void*) nullptr, result );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Reduction of small data set on CPU took " << timer.getRealTime() << " sec. " << std::endl;
      #endif
   }
   else {
      /***
       * Data can't be safely reduced on host, so continue with the reduction on the CUDA device.
       */
      LaterReductionOperation laterReductionOperation;
      while( reducedSize > 1 ) {
         reducedSize = CudaReductionKernelLauncher( laterReductionOperation,
                                                    reducedSize,
                                                    deviceAux1,
                                                    (ResultType*) 0,
                                                    deviceAux1 );
      }

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Reduction of small data set on GPU took " << timer.getRealTime() << " sec. " << std::endl;
         timer.reset();
         timer.start();
      #endif

      ResultType resultArray[ 1 ];
      ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory( resultArray, deviceAux1, reducedSize );
      result = resultArray[ 0 ];

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Transferring the result to CPU took " << timer.getRealTime() << " sec. " << std::endl;
      #endif
   }

   TNL_CHECK_CUDA_DEVICE;
#else
   throw Exceptions::CudaSupportMissing();
#endif
};

template< typename Operation, typename Index >
void
Reduction< Devices::Host >::
reduce( Operation& operation,
        const Index size,
        const typename Operation::DataType1* input1,
        const typename Operation::DataType2* input2,
        typename Operation::ResultType& result )
{
   typedef Index IndexType;
   typedef typename Operation::DataType1 DataType1;
   typedef typename Operation::DataType2 DataType2;
   typedef typename Operation::ResultType ResultType;

#ifdef HAVE_OPENMP
   constexpr int block_size = 128;
   if( TNL::Devices::Host::isOMPEnabled() && size >= 2 * block_size )
#pragma omp parallel
   {
      const int blocks = size / block_size;

      // first thread initializes the global result variable
      #pragma omp single nowait
      {
         result = operation.initialValue();
      }

      // initialize thread-local result variable
      ResultType r = operation.initialValue();

      #pragma omp for nowait
      for( int b = 0; b < blocks; b++ ) {
         const int offset = b * block_size;
         for( IndexType i = 0; i < block_size; i++ )
            operation.firstReduction( r, offset + i, input1, input2 );
      }

      // the first thread that reaches here processes the last, incomplete block
      #pragma omp single nowait
      {
         for( IndexType i = blocks * block_size; i < size; i++ )
            operation.firstReduction( r, i, input1, input2 );
      }

      // inter-thread reduction of local results
      #pragma omp critical
      {
         operation.commonReduction( result, r );
      }
   }
   else {
#endif
      result = operation.initialValue();
      for( IndexType i = 0; i < size; i++ )
         operation.firstReduction( result, i, input1, input2 );
#ifdef HAVE_OPENMP
   }
#endif
}

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
