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
bool
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
      DataType1 hostArray1[ Reduction_minGpuDataSize ];
      if( ! ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory( hostArray1, deviceInput1, size ) )
         return false;
      if( deviceInput2 ) {
         using _DT2 = typename std::conditional< std::is_same< DataType2, void >::value, DataType1, DataType2 >::type;
         _DT2 hostArray2[ Reduction_minGpuDataSize ];
         if( ! ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory( hostArray2, (_DT2*) deviceInput2, size ) )
            return false;
         return Reduction< Devices::Host >::reduce( operation, size, hostArray1, hostArray2, result );
      }
      else {
         return Reduction< Devices::Host >::reduce( operation, size, hostArray1, (DataType2*) nullptr, result );
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
      if( ! ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory( resultArray, deviceAux1, reducedSize ) )
         return false;
    
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
      if( ! ArrayOperations< Devices::Host, Devices::Cuda >::copyMemory( resultArray, deviceAux1, reducedSize ) )
         return false;
      result = resultArray[ 0 ];

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Transferring the result to CPU took " << timer.getRealTime() << " sec. " << std::endl;
      #endif
   }
 
   return TNL_CHECK_CUDA_DEVICE;
#else
   throw Exceptions::CudaSupportMissing();
#endif
};

template< typename Operation, typename Index >
bool
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

   return true;
}


#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

/****
 * Sum
 */
extern template bool reductionOnCudaDevice< tnlParallelReductionSum< char, int > >
                                   ( const tnlParallelReductionSum< char, int >& operation,
                                     const typename tnlParallelReductionSum< char, int > :: IndexType size,
                                     const typename tnlParallelReductionSum< char, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionSum< char, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionSum< char, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionSum< int, int > >
                                   ( const tnlParallelReductionSum< int, int >& operation,
                                     const typename tnlParallelReductionSum< int, int > :: IndexType size,
                                     const typename tnlParallelReductionSum< int, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionSum< int, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionSum< int, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionSum< float, int > >
                                   ( const tnlParallelReductionSum< float, int >& operation,
                                     const typename tnlParallelReductionSum< float, int > :: IndexType size,
                                     const typename tnlParallelReductionSum< float, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionSum< float, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionSum< float, int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionSum< double, int > >
                                   ( const tnlParallelReductionSum< double, int>& operation,
                                     const typename tnlParallelReductionSum< double, int > :: IndexType size,
                                     const typename tnlParallelReductionSum< double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionSum< double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionSum< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionSum< long double, int > >
                                   ( const tnlParallelReductionSum< long double, int>& operation,
                                     const typename tnlParallelReductionSum< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionSum< long double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionSum< long double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionSum< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool reductionOnCudaDevice< tnlParallelReductionSum< char, long int > >
                                   ( const tnlParallelReductionSum< char, long int >& operation,
                                     const typename tnlParallelReductionSum< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionSum< char, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionSum< char, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionSum< char, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionSum< int, long int > >
                                   ( const tnlParallelReductionSum< int, long int >& operation,
                                     const typename tnlParallelReductionSum< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionSum< int, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionSum< int, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionSum< int, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionSum< float, long int > >
                                   ( const tnlParallelReductionSum< float, long int >& operation,
                                     const typename tnlParallelReductionSum< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionSum< float, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionSum< float, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionSum< float, long int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionSum< double, long int > >
                                   ( const tnlParallelReductionSum< double, long int>& operation,
                                     const typename tnlParallelReductionSum< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionSum< double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionSum< double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionSum< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionSum< long double, long int > >
                                   ( const tnlParallelReductionSum< long double, long int>& operation,
                                     const typename tnlParallelReductionSum< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionSum< long double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionSum< long double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionSum< long double, long int> :: ResultType& result );
#endif
#endif

/****
 * Min
 */
extern template bool reductionOnCudaDevice< tnlParallelReductionMin< char, int > >
                                   ( const tnlParallelReductionMin< char, int >& operation,
                                     const typename tnlParallelReductionMin< char, int > :: IndexType size,
                                     const typename tnlParallelReductionMin< char, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionMin< char, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionMin< char, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionMin< int, int > >
                                   ( const tnlParallelReductionMin< int, int >& operation,
                                     const typename tnlParallelReductionMin< int, int > :: IndexType size,
                                     const typename tnlParallelReductionMin< int, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionMin< int, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionMin< int, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionMin< float, int > >
                                   ( const tnlParallelReductionMin< float, int >& operation,
                                     const typename tnlParallelReductionMin< float, int > :: IndexType size,
                                     const typename tnlParallelReductionMin< float, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionMin< float, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionMin< float, int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionMin< double, int > >
                                   ( const tnlParallelReductionMin< double, int >& operation,
                                     const typename tnlParallelReductionMin< double, int > :: IndexType size,
                                     const typename tnlParallelReductionMin< double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionMin< double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionMin< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionMin< long double, int > >
                                   ( const tnlParallelReductionMin< long double, int>& operation,
                                     const typename tnlParallelReductionMin< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionMin< long double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionMin< long double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionMin< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool reductionOnCudaDevice< tnlParallelReductionMin< char, long int > >
                                   ( const tnlParallelReductionMin< char, long int >& operation,
                                     const typename tnlParallelReductionMin< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionMin< char, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionMin< char, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionMin< char, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionMin< int, long int > >
                                   ( const tnlParallelReductionMin< int, long int >& operation,
                                     const typename tnlParallelReductionMin< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionMin< int, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionMin< int, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionMin< int, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionMin< float, long int > >
                                   ( const tnlParallelReductionMin< float, long int >& operation,
                                     const typename tnlParallelReductionMin< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionMin< float, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionMin< float, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionMin< float, long int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionMin< double, long int > >
                                   ( const tnlParallelReductionMin< double, long int>& operation,
                                     const typename tnlParallelReductionMin< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionMin< double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionMin< double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionMin< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionMin< long double, long int > >
                                   ( const tnlParallelReductionMin< long double, long int>& operation,
                                     const typename tnlParallelReductionMin< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionMin< long double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionMin< long double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionMin< long double, long int> :: ResultType& result );
#endif
#endif

/****
 * Max
 */

extern template bool reductionOnCudaDevice< tnlParallelReductionMax< char, int > >
                                   ( const tnlParallelReductionMax< char, int >& operation,
                                     const typename tnlParallelReductionMax< char, int > :: IndexType size,
                                     const typename tnlParallelReductionMax< char, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionMax< char, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionMax< char, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionMax< int, int > >
                                   ( const tnlParallelReductionMax< int, int >& operation,
                                     const typename tnlParallelReductionMax< int, int > :: IndexType size,
                                     const typename tnlParallelReductionMax< int, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionMax< int, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionMax< int, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionMax< float, int > >
                                   ( const tnlParallelReductionMax< float, int >& operation,
                                     const typename tnlParallelReductionMax< float, int > :: IndexType size,
                                     const typename tnlParallelReductionMax< float, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionMax< float, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionMax< float, int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionMax< double, int > >
                                   ( const tnlParallelReductionMax< double, int>& operation,
                                     const typename tnlParallelReductionMax< double, int > :: IndexType size,
                                     const typename tnlParallelReductionMax< double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionMax< double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionMax< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionMax< long double, int > >
                                   ( const tnlParallelReductionMax< long double, int>& operation,
                                     const typename tnlParallelReductionMax< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionMax< long double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionMax< long double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionMax< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool reductionOnCudaDevice< tnlParallelReductionMax< char, long int > >
                                   ( const tnlParallelReductionMax< char, long int >& operation,
                                     const typename tnlParallelReductionMax< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionMax< char, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionMax< char, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionMax< char, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionMax< int, long int > >
                                   ( const tnlParallelReductionMax< int, long int >& operation,
                                     const typename tnlParallelReductionMax< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionMax< int, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionMax< int, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionMax< int, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionMax< float, long int > >
                                   ( const tnlParallelReductionMax< float, long int >& operation,
                                     const typename tnlParallelReductionMax< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionMax< float, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionMax< float, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionMax< float, long int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionMax< double, long int > >
                                   ( const tnlParallelReductionMax< double, long int>& operation,
                                     const typename tnlParallelReductionMax< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionMax< double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionMax< double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionMax< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionMax< long double, long int > >
                                   ( const tnlParallelReductionMax< long double, long int>& operation,
                                     const typename tnlParallelReductionMax< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionMax< long double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionMax< long double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionMax< long double, long int> :: ResultType& result );
#endif
#endif


/****
 * Abs sum
 */

extern template bool reductionOnCudaDevice< tnlParallelReductionAbsSum< char, int > >
                                   ( const tnlParallelReductionAbsSum< char, int >& operation,
                                     const typename tnlParallelReductionAbsSum< char, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsSum< char, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsSum< char, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsSum< char, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionAbsSum< int, int > >
                                   ( const tnlParallelReductionAbsSum< int, int >& operation,
                                     const typename tnlParallelReductionAbsSum< int, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsSum< int, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsSum< int, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsSum< int, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionAbsSum< float, int > >
                                   ( const tnlParallelReductionAbsSum< float, int >& operation,
                                     const typename tnlParallelReductionAbsSum< float, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsSum< float, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsSum< float, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsSum< float, int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionAbsSum< double, int > >
                                   ( const tnlParallelReductionAbsSum< double, int>& operation,
                                     const typename tnlParallelReductionAbsSum< double, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsSum< double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsSum< double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsSum< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionAbsSum< long double, int > >
                                   ( const tnlParallelReductionAbsSum< long double, int>& operation,
                                     const typename tnlParallelReductionAbsSum< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsSum< long double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsSum< long double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsSum< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool reductionOnCudaDevice< tnlParallelReductionAbsSum< char, long int > >
                                   ( const tnlParallelReductionAbsSum< char, long int >& operation,
                                     const typename tnlParallelReductionAbsSum< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsSum< char, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsSum< char, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsSum< char, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionAbsSum< int, long int > >
                                   ( const tnlParallelReductionAbsSum< int, long int >& operation,
                                     const typename tnlParallelReductionAbsSum< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsSum< int, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsSum< int, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsSum< int, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionAbsSum< float, long int > >
                                   ( const tnlParallelReductionAbsSum< float, long int >& operation,
                                     const typename tnlParallelReductionAbsSum< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsSum< float, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsSum< float, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsSum< float, long int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionAbsSum< double, long int > >
                                   ( const tnlParallelReductionAbsSum< double, long int>& operation,
                                     const typename tnlParallelReductionAbsSum< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsSum< double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsSum< double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsSum< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionAbsSum< long double, long int > >
                                   ( const tnlParallelReductionAbsSum< long double, long int>& operation,
                                     const typename tnlParallelReductionAbsSum< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsSum< long double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsSum< long double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsSum< long double, long int> :: ResultType& result );
#endif
#endif

/****
 * Abs min
 */

extern template bool reductionOnCudaDevice< tnlParallelReductionAbsMin< char, int > >
                                   ( const tnlParallelReductionAbsMin< char, int >& operation,
                                     const typename tnlParallelReductionAbsMin< char, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMin< char, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsMin< char, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsMin< char, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionAbsMin< int, int > >
                                   ( const tnlParallelReductionAbsMin< int, int >& operation,
                                     const typename tnlParallelReductionAbsMin< int, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMin< int, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsMin< int, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsMin< int, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionAbsMin< float, int > >
                                   ( const tnlParallelReductionAbsMin< float, int >& operation,
                                     const typename tnlParallelReductionAbsMin< float, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMin< float, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsMin< float, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsMin< float, int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionAbsMin< double, int > >
                                   ( const tnlParallelReductionAbsMin< double, int>& operation,
                                     const typename tnlParallelReductionAbsMin< double, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMin< double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsMin< double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsMin< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionAbsMin< long double, int > >
                                   ( const tnlParallelReductionAbsMin< long double, int>& operation,
                                     const typename tnlParallelReductionAbsMin< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMin< long double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsMin< long double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsMin< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool reductionOnCudaDevice< tnlParallelReductionAbsMin< char, long int > >
                                   ( const tnlParallelReductionAbsMin< char, long int >& operation,
                                     const typename tnlParallelReductionAbsMin< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMin< char, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsMin< char, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsMin< char, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionAbsMin< int, long int > >
                                   ( const tnlParallelReductionAbsMin< int, long int >& operation,
                                     const typename tnlParallelReductionAbsMin< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMin< int, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsMin< int, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsMin< int, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionAbsMin< float, long int > >
                                   ( const tnlParallelReductionAbsMin< float, long int >& operation,
                                     const typename tnlParallelReductionAbsMin< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMin< float, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsMin< float, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsMin< float, long int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionAbsMin< double, long int > >
                                   ( const tnlParallelReductionAbsMin< double, long int>& operation,
                                     const typename tnlParallelReductionAbsMin< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMin< double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsMin< double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsMin< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionAbsMin< long double, long int > >
                                   ( const tnlParallelReductionAbsMin< long double, long int>& operation,
                                     const typename tnlParallelReductionAbsMin< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMin< long double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsMin< long double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsMin< long double, long int> :: ResultType& result );
#endif
#endif

/****
 * Abs max
 */

extern template bool reductionOnCudaDevice< tnlParallelReductionAbsMax< char, int > >
                                   ( const tnlParallelReductionAbsMax< char, int >& operation,
                                     const typename tnlParallelReductionAbsMax< char, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMax< char, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsMax< char, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsMax< char, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionAbsMax< int, int > >
                                   ( const tnlParallelReductionAbsMax< int, int >& operation,
                                     const typename tnlParallelReductionAbsMax< int, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMax< int, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsMax< int, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsMax< int, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionAbsMax< float, int > >
                                   ( const tnlParallelReductionAbsMax< float, int >& operation,
                                     const typename tnlParallelReductionAbsMax< float, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMax< float, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsMax< float, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsMax< float, int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionAbsMax< double, int > >
                                   ( const tnlParallelReductionAbsMax< double, int>& operation,
                                     const typename tnlParallelReductionAbsMax< double, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMax< double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsMax< double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsMax< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionAbsMax< long double, int > >
                                   ( const tnlParallelReductionAbsMax< long double, int>& operation,
                                     const typename tnlParallelReductionAbsMax< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMax< long double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsMax< long double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsMax< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool reductionOnCudaDevice< tnlParallelReductionAbsMax< char, long int > >
                                   ( const tnlParallelReductionAbsMax< char, long int >& operation,
                                     const typename tnlParallelReductionAbsMax< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMax< char, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsMax< char, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsMax< char, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionAbsMax< int, long int > >
                                   ( const tnlParallelReductionAbsMax< int, long int >& operation,
                                     const typename tnlParallelReductionAbsMax< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMax< int, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsMax< int, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsMax< int, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionAbsMax< float, long int > >
                                   ( const tnlParallelReductionAbsMax< float, long int >& operation,
                                     const typename tnlParallelReductionAbsMax< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMax< float, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsMax< float, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsMax< float, long int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionAbsMax< double, long int > >
                                   ( const tnlParallelReductionAbsMax< double, long int>& operation,
                                     const typename tnlParallelReductionAbsMax< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMax< double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsMax< double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsMax< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionAbsMax< long double, long int > >
                                   ( const tnlParallelReductionAbsMax< long double, long int>& operation,
                                     const typename tnlParallelReductionAbsMax< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionAbsMax< long double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionAbsMax< long double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionAbsMax< long double, long int> :: ResultType& result );
#endif
#endif

/****
 * Logical AND
 */
extern template bool reductionOnCudaDevice< tnlParallelReductionLogicalAnd< char, int > >
                                   ( const tnlParallelReductionLogicalAnd< char, int >& operation,
                                     const typename tnlParallelReductionLogicalAnd< char, int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalAnd< char, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLogicalAnd< char, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLogicalAnd< char, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionLogicalAnd< int, int > >
                                   ( const tnlParallelReductionLogicalAnd< int, int >& operation,
                                     const typename tnlParallelReductionLogicalAnd< int, int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalAnd< int, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLogicalAnd< int, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLogicalAnd< int, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionLogicalAnd< float, int > >
                                   ( const tnlParallelReductionLogicalAnd< float, int >& operation,
                                     const typename tnlParallelReductionLogicalAnd< float, int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalAnd< float, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLogicalAnd< float, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLogicalAnd< float, int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionLogicalAnd< double, int > >
                                   ( const tnlParallelReductionLogicalAnd< double, int>& operation,
                                     const typename tnlParallelReductionLogicalAnd< double, int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalAnd< double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLogicalAnd< double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLogicalAnd< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionLogicalAnd< long double, int > >
                                   ( const tnlParallelReductionLogicalAnd< long double, int>& operation,
                                     const typename tnlParallelReductionLogicalAnd< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalAnd< long double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLogicalAnd< long double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLogicalAnd< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool reductionOnCudaDevice< tnlParallelReductionLogicalAnd< char, long int > >
                                   ( const tnlParallelReductionLogicalAnd< char, long int >& operation,
                                     const typename tnlParallelReductionLogicalAnd< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalAnd< char, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLogicalAnd< char, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLogicalAnd< char, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionLogicalAnd< int, long int > >
                                   ( const tnlParallelReductionLogicalAnd< int, long int >& operation,
                                     const typename tnlParallelReductionLogicalAnd< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalAnd< int, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLogicalAnd< int, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLogicalAnd< int, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionLogicalAnd< float, long int > >
                                   ( const tnlParallelReductionLogicalAnd< float, long int >& operation,
                                     const typename tnlParallelReductionLogicalAnd< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalAnd< float, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLogicalAnd< float, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLogicalAnd< float, long int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionLogicalAnd< double, long int > >
                                   ( const tnlParallelReductionLogicalAnd< double, long int>& operation,
                                     const typename tnlParallelReductionLogicalAnd< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalAnd< double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLogicalAnd< double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLogicalAnd< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionLogicalAnd< long double, long int > >
                                   ( const tnlParallelReductionLogicalAnd< long double, long int>& operation,
                                     const typename tnlParallelReductionLogicalAnd< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalAnd< long double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLogicalAnd< long double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLogicalAnd< long double, long int> :: ResultType& result );
#endif
#endif

/****
 * Logical OR
 */
extern template bool reductionOnCudaDevice< tnlParallelReductionLogicalOr< char, int > >
                                   ( const tnlParallelReductionLogicalOr< char, int >& operation,
                                     const typename tnlParallelReductionLogicalOr< char, int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalOr< char, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLogicalOr< char, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLogicalOr< char, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionLogicalOr< int, int > >
                                   ( const tnlParallelReductionLogicalOr< int, int >& operation,
                                     const typename tnlParallelReductionLogicalOr< int, int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalOr< int, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLogicalOr< int, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLogicalOr< int, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionLogicalOr< float, int > >
                                   ( const tnlParallelReductionLogicalOr< float, int >& operation,
                                     const typename tnlParallelReductionLogicalOr< float, int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalOr< float, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLogicalOr< float, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLogicalOr< float, int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionLogicalOr< double, int > >
                                   ( const tnlParallelReductionLogicalOr< double, int>& operation,
                                     const typename tnlParallelReductionLogicalOr< double, int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalOr< double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLogicalOr< double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLogicalOr< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionLogicalOr< long double, int > >
                                   ( const tnlParallelReductionLogicalOr< long double, int>& operation,
                                     const typename tnlParallelReductionLogicalOr< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalOr< long double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLogicalOr< long double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLogicalOr< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool reductionOnCudaDevice< tnlParallelReductionLogicalOr< char, long int > >
                                   ( const tnlParallelReductionLogicalOr< char, long int >& operation,
                                     const typename tnlParallelReductionLogicalOr< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalOr< char, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLogicalOr< char, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLogicalOr< char, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionLogicalOr< int, long int > >
                                   ( const tnlParallelReductionLogicalOr< int, long int >& operation,
                                     const typename tnlParallelReductionLogicalOr< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalOr< int, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLogicalOr< int, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLogicalOr< int, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionLogicalOr< float, long int > >
                                   ( const tnlParallelReductionLogicalOr< float, long int >& operation,
                                     const typename tnlParallelReductionLogicalOr< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalOr< float, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLogicalOr< float, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLogicalOr< float, long int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionLogicalOr< double, long int > >
                                   ( const tnlParallelReductionLogicalOr< double, long int>& operation,
                                     const typename tnlParallelReductionLogicalOr< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalOr< double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLogicalOr< double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLogicalOr< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionLogicalOr< long double, long int > >
                                   ( const tnlParallelReductionLogicalOr< long double, long int>& operation,
                                     const typename tnlParallelReductionLogicalOr< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionLogicalOr< long double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLogicalOr< long double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLogicalOr< long double, long int> :: ResultType& result );
#endif
#endif

/****
 * Lp Norm
 */
extern template bool reductionOnCudaDevice< tnlParallelReductionLpNorm< float, int > >
                                   ( const tnlParallelReductionLpNorm< float, int >& operation,
                                     const typename tnlParallelReductionLpNorm< float, int > :: IndexType size,
                                     const typename tnlParallelReductionLpNorm< float, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLpNorm< float, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLpNorm< float, int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionLpNorm< double, int > >
                                   ( const tnlParallelReductionLpNorm< double, int>& operation,
                                     const typename tnlParallelReductionLpNorm< double, int > :: IndexType size,
                                     const typename tnlParallelReductionLpNorm< double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLpNorm< double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLpNorm< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionLpNorm< long double, int > >
                                   ( const tnlParallelReductionLpNorm< long double, int>& operation,
                                     const typename tnlParallelReductionLpNorm< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionLpNorm< long double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLpNorm< long double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLpNorm< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool reductionOnCudaDevice< tnlParallelReductionLpNorm< char, long int > >
                                   ( const tnlParallelReductionLpNorm< char, long int >& operation,
                                     const typename tnlParallelReductionLpNorm< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionLpNorm< char, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLpNorm< char, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLpNorm< char, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionLpNorm< int, long int > >
                                   ( const tnlParallelReductionLpNorm< int, long int >& operation,
                                     const typename tnlParallelReductionLpNorm< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionLpNorm< int, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLpNorm< int, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLpNorm< int, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionLpNorm< float, long int > >
                                   ( const tnlParallelReductionLpNorm< float, long int >& operation,
                                     const typename tnlParallelReductionLpNorm< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionLpNorm< float, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLpNorm< float, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLpNorm< float, long int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionLpNorm< double, long int > >
                                   ( const tnlParallelReductionLpNorm< double, long int>& operation,
                                     const typename tnlParallelReductionLpNorm< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionLpNorm< double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLpNorm< double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLpNorm< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionLpNorm< long double, long int > >
                                   ( const tnlParallelReductionLpNorm< long double, long int>& operation,
                                     const typename tnlParallelReductionLpNorm< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionLpNorm< long double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionLpNorm< long double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionLpNorm< long double, long int> :: ResultType& result );
#endif
#endif

/****
 * Equalities
 */
extern template bool reductionOnCudaDevice< tnlParallelReductionEqualities< char, int > >
                                   ( const tnlParallelReductionEqualities< char, int >& operation,
                                     const typename tnlParallelReductionEqualities< char, int > :: IndexType size,
                                     const typename tnlParallelReductionEqualities< char, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionEqualities< char, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionEqualities< char, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionEqualities< int, int > >
                                   ( const tnlParallelReductionEqualities< int, int >& operation,
                                     const typename tnlParallelReductionEqualities< int, int > :: IndexType size,
                                     const typename tnlParallelReductionEqualities< int, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionEqualities< int, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionEqualities< int, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionEqualities< float, int > >
                                   ( const tnlParallelReductionEqualities< float, int >& operation,
                                     const typename tnlParallelReductionEqualities< float, int > :: IndexType size,
                                     const typename tnlParallelReductionEqualities< float, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionEqualities< float, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionEqualities< float, int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionEqualities< double, int > >
                                   ( const tnlParallelReductionEqualities< double, int>& operation,
                                     const typename tnlParallelReductionEqualities< double, int > :: IndexType size,
                                     const typename tnlParallelReductionEqualities< double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionEqualities< double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionEqualities< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionEqualities< long double, int > >
                                   ( const tnlParallelReductionEqualities< long double, int>& operation,
                                     const typename tnlParallelReductionEqualities< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionEqualities< long double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionEqualities< long double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionEqualities< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool reductionOnCudaDevice< tnlParallelReductionEqualities< char, long int > >
                                   ( const tnlParallelReductionEqualities< char, long int >& operation,
                                     const typename tnlParallelReductionEqualities< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionEqualities< char, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionEqualities< char, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionEqualities< char, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionEqualities< int, long int > >
                                   ( const tnlParallelReductionEqualities< int, long int >& operation,
                                     const typename tnlParallelReductionEqualities< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionEqualities< int, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionEqualities< int, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionEqualities< int, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionEqualities< float, long int > >
                                   ( const tnlParallelReductionEqualities< float, long int >& operation,
                                     const typename tnlParallelReductionEqualities< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionEqualities< float, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionEqualities< float, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionEqualities< float, long int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionEqualities< double, long int > >
                                   ( const tnlParallelReductionEqualities< double, long int>& operation,
                                     const typename tnlParallelReductionEqualities< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionEqualities< double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionEqualities< double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionEqualities< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionEqualities< long double, long int > >
                                   ( const tnlParallelReductionEqualities< long double, long int>& operation,
                                     const typename tnlParallelReductionEqualities< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionEqualities< long double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionEqualities< long double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionEqualities< long double, long int> :: ResultType& result );
#endif
#endif

/****
 * Inequalities
 */
extern template bool reductionOnCudaDevice< tnlParallelReductionInequalities< char, int > >
                                   ( const tnlParallelReductionInequalities< char, int >& operation,
                                     const typename tnlParallelReductionInequalities< char, int > :: IndexType size,
                                     const typename tnlParallelReductionInequalities< char, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionInequalities< char, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionInequalities< char, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionInequalities< int, int > >
                                   ( const tnlParallelReductionInequalities< int, int >& operation,
                                     const typename tnlParallelReductionInequalities< int, int > :: IndexType size,
                                     const typename tnlParallelReductionInequalities< int, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionInequalities< int, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionInequalities< int, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionInequalities< float, int > >
                                   ( const tnlParallelReductionInequalities< float, int >& operation,
                                     const typename tnlParallelReductionInequalities< float, int > :: IndexType size,
                                     const typename tnlParallelReductionInequalities< float, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionInequalities< float, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionInequalities< float, int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionInequalities< double, int > >
                                   ( const tnlParallelReductionInequalities< double, int>& operation,
                                     const typename tnlParallelReductionInequalities< double, int > :: IndexType size,
                                     const typename tnlParallelReductionInequalities< double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionInequalities< double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionInequalities< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionInequalities< long double, int > >
                                   ( const tnlParallelReductionInequalities< long double, int>& operation,
                                     const typename tnlParallelReductionInequalities< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionInequalities< long double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionInequalities< long double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionInequalities< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool reductionOnCudaDevice< tnlParallelReductionInequalities< char, long int > >
                                   ( const tnlParallelReductionInequalities< char, long int >& operation,
                                     const typename tnlParallelReductionInequalities< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionInequalities< char, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionInequalities< char, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionInequalities< char, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionInequalities< int, long int > >
                                   ( const tnlParallelReductionInequalities< int, long int >& operation,
                                     const typename tnlParallelReductionInequalities< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionInequalities< int, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionInequalities< int, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionInequalities< int, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionInequalities< float, long int > >
                                   ( const tnlParallelReductionInequalities< float, long int >& operation,
                                     const typename tnlParallelReductionInequalities< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionInequalities< float, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionInequalities< float, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionInequalities< float, long int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionInequalities< double, long int > >
                                   ( const tnlParallelReductionInequalities< double, long int>& operation,
                                     const typename tnlParallelReductionInequalities< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionInequalities< double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionInequalities< double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionInequalities< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionInequalities< long double, long int > >
                                   ( const tnlParallelReductionInequalities< long double, long int>& operation,
                                     const typename tnlParallelReductionInequalities< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionInequalities< long double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionInequalities< long double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionInequalities< long double, long int> :: ResultType& result );
#endif
#endif

/****
 * ScalarProduct
 */
extern template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< char, int > >
                                   ( const tnlParallelReductionScalarProduct< char, int >& operation,
                                     const typename tnlParallelReductionScalarProduct< char, int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< char, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< char, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< char, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< int, int > >
                                   ( const tnlParallelReductionScalarProduct< int, int >& operation,
                                     const typename tnlParallelReductionScalarProduct< int, int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< int, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< int, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< int, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< float, int > >
                                   ( const tnlParallelReductionScalarProduct< float, int >& operation,
                                     const typename tnlParallelReductionScalarProduct< float, int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< float, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< float, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< float, int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< double, int > >
                                   ( const tnlParallelReductionScalarProduct< double, int>& operation,
                                     const typename tnlParallelReductionScalarProduct< double, int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< long double, int > >
                                   ( const tnlParallelReductionScalarProduct< long double, int>& operation,
                                     const typename tnlParallelReductionScalarProduct< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< long double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< long double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< char, long int > >
                                   ( const tnlParallelReductionScalarProduct< char, long int >& operation,
                                     const typename tnlParallelReductionScalarProduct< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< char, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< char, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< char, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< int, long int > >
                                   ( const tnlParallelReductionScalarProduct< int, long int >& operation,
                                     const typename tnlParallelReductionScalarProduct< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< int, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< int, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< int, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< float, long int > >
                                   ( const tnlParallelReductionScalarProduct< float, long int >& operation,
                                     const typename tnlParallelReductionScalarProduct< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< float, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< float, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< float, long int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< double, long int > >
                                   ( const tnlParallelReductionScalarProduct< double, long int>& operation,
                                     const typename tnlParallelReductionScalarProduct< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionScalarProduct< long double, long int > >
                                   ( const tnlParallelReductionScalarProduct< long double, long int>& operation,
                                     const typename tnlParallelReductionScalarProduct< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionScalarProduct< long double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionScalarProduct< long double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionScalarProduct< long double, long int> :: ResultType& result );
#endif
#endif

/****
 * Diff sum
 */

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< char, int > >
                                   ( const tnlParallelReductionDiffSum< char, int >& operation,
                                     const typename tnlParallelReductionDiffSum< char, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< char, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< char, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffSum< char, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< int, int > >
                                   ( const tnlParallelReductionDiffSum< int, int >& operation,
                                     const typename tnlParallelReductionDiffSum< int, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< int, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< int, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffSum< int, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< float, int > >
                                   ( const tnlParallelReductionDiffSum< float, int >& operation,
                                     const typename tnlParallelReductionDiffSum< float, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< float, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< float, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffSum< float, int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< double, int > >
                                   ( const tnlParallelReductionDiffSum< double, int>& operation,
                                     const typename tnlParallelReductionDiffSum< double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffSum< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< long double, int > >
                                   ( const tnlParallelReductionDiffSum< long double, int>& operation,
                                     const typename tnlParallelReductionDiffSum< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< long double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< long double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffSum< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< char, long int > >
                                   ( const tnlParallelReductionDiffSum< char, long int >& operation,
                                     const typename tnlParallelReductionDiffSum< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< char, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< char, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffSum< char, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< int, long int > >
                                   ( const tnlParallelReductionDiffSum< int, long int >& operation,
                                     const typename tnlParallelReductionDiffSum< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< int, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< int, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffSum< int, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< float, long int > >
                                   ( const tnlParallelReductionDiffSum< float, long int >& operation,
                                     const typename tnlParallelReductionDiffSum< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< float, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< float, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffSum< float, long int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< double, long int > >
                                   ( const tnlParallelReductionDiffSum< double, long int>& operation,
                                     const typename tnlParallelReductionDiffSum< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffSum< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffSum< long double, long int > >
                                   ( const tnlParallelReductionDiffSum< long double, long int>& operation,
                                     const typename tnlParallelReductionDiffSum< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffSum< long double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffSum< long double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffSum< long double, long int> :: ResultType& result );
#endif
#endif

/****
 * Diff min
 */

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffMin< char, int > >
                                   ( const tnlParallelReductionDiffMin< char, int >& operation,
                                     const typename tnlParallelReductionDiffMin< char, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMin< char, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffMin< char, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffMin< char, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffMin< int, int > >
                                   ( const tnlParallelReductionDiffMin< int, int >& operation,
                                     const typename tnlParallelReductionDiffMin< int, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMin< int, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffMin< int, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffMin< int, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffMin< float, int > >
                                   ( const tnlParallelReductionDiffMin< float, int >& operation,
                                     const typename tnlParallelReductionDiffMin< float, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMin< float, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffMin< float, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffMin< float, int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffMin< double, int > >
                                   ( const tnlParallelReductionDiffMin< double, int>& operation,
                                     const typename tnlParallelReductionDiffMin< double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMin< double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffMin< double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffMin< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffMin< long double, int > >
                                   ( const tnlParallelReductionDiffMin< long double, int>& operation,
                                     const typename tnlParallelReductionDiffMin< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMin< long double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffMin< long double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffMin< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffMin< char, long int > >
                                   ( const tnlParallelReductionDiffMin< char, long int >& operation,
                                     const typename tnlParallelReductionDiffMin< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMin< char, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffMin< char, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffMin< char, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffMin< int, long int > >
                                   ( const tnlParallelReductionDiffMin< int, long int >& operation,
                                     const typename tnlParallelReductionDiffMin< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMin< int, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffMin< int, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffMin< int, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffMin< float, long int > >
                                   ( const tnlParallelReductionDiffMin< float, long int >& operation,
                                     const typename tnlParallelReductionDiffMin< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMin< float, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffMin< float, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffMin< float, long int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffMin< double, long int > >
                                   ( const tnlParallelReductionDiffMin< double, long int>& operation,
                                     const typename tnlParallelReductionDiffMin< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMin< double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffMin< double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffMin< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffMin< long double, long int > >
                                   ( const tnlParallelReductionDiffMin< long double, long int>& operation,
                                     const typename tnlParallelReductionDiffMin< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMin< long double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffMin< long double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffMin< long double, long int> :: ResultType& result );
#endif
#endif

/****
 * Diff max
 */

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffMax< char, int > >
                                   ( const tnlParallelReductionDiffMax< char, int >& operation,
                                     const typename tnlParallelReductionDiffMax< char, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMax< char, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffMax< char, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffMax< char, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffMax< int, int > >
                                   ( const tnlParallelReductionDiffMax< int, int >& operation,
                                     const typename tnlParallelReductionDiffMax< int, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMax< int, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffMax< int, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffMax< int, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffMax< float, int > >
                                   ( const tnlParallelReductionDiffMax< float, int >& operation,
                                     const typename tnlParallelReductionDiffMax< float, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMax< float, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffMax< float, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffMax< float, int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffMax< double, int > >
                                   ( const tnlParallelReductionDiffMax< double, int>& operation,
                                     const typename tnlParallelReductionDiffMax< double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMax< double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffMax< double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffMax< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffMax< long double, int > >
                                   ( const tnlParallelReductionDiffMax< long double, int>& operation,
                                     const typename tnlParallelReductionDiffMax< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMax< long double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffMax< long double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffMax< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffMax< char, long int > >
                                   ( const tnlParallelReductionDiffMax< char, long int >& operation,
                                     const typename tnlParallelReductionDiffMax< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMax< char, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffMax< char, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffMax< char, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffMax< int, long int > >
                                   ( const tnlParallelReductionDiffMax< int, long int >& operation,
                                     const typename tnlParallelReductionDiffMax< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMax< int, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffMax< int, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffMax< int, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffMax< float, long int > >
                                   ( const tnlParallelReductionDiffMax< float, long int >& operation,
                                     const typename tnlParallelReductionDiffMax< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMax< float, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffMax< float, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffMax< float, long int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffMax< double, long int > >
                                   ( const tnlParallelReductionDiffMax< double, long int>& operation,
                                     const typename tnlParallelReductionDiffMax< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMax< double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffMax< double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffMax< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffMax< long double, long int > >
                                   ( const tnlParallelReductionDiffMax< long double, long int>& operation,
                                     const typename tnlParallelReductionDiffMax< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffMax< long double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffMax< long double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffMax< long double, long int> :: ResultType& result );
#endif
#endif

/****
 * Diff abs sum
 */

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< char, int > >
                                   ( const tnlParallelReductionDiffAbsSum< char, int >& operation,
                                     const typename tnlParallelReductionDiffAbsSum< char, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< char, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< char, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< char, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< int, int > >
                                   ( const tnlParallelReductionDiffAbsSum< int, int >& operation,
                                     const typename tnlParallelReductionDiffAbsSum< int, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< int, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< int, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< int, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< float, int > >
                                   ( const tnlParallelReductionDiffAbsSum< float, int >& operation,
                                     const typename tnlParallelReductionDiffAbsSum< float, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< float, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< float, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< float, int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< double, int > >
                                   ( const tnlParallelReductionDiffAbsSum< double, int>& operation,
                                     const typename tnlParallelReductionDiffAbsSum< double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< long double, int > >
                                   ( const tnlParallelReductionDiffAbsSum< long double, int>& operation,
                                     const typename tnlParallelReductionDiffAbsSum< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< long double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< long double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< char, long int > >
                                   ( const tnlParallelReductionDiffAbsSum< char, long int >& operation,
                                     const typename tnlParallelReductionDiffAbsSum< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< char, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< char, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< char, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< int, long int > >
                                   ( const tnlParallelReductionDiffAbsSum< int, long int >& operation,
                                     const typename tnlParallelReductionDiffAbsSum< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< int, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< int, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< int, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< float, long int > >
                                   ( const tnlParallelReductionDiffAbsSum< float, long int >& operation,
                                     const typename tnlParallelReductionDiffAbsSum< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< float, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< float, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< float, long int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< double, long int > >
                                   ( const tnlParallelReductionDiffAbsSum< double, long int>& operation,
                                     const typename tnlParallelReductionDiffAbsSum< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsSum< long double, long int > >
                                   ( const tnlParallelReductionDiffAbsSum< long double, long int>& operation,
                                     const typename tnlParallelReductionDiffAbsSum< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsSum< long double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsSum< long double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsSum< long double, long int> :: ResultType& result );
#endif
#endif

/****
 * Diff abs min
 */

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMin< char, int > >
                                   ( const tnlParallelReductionDiffAbsMin< char, int >& operation,
                                     const typename tnlParallelReductionDiffAbsMin< char, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMin< char, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMin< char, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMin< char, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMin< int, int > >
                                   ( const tnlParallelReductionDiffAbsMin< int, int >& operation,
                                     const typename tnlParallelReductionDiffAbsMin< int, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMin< int, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMin< int, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMin< int, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMin< float, int > >
                                   ( const tnlParallelReductionDiffAbsMin< float, int >& operation,
                                     const typename tnlParallelReductionDiffAbsMin< float, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMin< float, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMin< float, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMin< float, int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMin< double, int > >
                                   ( const tnlParallelReductionDiffAbsMin< double, int>& operation,
                                     const typename tnlParallelReductionDiffAbsMin< double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMin< double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMin< double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMin< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMin< long double, int > >
                                   ( const tnlParallelReductionDiffAbsMin< long double, int>& operation,
                                     const typename tnlParallelReductionDiffAbsMin< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMin< long double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMin< long double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMin< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMin< char, long int > >
                                   ( const tnlParallelReductionDiffAbsMin< char, long int >& operation,
                                     const typename tnlParallelReductionDiffAbsMin< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMin< char, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMin< char, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMin< char, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMin< int, long int > >
                                   ( const tnlParallelReductionDiffAbsMin< int, long int >& operation,
                                     const typename tnlParallelReductionDiffAbsMin< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMin< int, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMin< int, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMin< int, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMin< float, long int > >
                                   ( const tnlParallelReductionDiffAbsMin< float, long int >& operation,
                                     const typename tnlParallelReductionDiffAbsMin< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMin< float, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMin< float, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMin< float, long int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMin< double, long int > >
                                   ( const tnlParallelReductionDiffAbsMin< double, long int>& operation,
                                     const typename tnlParallelReductionDiffAbsMin< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMin< double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMin< double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMin< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMin< long double, long int > >
                                   ( const tnlParallelReductionDiffAbsMin< long double, long int>& operation,
                                     const typename tnlParallelReductionDiffAbsMin< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMin< long double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMin< long double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMin< long double, long int> :: ResultType& result );
#endif
#endif

/****
 * Diff abs max
 */

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMax< char, int > >
                                   ( const tnlParallelReductionDiffAbsMax< char, int >& operation,
                                     const typename tnlParallelReductionDiffAbsMax< char, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMax< char, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMax< char, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMax< char, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMax< int, int > >
                                   ( const tnlParallelReductionDiffAbsMax< int, int >& operation,
                                     const typename tnlParallelReductionDiffAbsMax< int, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMax< int, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMax< int, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMax< int, int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMax< float, int > >
                                   ( const tnlParallelReductionDiffAbsMax< float, int >& operation,
                                     const typename tnlParallelReductionDiffAbsMax< float, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMax< float, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMax< float, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMax< float, int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMax< double, int > >
                                   ( const tnlParallelReductionDiffAbsMax< double, int>& operation,
                                     const typename tnlParallelReductionDiffAbsMax< double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMax< double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMax< double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMax< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMax< long double, int > >
                                   ( const tnlParallelReductionDiffAbsMax< long double, int>& operation,
                                     const typename tnlParallelReductionDiffAbsMax< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMax< long double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMax< long double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMax< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMax< char, long int > >
                                   ( const tnlParallelReductionDiffAbsMax< char, long int >& operation,
                                     const typename tnlParallelReductionDiffAbsMax< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMax< char, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMax< char, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMax< char, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMax< int, long int > >
                                   ( const tnlParallelReductionDiffAbsMax< int, long int >& operation,
                                     const typename tnlParallelReductionDiffAbsMax< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMax< int, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMax< int, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMax< int, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMax< float, long int > >
                                   ( const tnlParallelReductionDiffAbsMax< float, long int >& operation,
                                     const typename tnlParallelReductionDiffAbsMax< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMax< float, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMax< float, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMax< float, long int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMax< double, long int > >
                                   ( const tnlParallelReductionDiffAbsMax< double, long int>& operation,
                                     const typename tnlParallelReductionDiffAbsMax< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMax< double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMax< double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMax< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffAbsMax< long double, long int > >
                                   ( const tnlParallelReductionDiffAbsMax< long double, long int>& operation,
                                     const typename tnlParallelReductionDiffAbsMax< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffAbsMax< long double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffAbsMax< long double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffAbsMax< long double, long int> :: ResultType& result );
#endif
#endif


/****
 * Diff Lp Norm
 */
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffLpNorm< float, int > >
                                   ( const tnlParallelReductionDiffLpNorm< float, int >& operation,
                                     const typename tnlParallelReductionDiffLpNorm< float, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffLpNorm< float, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffLpNorm< float, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffLpNorm< float, int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffLpNorm< double, int > >
                                   ( const tnlParallelReductionDiffLpNorm< double, int>& operation,
                                     const typename tnlParallelReductionDiffLpNorm< double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffLpNorm< double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffLpNorm< double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffLpNorm< double, int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffLpNorm< long double, int > >
                                   ( const tnlParallelReductionDiffLpNorm< long double, int>& operation,
                                     const typename tnlParallelReductionDiffLpNorm< long double, int > :: IndexType size,
                                     const typename tnlParallelReductionDiffLpNorm< long double, int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffLpNorm< long double, int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffLpNorm< long double, int> :: ResultType& result );
#endif

#ifdef INSTANTIATE_LONG_INT
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffLpNorm< char, long int > >
                                   ( const tnlParallelReductionDiffLpNorm< char, long int >& operation,
                                     const typename tnlParallelReductionDiffLpNorm< char, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffLpNorm< char, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffLpNorm< char, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffLpNorm< char, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffLpNorm< int, long int > >
                                   ( const tnlParallelReductionDiffLpNorm< int, long int >& operation,
                                     const typename tnlParallelReductionDiffLpNorm< int, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffLpNorm< int, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffLpNorm< int, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffLpNorm< int, long int > :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffLpNorm< float, long int > >
                                   ( const tnlParallelReductionDiffLpNorm< float, long int >& operation,
                                     const typename tnlParallelReductionDiffLpNorm< float, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffLpNorm< float, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffLpNorm< float, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffLpNorm< float, long int> :: ResultType& result );

extern template bool reductionOnCudaDevice< tnlParallelReductionDiffLpNorm< double, long int > >
                                   ( const tnlParallelReductionDiffLpNorm< double, long int>& operation,
                                     const typename tnlParallelReductionDiffLpNorm< double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffLpNorm< double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffLpNorm< double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffLpNorm< double, long int> :: ResultType& result );

#ifdef INSTANTIATE_LONG_DOUBLE
extern template bool reductionOnCudaDevice< tnlParallelReductionDiffLpNorm< long double, long int > >
                                   ( const tnlParallelReductionDiffLpNorm< long double, long int>& operation,
                                     const typename tnlParallelReductionDiffLpNorm< long double, long int > :: IndexType size,
                                     const typename tnlParallelReductionDiffLpNorm< long double, long int > :: DataType1* deviceInput1,
                                     const typename tnlParallelReductionDiffLpNorm< long double, long int > :: DataType2* deviceInput2,
                                     typename tnlParallelReductionDiffLpNorm< long double, long int> :: ResultType& result );
#endif
#endif

#endif /* TEMPLATE_EXPLICIT_INSTANTIATION */

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
