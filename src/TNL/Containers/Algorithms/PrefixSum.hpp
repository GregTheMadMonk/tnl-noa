/***************************************************************************
                          PrefixSum.hpp  -  description
                             -------------------
    begin                : Mar 24, 2013
    copyright            : (C) 2013 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include "PrefixSum.h"

//#define CUDA_REDUCTION_PROFILING

#include <TNL/Assert.h>
#include <TNL/Exceptions/CudaSupportMissing.h>
#include <TNL/Containers/Algorithms/PrefixSumOperations.h>
#include <TNL/Containers/Algorithms/ArrayOperations.h>
#include <TNL/Containers/Algorithms/CudaPrefixSumKernel.h>

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
static constexpr int PrefixSum_minGpuDataSize = 256;//65536; //16384;//1024;//256;

////
// PrefixSum on host
template< typename Index,
          typename Result,
          typename PrefixSumOperation,
          typename VolatilePrefixSumOperation >
Result
PrefixSum< Devices::Host >::
inclusive( const Index size,
           PrefixSumOperation& reduction,
           VolatilePrefixSumOperation& volatilePrefixSum,
           const Result& zero )
{
   using IndexType = Index;
   using ResultType = Result;

}

template< typename Index,
          typename Result,
          typename PrefixSumOperation,
          typename VolatilePrefixSumOperation >
Result
PrefixSum< Devices::Host >::
exclusive( const Index size,
           PrefixSumOperation& reduction,
           VolatilePrefixSumOperation& volatilePrefixSum,
           const Result& zero )
{
   using IndexType = Index;
   using ResultType = Result;

}




template< typename Index,
          typename Result,
          typename PrefixSumOperation,
          typename VolatilePrefixSumOperation >
Result
PrefixSum< Devices::Cuda >::
   reduce( const Index size,
           PrefixSumOperation& reduction,
           VolatilePrefixSumOperation& volatilePrefixSum,
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

   CudaPrefixSumKernelLauncher< IndexType, ResultType > reductionLauncher( size );

   /****
    * Reduce the data on the CUDA device.
    */
   ResultType* deviceAux1( 0 );
   IndexType reducedSize = reductionLauncher.start(
      reduction,
      volatilePrefixSum,
      dataFetcher,
      zero,
      deviceAux1 );
   #ifdef CUDA_REDUCTION_PROFILING
      timer.stop();
      std::cout << "   PrefixSum on GPU to size " << reducedSize << " took " << timer.getRealTime() << " sec. " << std::endl;
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
      const ResultType result = PrefixSum< Devices::Host >::reduce( reducedSize, reduction, volatilePrefixSum, fetch, zero );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   PrefixSum of small data set on CPU took " << timer.getRealTime() << " sec. " << std::endl;
      #endif
      return result;
   }
   else {
      /***
       * Data can't be safely reduced on host, so continue with the reduction on the CUDA device.
       */
      auto result = reductionLauncher.finish( reduction, volatilePrefixSum, zero );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   PrefixSum of small data set on GPU took " << timer.getRealTime() << " sec. " << std::endl;
         timer.reset();
         timer.start();
      #endif

      return result;
   }
#else
   throw Exceptions::CudaSupportMissing();
#endif
};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
