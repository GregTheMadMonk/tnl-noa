/***************************************************************************
                          array-operations.h  -  description
                             -------------------
    begin                : Dec 30, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <cstring>

#include "../Benchmarks.h"

#include <TNL/Containers/Array.h>

namespace TNL {
namespace Benchmarks {

template< typename Real = double,
          typename Index = int,
          template<typename> class HostAllocator = Allocators::Default< Devices::Host >::Allocator,
          template<typename> class CudaAllocator = Allocators::Default< Devices::Cuda >::Allocator >
void
benchmarkArrayOperations( Benchmark & benchmark,
                          const long & size )
{
   using HostArray = Containers::Array< Real, Devices::Host, Index, HostAllocator< Real > >;
   using CudaArray = Containers::Array< Real, Devices::Cuda, Index, CudaAllocator< Real > >;

   double datasetSize = (double) size * sizeof( Real ) / oneGB;

   HostArray hostArray, hostArray2;
   CudaArray deviceArray, deviceArray2;
   hostArray.setSize( size );
   hostArray2.setSize( size );
#ifdef HAVE_CUDA
   deviceArray.setSize( size );
   deviceArray2.setSize( size );
#endif

   Real resultHost;


   // reset functions
   auto reset1 = [&]() {
      hostArray.setValue( 1.0 );
#ifdef HAVE_CUDA
      deviceArray.setValue( 1.0 );
#endif
   };
   auto reset2 = [&]() {
      hostArray2.setValue( 1.0 );
#ifdef HAVE_CUDA
      deviceArray2.setValue( 1.0 );
#endif
   };
   auto reset12 = [&]() {
      reset1();
      reset2();
   };


   reset12();


   if( std::is_fundamental< Real >::value ) {
      // std::memcmp
      auto compareHost = [&]() {
         if( std::memcmp( hostArray.getData(), hostArray2.getData(), hostArray.getSize() * sizeof(Real) ) == 0 )
            resultHost = true;
         else
            resultHost = false;
      };
      benchmark.setOperation( "comparison (memcmp)", 2 * datasetSize );
      benchmark.time< Devices::Host >( reset12, "CPU", compareHost );

      // std::memcpy and cudaMemcpy
      auto copyHost = [&]() {
         std::memcpy( hostArray.getData(), hostArray2.getData(), hostArray.getSize() * sizeof(Real) );
      };
      benchmark.setOperation( "copy (memcpy)", 2 * datasetSize );
      benchmark.time< Devices::Host >( reset12, "CPU", copyHost );
#ifdef HAVE_CUDA
      auto copyCuda = [&]() {
         cudaMemcpy( deviceArray.getData(),
                     deviceArray2.getData(),
                     deviceArray.getSize() * sizeof(Real),
                     cudaMemcpyDeviceToDevice );
         TNL_CHECK_CUDA_DEVICE;
      };
      benchmark.time< Devices::Cuda >( reset12, "GPU", copyCuda );
#endif
   }


   auto compareHost = [&]() {
      resultHost = (int) ( hostArray == hostArray2 );
   };
   benchmark.setOperation( "comparison (operator==)", 2 * datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU", compareHost );
#ifdef HAVE_CUDA
   Real resultDevice;
   auto compareCuda = [&]() {
      resultDevice = (int) ( deviceArray == deviceArray2 );
   };
   benchmark.time< Devices::Cuda >( reset1, "GPU", compareCuda );
#endif


   auto copyAssignHostHost = [&]() {
      hostArray = hostArray2;
   };
   benchmark.setOperation( "copy (operator=)", 2 * datasetSize );
   // copyBasetime is used later inside HAVE_CUDA guard, so the compiler will
   // complain when compiling without CUDA
   const double copyBasetime = benchmark.time< Devices::Host >( reset1, "CPU", copyAssignHostHost );
   (void)copyBasetime;  // ignore unused variable
#ifdef HAVE_CUDA
   auto copyAssignCudaCuda = [&]() {
      deviceArray = deviceArray2;
   };
   benchmark.time< Devices::Cuda >( reset1, "GPU", copyAssignCudaCuda );
#endif


#ifdef HAVE_CUDA
   auto copyAssignHostCuda = [&]() {
      deviceArray = hostArray;
   };
   auto copyAssignCudaHost = [&]() {
      hostArray = deviceArray;
   };
   benchmark.setOperation( "copy (operator=)", datasetSize, copyBasetime );
   benchmark.time< Devices::Cuda >( reset1, "CPU->GPU", copyAssignHostCuda );
   benchmark.time< Devices::Cuda >( reset1, "GPU->CPU", copyAssignCudaHost );
#endif


   auto setValueHost = [&]() {
      hostArray.setValue( 3.0 );
   };
   benchmark.setOperation( "setValue", datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU", setValueHost );
#ifdef HAVE_CUDA
   auto setValueCuda = [&]() {
      deviceArray.setValue( 3.0 );
   };
   benchmark.time< Devices::Cuda >( reset1, "GPU", setValueCuda );
#endif


   auto setSizeHost = [&]() {
      hostArray.setSize( size );
   };
   auto resetSize1 = [&]() {
      hostArray.reset();
#ifdef HAVE_CUDA
      deviceArray.reset();
#endif
   };
   benchmark.setOperation( "allocation (setSize)", datasetSize );
   benchmark.time< Devices::Host >( resetSize1, "CPU", setSizeHost );
#ifdef HAVE_CUDA
   auto setSizeCuda = [&]() {
      deviceArray.setSize( size );
   };
   benchmark.time< Devices::Cuda >( resetSize1, "GPU", setSizeCuda );
#endif


   auto resetSizeHost = [&]() {
      hostArray.reset();
   };
   auto setSize1 = [&]() {
      hostArray.setSize( size );
#ifdef HAVE_CUDA
      deviceArray.setSize( size );
#endif
   };
   benchmark.setOperation( "deallocation (reset)", datasetSize );
   benchmark.time< Devices::Host >( setSize1, "CPU", resetSizeHost );
#ifdef HAVE_CUDA
   auto resetSizeCuda = [&]() {
      deviceArray.reset();
   };
   benchmark.time< Devices::Cuda >( setSize1, "GPU", resetSizeCuda );
#endif
}

} // namespace Benchmarks
} // namespace TNL
