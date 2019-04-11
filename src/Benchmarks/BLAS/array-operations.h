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

#include "../Benchmarks.h"

#include <TNL/Containers/Array.h>

namespace TNL {
namespace Benchmarks {

template< typename Real = double,
          typename Index = int >
bool
benchmarkArrayOperations( Benchmark & benchmark,
                          const long & size )
{
   typedef Containers::Array< Real, Devices::Host, Index > HostArray;
   typedef Containers::Array< Real, Devices::Cuda, Index > CudaArray;
   using namespace std;

   double datasetSize = (double) size * sizeof( Real ) / oneGB;

   HostArray hostArray, hostArray2;
   CudaArray deviceArray, deviceArray2;
   hostArray.setSize( size );
   hostArray2.setSize( size );
#ifdef HAVE_CUDA
   deviceArray.setSize( size );
   deviceArray2.setSize( size );
#endif

   Real resultHost, resultDevice;


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


   auto compareHost = [&]() {
      resultHost = (int) ( hostArray == hostArray2 );
   };
   auto compareCuda = [&]() {
      resultDevice = (int) ( deviceArray == deviceArray2 );
   };
   benchmark.setOperation( "comparison (operator==)", 2 * datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU", compareHost );
#ifdef HAVE_CUDA
   benchmark.time< Devices::Cuda >( reset1, "GPU", compareCuda );
#endif


   auto copyAssignHostHost = [&]() {
      hostArray = hostArray2;
   };
   auto copyAssignCudaCuda = [&]() {
      deviceArray = deviceArray2;
   };
   benchmark.setOperation( "copy (operator=)", 2 * datasetSize );
   // copyBasetime is used later inside HAVE_CUDA guard, so the compiler will
   // complain when compiling without CUDA
   const double copyBasetime = benchmark.time< Devices::Host >( reset1, "CPU", copyAssignHostHost );
#ifdef HAVE_CUDA
   benchmark.time< Devices::Cuda >( reset1, "GPU", copyAssignCudaCuda );
#endif


   auto copyAssignHostCuda = [&]() {
      deviceArray = hostArray;
   };
   auto copyAssignCudaHost = [&]() {
      hostArray = deviceArray;
   };
#ifdef HAVE_CUDA
   benchmark.setOperation( "copy (operator=)", datasetSize, copyBasetime );
   benchmark.time< Devices::Cuda >( reset1, "CPU->GPU", copyAssignHostCuda );
   benchmark.time< Devices::Cuda >( reset1, "GPU->CPU", copyAssignCudaHost );
#endif


   auto setValueHost = [&]() {
      hostArray.setValue( 3.0 );
   };
   auto setValueCuda = [&]() {
      deviceArray.setValue( 3.0 );
   };
   benchmark.setOperation( "setValue", datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU", setValueHost );
#ifdef HAVE_CUDA
   benchmark.time< Devices::Cuda >( reset1, "GPU", setValueCuda );
#endif


   auto setSizeHost = [&]() {
      hostArray.setSize( size );
   };
   auto setSizeCuda = [&]() {
      deviceArray.setSize( size );
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
   benchmark.time< Devices::Cuda >( resetSize1, "GPU", setSizeCuda );
#endif


   auto resetSizeHost = [&]() {
      hostArray.reset();
   };
   auto resetSizeCuda = [&]() {
      deviceArray.reset();
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
   benchmark.time< Devices::Cuda >( setSize1, "GPU", resetSizeCuda );
#endif

   return true;
}

} // namespace Benchmarks
} // namespace TNL
