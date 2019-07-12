/***************************************************************************
                          vector-operations.h  -  description
                             -------------------
    begin                : Dec 30, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <stdlib.h> // srand48

#include "../Benchmarks.h"

#include <TNL/Containers/Vector.h>
#include "CommonVectorOperations.h"

#ifdef HAVE_BLAS
#include "blasWrappers.h"
#endif

#ifdef HAVE_CUDA
#include "cublasWrappers.h"
#endif

namespace TNL {
namespace Benchmarks {

template< typename Real = double,
          typename Index = int >
bool
benchmarkVectorOperations( Benchmark & benchmark,
                           const long & size )
{
   using HostVector = Containers::Vector< Real, Devices::Host, Index >;
   using CudaVector =  Containers::Vector< Real, Devices::Cuda, Index >;
   using HostView = Containers::VectorView< Real, Devices::Host, Index >;
   using CudaView =  Containers::VectorView< Real, Devices::Cuda, Index >;

   using namespace std;

   double datasetSize = (double) size * sizeof( Real ) / oneGB;

   HostVector hostVector( size ), hostVector2( size ), hostVector3( size ), hostVector4( size );
   CudaVector deviceVector, deviceVector2, deviceVector3, deviceVector4;
#ifdef HAVE_CUDA
   deviceVector.setSize( size );
   deviceVector2.setSize( size );
   deviceVector3.setSize( size );
   deviceVector4.setSize( size );
#endif

   HostView hostView( hostVector ), hostView2( hostVector2 ), hostView3( hostVector3 ), hostView4( hostVector4 );
   CudaView deviceView( deviceVector ), deviceView2( deviceVector2 ), deviceView3( deviceVector3 ), deviceView4( deviceVector4 );

   Real resultHost, resultDevice;

#ifdef HAVE_CUDA
   cublasHandle_t cublasHandle;
   cublasCreate( &cublasHandle );
#endif


   // reset functions
   // (Make sure to always use some in benchmarks, even if it's not necessary
   // to assure correct result - it helps to clear cache and avoid optimizations
   // of the benchmark loop.)
   auto reset1 = [&]() {
      hostVector.setValue( 1.0 );
#ifdef HAVE_CUDA
      deviceVector.setValue( 1.0 );
#endif
      // A relatively harmless call to keep the compiler from realizing we
      // don't actually do any useful work with the result of the reduction.
      srand48(resultHost);
      resultHost = resultDevice = 0.0;
   };
   auto reset2 = [&]() {
      hostVector2.setValue( 1.0 );
#ifdef HAVE_CUDA
      deviceVector2.setValue( 1.0 );
#endif
   };
   auto reset3 = [&]() {
      hostVector3.setValue( 1.0 );
#ifdef HAVE_CUDA
      deviceVector3.setValue( 1.0 );
#endif
   };
   auto reset4 = [&]() {
      hostVector4.setValue( 1.0 );
#ifdef HAVE_CUDA
      deviceVector4.setValue( 1.0 );
#endif
   };


   auto resetAll = [&]() {
      reset1();
      reset2();
      reset3();
      reset4();
   };

   resetAll();

   ////
   // Max
   auto maxHost = [&]() {
      resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorMax( hostVector );
   };
   auto maxCuda = [&]() {
      resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorMax( deviceVector );
   };
   auto maxHostET = [&]() {
      resultHost = max( hostView );
   };
   auto maxCudaET = [&]() {
      resultDevice = max( deviceView );
   };

   benchmark.setOperation( "max", datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU", maxHost );
   benchmark.time< Devices::Host >( reset1, "CPU ET", maxHostET );
#ifdef HAVE_CUDA
   benchmark.time< Devices::Cuda >( reset1, "GPU", maxCuda );
   benchmark.time< Devices::Cuda >( reset1, "GPU ET", maxCudaET );
#endif

   ////
   // Min
   auto minHost = [&]() {
      resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorMin( hostVector );
   };
   auto minCuda = [&]() {
      resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorMin( deviceVector );
   };
   auto minHostET = [&]() {
      resultHost = min( hostView );
   };
   auto minCudaET = [&]() {
      resultDevice = min( deviceView );
   };
   benchmark.setOperation( "min", datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU", minHost );
   benchmark.time< Devices::Host >( reset1, "CPU ET", minHostET );
#ifdef HAVE_CUDA
   benchmark.time< Devices::Cuda >( reset1, "GPU", minCuda );
   benchmark.time< Devices::Cuda >( reset1, "GPU ET", minCudaET );
#endif

   ////
   // Absmax
   auto absMaxHost = [&]() {
      resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorAbsMax( hostVector );
   };
   auto absMaxCuda = [&]() {
      resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorAbsMax( deviceVector );
   };
   auto absMaxHostET = [&]() {
      resultHost = max( abs( hostView ) );
   };
   auto absMaxCudaET = [&]() {
      resultDevice = max( abs( deviceView ) );
   };
#ifdef HAVE_BLAS
   auto absMaxBlas = [&]() {
      int index = blasIgamax( size, hostVector.getData(), 1 );
      resultHost = hostVector.getElement( index );
   };
#endif
#ifdef HAVE_CUDA
   auto absMaxCublas = [&]() {
      int index = 0;
      cublasIgamax( cublasHandle, size,
                    deviceVector.getData(), 1,
                    &index );
      resultDevice = deviceVector.getElement( index );
   };
#endif
   benchmark.setOperation( "absMax", datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU", absMaxHost );
   benchmark.time< Devices::Host >( reset1, "CPU ET", absMaxHostET );
#ifdef HAVE_BLAS
   benchmark.time< Devices::Host >( reset1, "CPU BLAS", absMaxBlas );
#endif
#ifdef HAVE_CUDA
   benchmark.time< Devices::Cuda >( reset1, "GPU", absMaxCuda );
   benchmark.time< Devices::Cuda >( reset1, "GPU ET", absMaxCudaET );
   benchmark.time< Devices::Cuda >( reset1, "cuBLAS", absMaxCublas );
#endif

   ////
   // Absmin
   auto absMinHost = [&]() {
      resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorAbsMin( hostVector );
   };
   auto absMinCuda = [&]() {
      resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorAbsMin( deviceVector );
   };
   auto absMinHostET = [&]() {
      resultHost = min( abs( hostView ) );
   };
   auto absMinCudaET = [&]() {
      resultDevice = min( abs( deviceView ) );
   };
/*#ifdef HAVE_BLAS
   auto absMinBlas = [&]() {
      int index = blasIgamin( size, hostVector.getData(), 1 );
      resultHost = hostVector.getElement( index );
   };
#endif*/
#ifdef HAVE_CUDA
   auto absMinCublas = [&]() {
      int index = 0;
      cublasIgamin( cublasHandle, size,
                    deviceVector.getData(), 1,
                    &index );
      resultDevice = deviceVector.getElement( index );
   };
#endif
   benchmark.setOperation( "absMin", datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU", absMinHost );
   benchmark.time< Devices::Host >( reset1, "CPU ET", absMinHostET );
   //benchmark.time< Devices::Host >( reset1, "CPU BLAS", absMinBlas );
#ifdef HAVE_CUDA
   benchmark.time< Devices::Cuda >( reset1, "GPU", absMinCuda );
   benchmark.time< Devices::Cuda >( reset1, "GPU ET", absMinCudaET );
   benchmark.time< Devices::Cuda >( reset1, "cuBLAS", absMinCublas );
#endif

   ////
   // Sum
   auto sumHost = [&]() {
      resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorSum( hostVector );
   };
   auto sumCuda = [&]() {
      resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorSum( deviceVector );
   };
   auto sumHostET = [&]() {
      resultHost = sum( hostView );
   };
   auto sumCudaET = [&]() {
      resultDevice = sum( deviceView );
   };
   benchmark.setOperation( "sum", datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU", sumHost );
   benchmark.time< Devices::Host >( reset1, "CPU ET", sumHostET );
#ifdef HAVE_CUDA
   benchmark.time< Devices::Cuda >( reset1, "GPU", sumCuda );
   benchmark.time< Devices::Cuda >( reset1, "GPU ET", sumCudaET );
#endif

   ////
   // L1 norm
   auto l1normHost = [&]() {
      resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorLpNorm( hostVector, 1.0 );
   };
   auto l1normCuda = [&]() {
      resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorLpNorm( deviceVector, 1.0 );
   };
   auto l1normHostET = [&]() {
      resultHost = lpNorm( hostView, 1.0 );
   };
   auto l1normCudaET = [&]() {
      resultDevice = lpNorm( deviceView, 1.0 );
   };
#ifdef HAVE_BLAS
   auto l1normBlas = [&]() {
      resultHost = blasGasum( size, hostVector.getData(), 1 );
   };
#endif
#ifdef HAVE_CUDA
   auto l1normCublas = [&]() {
      cublasGasum( cublasHandle, size,
                   deviceVector.getData(), 1,
                   &resultDevice );
   };
#endif
   benchmark.setOperation( "l1 norm", datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU", l1normHost );
   benchmark.time< Devices::Host >( reset1, "CPU ET", l1normHostET );
#ifdef HAVE_BLAS
   benchmark.time< Devices::Host >( reset1, "CPU BLAS", l1normBlas );
#endif
#ifdef HAVE_CUDA
   benchmark.time< Devices::Cuda >( reset1, "GPU", l1normCuda );
   benchmark.time< Devices::Cuda >( reset1, "GPU ET", l1normCudaET );
   benchmark.time< Devices::Cuda >( reset1, "cuBLAS", l1normCublas );
#endif

   ////
   // L2 norm
   auto l2normHost = [&]() {
      resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorLpNorm( hostVector, 2.0 );
   };
   auto l2normCuda = [&]() {
      resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorLpNorm( deviceVector, 2.0 );
   };
   auto l2normHostET = [&]() {
      resultHost = lpNorm( hostView, 2.0 );
   };
   auto l2normCudaET = [&]() {
      resultDevice = lpNorm( deviceView, 2.0 );
   };
#ifdef HAVE_BLAS
   auto l2normBlas = [&]() {
      resultHost = blasGnrm2( size, hostVector.getData(), 1 );
   };
#endif
#ifdef HAVE_CUDA
   auto l2normCublas = [&]() {
      cublasGnrm2( cublasHandle, size,
                   deviceVector.getData(), 1,
                   &resultDevice );
   };
#endif
   benchmark.setOperation( "l2 norm", datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU", l2normHost );
   benchmark.time< Devices::Host >( reset1, "CPU ET", l2normHostET );
#ifdef HAVE_BLAS
   benchmark.time< Devices::Host >( reset1, "CPU BLAS", l2normBlas );
#endif
#ifdef HAVE_CUDA
   benchmark.time< Devices::Cuda >( reset1, "GPU", l2normCuda );
   benchmark.time< Devices::Cuda >( reset1, "GPU ET", l2normCudaET );
   benchmark.time< Devices::Cuda >( reset1, "cuBLAS", l2normCublas );
#endif

   ////
   // L3 norm
   auto l3normHost = [&]() {
      resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorLpNorm( hostVector, 3.0 );
   };
   auto l3normCuda = [&]() {
      resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorLpNorm( deviceVector, 3.0 );
   };
   auto l3normHostET = [&]() {
      resultHost = lpNorm( hostView, 3.0 );
   };
   auto l3normCudaET = [&]() {
      resultDevice = lpNorm( deviceView, 3.0 );
   };

   benchmark.setOperation( "l3 norm", datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU", l3normHost );
   benchmark.time< Devices::Host >( reset1, "CPU ET", l3normHostET );
#ifdef HAVE_CUDA
   benchmark.time< Devices::Cuda >( reset1, "GPU", l3normCuda );
   benchmark.time< Devices::Cuda >( reset1, "GPU ET", l3normCudaET );
#endif

   ////
   // Scalar product
   auto scalarProductHost = [&]() {
      resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getScalarProduct( hostVector, hostVector2 );
   };
   auto scalarProductCuda = [&]() {
      resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getScalarProduct( deviceVector, deviceVector2 );
   };
   auto scalarProductHostET = [&]() {
      resultHost = ( hostView, hostView2 );
   };
   auto scalarProductCudaET = [&]() {
      resultDevice = ( deviceView, deviceView2 );
   };

#ifdef HAVE_BLAS
   auto scalarProductBlas = [&]() {
      resultHost = blasGdot( size, hostVector.getData(), 1, hostVector2.getData(), 1 );
   };
#endif
#ifdef HAVE_CUDA
   auto scalarProductCublas = [&]() {
      cublasGdot( cublasHandle, size,
                  deviceVector.getData(), 1,
                  deviceVector2.getData(), 1,
                  &resultDevice );
   };
#endif
   benchmark.setOperation( "scalar product", 2 * datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU", scalarProductHost );
   benchmark.time< Devices::Host >( reset1, "CPU ET", scalarProductHostET );
#ifdef HAVE_BLAS
   benchmark.time< Devices::Host >( reset1, "CPU BLAS", scalarProductBlas );
#endif
#ifdef HAVE_CUDA
   benchmark.time< Devices::Cuda >( reset1, "GPU", scalarProductCuda );
   benchmark.time< Devices::Cuda >( reset1, "GPU ET", scalarProductCudaET );
   benchmark.time< Devices::Cuda >( reset1, "cuBLAS", scalarProductCublas );
#endif

   ////
   // Prefix sum
   /*
   std::cout << "Benchmarking prefix-sum:" << std::endl;
   timer.reset();
   timer.start();
   hostVector.computePrefixSum();
   timer.stop();
   timeHost = timer.getTime();
   bandwidth = 2 * datasetSize / timer.getTime();
   std::cout << "  CPU: bandwidth: " << bandwidth << " GB/sec, time: " << timer.getTime() << " sec." << std::endl;

   timer.reset();
   timer.start();
   deviceVector.computePrefixSum();
   timer.stop();
   timeDevice = timer.getTime();
   bandwidth = 2 * datasetSize / timer.getTime();
   std::cout << "  GPU: bandwidth: " << bandwidth << " GB/sec, time: " << timer.getTime() << " sec." << std::endl;
   std::cout << "  CPU/GPU speedup: " << timeHost / timeDevice << std::endl;

   HostVector auxHostVector;
   auxHostVector.setLike( deviceVector );
   auxHostVector = deviceVector;
   for( int i = 0; i < size; i++ )
      if( hostVector.getElement( i ) != auxHostVector.getElement( i ) )
      {
         std::cerr << "Error in prefix sum at position " << i << ":  " << hostVector.getElement( i ) << " != " << auxHostVector.getElement( i ) << std::endl;
      }
   */


   ////
   // Scalar multiplication
   auto multiplyHost = [&]() {
      hostVector *= 0.5;
   };
   auto multiplyCuda = [&]() {
      deviceVector *= 0.5;
   };
#ifdef HAVE_CUDA
   auto multiplyCublas = [&]() {
      const Real alpha = 0.5;
      cublasGscal( cublasHandle, size,
                   &alpha,
                   deviceVector.getData(), 1 );
   };
#endif
   benchmark.setOperation( "scalar multiplication", 2 * datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU", multiplyHost );
#ifdef HAVE_CUDA
   benchmark.time< Devices::Cuda >( reset1, "GPU", multiplyCuda );
   benchmark.time< Devices::Cuda >( reset1, "cuBLAS", multiplyCublas );
#endif

   ////
   // Vector addition
   auto addVectorHost = [&]() {
      hostVector.addVector( hostVector2 );
   };
   auto addVectorCuda = [&]() {
      deviceVector.addVector( deviceVector2 );
   };
   auto addVectorHostET = [&]() {
      hostView += hostView2;
   };
   auto addVectorCudaET = [&]() {
      deviceView += deviceView2;
   };
#ifdef HAVE_BLAS
   auto addVectorBlas = [&]() {
      const Real alpha = 1.0;
      blasGaxpy( size, alpha,
                 hostVector2.getData(), 1,
                 hostVector.getData(), 1 );
   };
#endif
#ifdef HAVE_CUDA
   auto addVectorCublas = [&]() {
      const Real alpha = 1.0;
      cublasGaxpy( cublasHandle, size,
                   &alpha,
                   deviceVector2.getData(), 1,
                   deviceVector.getData(), 1 );
   };
#endif
   benchmark.setOperation( "vector addition", 3 * datasetSize );
   benchmark.time< Devices::Host >( resetAll, "CPU", addVectorHost );
   benchmark.time< Devices::Host >( resetAll, "CPU ET", addVectorHostET );
#ifdef HAVE_BLAS
   benchmark.time< Devices::Host >( resetAll, "CPU BLAS", addVectorBlas );
#endif
#ifdef HAVE_CUDA
   benchmark.time< Devices::Cuda >( resetAll, "GPU", addVectorCuda );
   benchmark.time< Devices::Cuda >( resetAll, "GPU ET", addVectorCudaET );
   benchmark.time< Devices::Cuda >( resetAll, "cuBLAS", addVectorCublas );
#endif

   ////
   // Two vectors addition
   auto addTwoVectorsHost = [&]() {
      hostVector.addVector( hostVector2 );
      hostVector.addVector( hostVector3 );
   };
   auto addTwoVectorsCuda = [&]() {
      deviceVector.addVector( deviceVector2 );
      deviceVector.addVector( deviceVector3 );
   };
   auto addTwoVectorsHostET = [&]() {
      hostView += hostView2 + hostView3;
   };
   auto addTwoVectorsCudaET = [&]() {
      deviceView += deviceView2 + deviceView3;
   };
#ifdef HAVE_BLAS
   auto addTwoVectorsBlas = [&]() {
      const Real alpha = 1.0;
      blasGaxpy( size, alpha,
                 hostVector2.getData(), 1,
                 hostVector.getData(), 1 );
      blasGaxpy( size, alpha,
                 hostVector3.getData(), 1,
                 hostVector.getData(), 1 );
   };
#endif
#ifdef HAVE_CUDA
   auto addTwoVectorsCublas = [&]() {
      const Real alpha = 1.0;
      cublasGaxpy( cublasHandle, size,
                   &alpha,
                   deviceVector2.getData(), 1,
                   deviceVector.getData(), 1 );
      cublasGaxpy( cublasHandle, size,
                   &alpha,
                   deviceVector3.getData(), 1,
                   deviceVector.getData(), 1 );
   };
#endif
   benchmark.setOperation( "two vectors addition", 4 * datasetSize );
   benchmark.time< Devices::Host >( resetAll, "CPU", addTwoVectorsHost );
   benchmark.time< Devices::Host >( resetAll, "CPU ET", addTwoVectorsHostET );
#ifdef HAVE_BLAS
   benchmark.time< Devices::Host >( resetAll, "CPU BLAS", addTwoVectorsBlas );
#endif
#ifdef HAVE_CUDA
   benchmark.time< Devices::Cuda >( resetAll, "GPU", addTwoVectorsCuda );
   benchmark.time< Devices::Cuda >( resetAll, "GPU ET", addTwoVectorsCudaET );
   benchmark.time< Devices::Cuda >( resetAll, "cuBLAS", addTwoVectorsCublas );
#endif

   ////
   // Three vectors addition
   auto addThreeVectorsHost = [&]() {
      hostVector.addVector( hostVector2 );
      hostVector.addVector( hostVector3 );
      hostVector.addVector( hostVector4 );
   };
   auto addThreeVectorsCuda = [&]() {
      deviceVector.addVector( deviceVector2 );
      deviceVector.addVector( deviceVector3 );
      deviceVector.addVector( deviceVector4 );
   };
   auto addThreeVectorsHostET = [&]() {
      hostView += hostView2 + hostView3 + hostView4;
   };
   auto addThreeVectorsCudaET = [&]() {
      deviceView += deviceView2 + deviceView3 + deviceView4;
   };
#ifdef HAVE_BLAS
   auto addThreeVectorsBlas = [&]() {
      const Real alpha = 1.0;
      blasGaxpy( size, alpha,
                 hostVector2.getData(), 1,
                 hostVector.getData(), 1 );
      blasGaxpy( size, alpha,
                 hostVector3.getData(), 1,
                 hostVector.getData(), 1 );
       blasGaxpy( size, alpha,
                 hostVector4.getData(), 1,
                 hostVector.getData(), 1 );
   };
#endif
#ifdef HAVE_CUDA
   auto addThreeVectorsCublas = [&]() {
      const Real alpha = 1.0;
      cublasGaxpy( cublasHandle, size,
                   &alpha,
                   deviceVector2.getData(), 1,
                   deviceVector.getData(), 1 );
      cublasGaxpy( cublasHandle, size,
                   &alpha,
                   deviceVector3.getData(), 1,
                   deviceVector.getData(), 1 );
      cublasGaxpy( cublasHandle, size,
                   &alpha,
                   deviceVector4.getData(), 1,
                   deviceVector.getData(), 1 );
   };
#endif
   benchmark.setOperation( "three vectors addition", 5 * datasetSize );
   benchmark.time< Devices::Host >( resetAll, "CPU", addThreeVectorsHost );
   benchmark.time< Devices::Host >( resetAll, "CPU ET", addThreeVectorsHostET );
#ifdef HAVE_BLAS
   benchmark.time< Devices::Host >( resetAll, "CPU BLAS", addThreeVectorsBlas );
#endif
#ifdef HAVE_CUDA
   benchmark.time< Devices::Cuda >( resetAll, "GPU", addThreeVectorsCuda );
   benchmark.time< Devices::Cuda >( resetAll, "GPU ET", addThreeVectorsCudaET );
   benchmark.time< Devices::Cuda >( resetAll, "cuBLAS", addThreeVectorsCublas );
#endif

#ifdef HAVE_CUDA
   cublasDestroy( cublasHandle );
#endif

   return true;
}

} // namespace Benchmarks
} // namespace TNL
