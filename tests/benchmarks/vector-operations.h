#pragma once

#include "benchmarks.h"

#include <core/vectors/tnlVector.h>

#ifdef HAVE_CUBLAS
//#include <cublas.h>
#endif

namespace tnl
{
namespace benchmarks
{

template< typename Real = double,
          typename Index = int >
bool
benchmarkVectorOperations( const int & loops,
                           const int & size )
{
    typedef tnlVector< Real, tnlHost, Index > HostVector;
    typedef tnlVector< Real, tnlCuda, Index > CudaVector;
    using namespace std;

    double datasetSize = ( double ) ( loops * size ) * sizeof( Real ) / oneGB;

    HostVector hostVector, hostVector2;
    CudaVector deviceVector, deviceVector2;
    hostVector.setSize( size );
    if( ! deviceVector.setSize( size ) )
        return false;
    hostVector2.setLike( hostVector );
    if( ! deviceVector2.setLike( deviceVector ) )
        return false;

    Real resultHost, resultDevice;


    // reset functions
    // (Make sure to always use some in benchmarks, even if it's not necessary
    // to assure correct result - it helps to clear cache and avoid optimizations
    // of the benchmark loop.)
    auto reset1 = [&]() {
        hostVector.setValue( 1.0 );
        deviceVector.setValue( 1.0 );
    };
    auto reset2 = [&]() {
        hostVector2.setValue( 1.0 );
        deviceVector2.setValue( 1.0 );
    };
    auto reset12 = [&]() {
        reset1();
        reset2();
    };


    reset12();


    auto copyAssignHostHost = [&]() {
        hostVector = hostVector2;
    };
    auto copyAssignHostCuda = [&]() {
        deviceVector = hostVector;
    };
    auto copyAssignCudaCuda = [&]() {
        deviceVector = hostVector;
    };
    benchmarkOperation( "copy assigment", datasetSize, loops, reset1,
                        "CPU->CPU", copyAssignHostHost,
                        "CPU->GPU", copyAssignHostCuda,
                        "GPU->GPU", copyAssignCudaCuda );


    auto compareHost = [&]() {
        resultHost = (int) hostVector == hostVector2;
    };
    auto compareCuda = [&]() {
        resultDevice = (int) deviceVector == deviceVector2;
    };
    benchmarkOperation( "comparison (operator==)", 2 * datasetSize, loops, reset1,
                        "CPU", compareHost,
                        "GPU", compareCuda );


    auto multiplyHost = [&]() {
        hostVector *= 0.5;
    };
    auto multiplyCuda = [&]() {
        deviceVector *= 0.5;
    };
    benchmarkOperation( "scalar multiplication", 2 * datasetSize, loops, reset1,
                        "CPU", multiplyHost,
                        "GPU", multiplyCuda );


    auto addVectorHost = [&]() {
        hostVector.addVector( hostVector2 );
    };
    auto addVectorCuda = [&]() {
        deviceVector.addVector( deviceVector2 );
    };
    benchmarkOperation( "vector addition", 3 * datasetSize, loops, reset1,
                        "CPU", addVectorHost,
                        "GPU", addVectorCuda );


    auto maxHost = [&]() {
        resultHost = hostVector.max();
    };
    auto maxCuda = [&]() {
        resultDevice = deviceVector.max();
    };
    benchmarkOperation( "max", datasetSize, loops, reset1,
                        "CPU", maxHost,
                        "GPU", maxCuda );


    auto minHost = [&]() {
        resultHost = hostVector.min();
    };
    auto minCuda = [&]() {
        resultDevice = deviceVector.min();
    };
    benchmarkOperation( "min", datasetSize, loops, reset1,
                        "CPU", minHost,
                        "GPU", minCuda );


    auto absMaxHost = [&]() {
        resultHost = hostVector.absMax();
    };
    auto absMaxCuda = [&]() {
        resultDevice = deviceVector.absMax();
    };
    benchmarkOperation( "absMax", datasetSize, loops, reset1,
                        "CPU", absMaxHost,
                        "GPU", absMaxCuda );


    auto absMinHost = [&]() {
        resultHost = hostVector.absMin();
    };
    auto absMinCuda = [&]() {
        resultDevice = deviceVector.absMin();
    };
    benchmarkOperation( "absMin", datasetSize, loops, reset1,
                        "CPU", absMinHost,
                        "GPU", absMinCuda );


    auto sumHost = [&]() {
        resultHost = hostVector.sum();
    };
    auto sumCuda = [&]() {
        resultDevice = deviceVector.sum();
    };
    benchmarkOperation( "sum", datasetSize, loops, reset1,
                        "CPU", sumHost,
                        "GPU", sumCuda );


    auto l1normHost = [&]() {
        resultHost = hostVector.lpNorm( 1.0 );
    };
    auto l1normCuda = [&]() {
        resultDevice = deviceVector.lpNorm( 1.0 );
    };
    benchmarkOperation( "l1 norm", datasetSize, loops, reset1,
                        "CPU", l1normHost,
                        "GPU", l1normCuda );


    auto l2normHost = [&]() {
        resultHost = hostVector.lpNorm( 2.0 );
    };
    auto l2normCuda = [&]() {
        resultDevice = deviceVector.lpNorm( 2.0 );
    };
    benchmarkOperation( "l2 norm", datasetSize, loops, reset1,
                        "CPU", l2normHost,
                        "GPU", l2normCuda );


    auto l3normHost = [&]() {
        resultHost = hostVector.lpNorm( 3.0 );
    };
    auto l3normCuda = [&]() {
        resultDevice = deviceVector.lpNorm( 3.0 );
    };
    benchmarkOperation( "l3 norm", datasetSize, loops, reset1,
                        "CPU", l3normHost,
                        "GPU", l3normCuda );


    auto scalarProductHost = [&]() {
        resultHost = hostVector.scalarProduct( hostVector2 );
    };
    auto scalarProductCuda = [&]() {
        resultDevice = deviceVector.scalarProduct( deviceVector2 );
    };
    benchmarkOperation( "scalar product", 2 * datasetSize, loops, reset1,
                        "CPU", scalarProductHost,
                        "GPU", scalarProductCuda );

/* TODO
#ifdef HAVE_CUBLAS
   cout << "Benchmarking scalar product on GPU with Cublas: " << endl;
   cublasHandle_t handle;
   cublasCreate( &handle );
   timer.reset();
   timer.start();
   for( int i = 0; i < loops; i++ )
      cublasDdot( handle,
                  size,
                  deviceVector.getData(), 1,
                  deviceVector.getData(), 1,
                  &resultDevice );
   cudaThreadSynchronize();
   timer.stop();
   bandwidth = 2 * datasetSize / timer.getTime();
   cout << "bandwidth: " << bandwidth << " GB/sec, time: " << timer.getTime() << " sec." << endl;
#endif
*/

    /*
    cout << "Benchmarking prefix-sum:" << endl;
    timer.reset();
    timer.start();
    hostVector.computePrefixSum();
    timer.stop();
    timeHost = timer.getTime();
    bandwidth = 2 * datasetSize / loops / timer.getTime();
    cout << "  CPU: bandwidth: " << bandwidth << " GB/sec, time: " << timer.getTime() << " sec." << endl;

    timer.reset();
    timer.start();
    deviceVector.computePrefixSum();
    timer.stop();
    timeDevice = timer.getTime();
    bandwidth = 2 * datasetSize / loops / timer.getTime();
    cout << "  GPU: bandwidth: " << bandwidth << " GB/sec, time: " << timer.getTime() << " sec." << endl;
    cout << "  CPU/GPU speedup: " << timeHost / timeDevice << endl;

    HostVector auxHostVector;
    auxHostVector.setLike( deviceVector );
    auxHostVector = deviceVector;
    for( int i = 0; i < size; i++ )
       if( hostVector.getElement( i ) != auxHostVector.getElement( i ) )
       {
          cerr << "Error in prefix sum at position " << i << ":  " << hostVector.getElement( i ) << " != " << auxHostVector.getElement( i ) << endl;
       }
    */

    return true;
}

} // namespace benchmarks
} // namespace tnl
