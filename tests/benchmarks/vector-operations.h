#pragma once

#include "benchmarks.h"

#include <core/vectors/tnlVector.h>

#ifdef HAVE_CUBLAS
#include "cublasWrappers.h"
#endif

namespace tnl
{
namespace benchmarks
{

template< typename Real = double,
          typename Index = int >
bool
benchmarkVectorOperations( Benchmark & benchmark,
                           const int & loops,
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

#ifdef HAVE_CUBLAS
    cublasHandle_t cublasHandle;
    cublasCreate( &cublasHandle );
#endif


    // reset functions
    // (Make sure to always use some in benchmarks, even if it's not necessary
    // to assure correct result - it helps to clear cache and avoid optimizations
    // of the benchmark loop.)
    auto reset1 = [&]() {
        hostVector.setValue( 1.0 );
        deviceVector.setValue( 1.0 );
        resultHost = resultDevice = 0.0;
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


    auto multiplyHost = [&]() {
        hostVector *= 0.5;
    };
    auto multiplyCuda = [&]() {
        deviceVector *= 0.5;
    };
    benchmark.setOperation( "scalar multiplication", 2 * datasetSize );
    benchmark.time( reset1,
                    "CPU", multiplyHost,
                    "GPU", multiplyCuda );


    auto addVectorHost = [&]() {
        hostVector.addVector( hostVector2 );
    };
    auto addVectorCuda = [&]() {
        deviceVector.addVector( deviceVector2 );
    };
    benchmark.setOperation( "vector addition", 3 * datasetSize );
    benchmark.time( reset1,
                    "CPU", addVectorHost,
                    "GPU", addVectorCuda );


    auto maxHost = [&]() {
        resultHost = hostVector.max();
    };
    auto maxCuda = [&]() {
        resultDevice = deviceVector.max();
    };
    benchmark.setOperation( "max", datasetSize );
    benchmark.time( reset1,
                    "CPU", maxHost,
                    "GPU", maxCuda );


    auto minHost = [&]() {
        resultHost = hostVector.min();
    };
    auto minCuda = [&]() {
        resultDevice = deviceVector.min();
    };
    benchmark.setOperation( "min", datasetSize );
    benchmark.time( reset1,
                    "CPU", minHost,
                    "GPU", minCuda );


    auto absMaxHost = [&]() {
        resultHost = hostVector.absMax();
    };
    auto absMaxCuda = [&]() {
        resultDevice = deviceVector.absMax();
    };
    benchmark.setOperation( "absMax", datasetSize );
    benchmark.time( reset1,
                    "CPU", absMaxHost,
                    "GPU", absMaxCuda );


    auto absMinHost = [&]() {
        resultHost = hostVector.absMin();
    };
    auto absMinCuda = [&]() {
        resultDevice = deviceVector.absMin();
    };
    benchmark.setOperation( "absMin", datasetSize );
    benchmark.time( reset1,
                    "CPU", absMinHost,
                    "GPU", absMinCuda );


    auto sumHost = [&]() {
        resultHost = hostVector.sum();
    };
    auto sumCuda = [&]() {
        resultDevice = deviceVector.sum();
    };
    benchmark.setOperation( "sum", datasetSize );
    benchmark.time( reset1,
                    "CPU", sumHost,
                    "GPU", sumCuda );


    auto l1normHost = [&]() {
        resultHost = hostVector.lpNorm( 1.0 );
    };
    auto l1normCuda = [&]() {
        resultDevice = deviceVector.lpNorm( 1.0 );
    };
    benchmark.setOperation( "l1 norm", datasetSize );
    benchmark.time( reset1,
                    "CPU", l1normHost,
                    "GPU", l1normCuda );


    auto l2normHost = [&]() {
        resultHost = hostVector.lpNorm( 2.0 );
    };
    auto l2normCuda = [&]() {
        resultDevice = deviceVector.lpNorm( 2.0 );
    };
    benchmark.setOperation( "l2 norm", datasetSize );
    benchmark.time( reset1,
                    "CPU", l2normHost,
                    "GPU", l2normCuda );


    auto l3normHost = [&]() {
        resultHost = hostVector.lpNorm( 3.0 );
    };
    auto l3normCuda = [&]() {
        resultDevice = deviceVector.lpNorm( 3.0 );
    };
    benchmark.setOperation( "l3 norm", datasetSize );
    benchmark.time( reset1,
                    "CPU", l3normHost,
                    "GPU", l3normCuda );


    auto scalarProductHost = [&]() {
        resultHost = hostVector.scalarProduct( hostVector2 );
    };
    auto scalarProductCuda = [&]() {
        resultDevice = deviceVector.scalarProduct( deviceVector2 );
    };
#ifdef HAVE_CUBLAS
    auto scalarProductCublas = [&]() {
        cublasGdot( cublasHandle, size,
                    deviceVector.getData(), 1,
                    deviceVector2.getData(), 1,
                    &resultDevice );
    };
#endif
    benchmark.setOperation( "scalar product", 2 * datasetSize );
    benchmark.time( reset1,
                    "CPU", scalarProductHost,
                    "GPU", scalarProductCuda
#ifdef HAVE_CUBLAS
                  , "cuBLAS", scalarProductCublas
#endif
                  );

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

#ifdef HAVE_CUBLAS
    cublasDestroy( cublasHandle );
#endif

    return true;
}

} // namespace benchmarks
} // namespace tnl