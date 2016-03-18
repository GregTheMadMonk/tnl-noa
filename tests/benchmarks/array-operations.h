#pragma once

#include "benchmarks.h"

#include <core/arrays/tnlArray.h>

namespace tnl
{
namespace benchmarks
{

template< typename Real = double,
          typename Index = int >
bool
benchmarkArrayOperations( Benchmark & benchmark,
                          const int & loops,
                          const int & size )
{
    typedef tnlArray< Real, tnlHost, Index > HostArray;
    typedef tnlArray< Real, tnlCuda, Index > CudaArray;
    using namespace std;

    double datasetSize = ( double ) ( loops * size ) * sizeof( Real ) / oneGB;

    HostArray hostArray, hostArray2;
    CudaArray deviceArray, deviceArray2;
    hostArray.setSize( size );
    if( ! deviceArray.setSize( size ) )
        return false;
    hostArray2.setLike( hostArray );
    if( ! deviceArray2.setLike( deviceArray ) )
        return false;

    Real resultHost, resultDevice;


    // reset functions
    auto reset1 = [&]() {
        hostArray.setValue( 1.0 );
        deviceArray.setValue( 1.0 );
    };
    auto reset2 = [&]() {
        hostArray2.setValue( 1.0 );
        deviceArray2.setValue( 1.0 );
    };
    auto reset12 = [&]() {
        reset1();
        reset2();
    };


    reset12();


    auto compareHost = [&]() {
        resultHost = (int) hostArray == hostArray2;
    };
    auto compareCuda = [&]() {
        resultDevice = (int) deviceArray == deviceArray2;
    };
    benchmark.setOperation( "comparison (operator==)", 2 * datasetSize );
    benchmark.time( reset1,
                    "CPU", compareHost,
                    "GPU", compareCuda );


    auto copyAssignHostHost = [&]() {
        hostArray = hostArray2;
    };
    auto copyAssignCudaCuda = [&]() {
        deviceArray = deviceArray2;
    };
    benchmark.setOperation( "copy (operator=)", 2 * datasetSize );
    double basetime = benchmark.time( reset1,
                    "CPU", copyAssignHostHost,
                    "GPU", copyAssignCudaCuda );


    auto copyAssignHostCuda = [&]() {
        deviceArray = hostArray;
    };
    auto copyAssignCudaHost = [&]() {
        hostArray = deviceArray;
    };
    benchmark.setOperation( "copy (operator=)", datasetSize, basetime );
    benchmark.time( reset1,
                    "CPU->GPU", copyAssignHostCuda,
                    "GPU->CPU", copyAssignCudaHost );

    return true;
}

} // namespace benchmarks
} // namespace tnl
