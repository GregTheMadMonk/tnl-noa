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
    if( ! hostArray.setSize( size ) ||
        ! hostArray2.setSize( size ) ||
        ! deviceArray.setSize( size ) ||
        ! deviceArray2.setSize( size ) )
    {
        const char* msg = "error: allocation of arrays failed";
        cerr << msg << endl;
        benchmark.addErrorMessage( msg );
        return false;
    }

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


    auto setValueHost = [&]() {
        hostArray.setValue( 3.0 );
    };
    auto setValueCuda = [&]() {
        deviceArray.setValue( 3.0 );
    };
    benchmark.setOperation( "setValue", datasetSize );
    benchmark.time( reset1,
                    "CPU", setValueHost,
                    "GPU", setValueCuda );


    auto setSizeHost = [&]() {
        hostArray.setSize( size );
    };
    auto setSizeCuda = [&]() {
        deviceArray.setSize( size );
    };
    auto resetSize1 = [&]() {
        hostArray.reset();
        deviceArray.reset();
    };
    benchmark.setOperation( "allocation (setSize)", datasetSize );
    benchmark.time( resetSize1,
                    "CPU", setSizeHost,
                    "GPU", setSizeCuda );


    auto resetSizeHost = [&]() {
        hostArray.reset();
    };
    auto resetSizeCuda = [&]() {
        deviceArray.reset();
    };
    auto setSize1 = [&]() {
        hostArray.setSize( size );
        deviceArray.setSize( size );
    };
    benchmark.setOperation( "deallocation (reset)", datasetSize );
    benchmark.time( setSize1,
                    "CPU", resetSizeHost,
                    "GPU", resetSizeCuda );

    return true;
}

} // namespace benchmarks
} // namespace tnl
