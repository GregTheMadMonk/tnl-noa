#pragma once

#include "benchmarks.h"

#include <TNL/Containers/Array.h>

namespace TNL
{
namespace benchmarks
{

template< typename Real = double,
          typename Index = int >
bool
benchmarkArrayOperations( Benchmark & benchmark,
                          const int & loops,
                          const long & size )
{
    typedef Containers::Array< Real, Devices::Host, Index > HostArray;
    typedef Containers::Array< Real, Devices::Cuda, Index > CudaArray;
    using namespace std;

    double datasetSize = ( double ) ( loops * size ) * sizeof( Real ) / oneGB;

    HostArray hostArray, hostArray2;
    CudaArray deviceArray, deviceArray2;
    if( ! hostArray.setSize( size ) ||
        ! hostArray2.setSize( size )
#ifdef HAVE_CUDA
        ||
        ! deviceArray.setSize( size ) ||
        ! deviceArray2.setSize( size )
#endif
    )

    {
        const char* msg = "error: allocation of arrays failed";
        std::cerr << msg << std::endl;
        benchmark.addErrorMessage( msg );
        return false;
    }

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
        resultHost = (int) hostArray == hostArray2;
    };
    auto compareCuda = [&]() {
        resultDevice = (int) deviceArray == deviceArray2;
    };
    benchmark.setOperation( "comparison (operator==)", 2 * datasetSize );
    benchmark.time( reset1, "CPU", compareHost );
#ifdef HAVE_CUDA
    benchmark.time( reset1, "GPU", compareCuda );
#endif


    auto copyAssignHostHost = [&]() {
        hostArray = hostArray2;
    };
    auto copyAssignCudaCuda = [&]() {
        deviceArray = deviceArray2;
    };
    benchmark.setOperation( "copy (operator=)", 2 * datasetSize );
    benchmark.time( reset1, "CPU", copyAssignHostHost );
#ifdef HAVE_CUDA
    benchmark.time( reset1, "GPU", copyAssignCudaCuda );
#endif


    auto copyAssignHostCuda = [&]() {
        deviceArray = hostArray;
    };
    auto copyAssignCudaHost = [&]() {
        hostArray = deviceArray;
    };
#ifdef HAVE_CUDA
    benchmark.setOperation( "copy (operator=)", datasetSize, basetime );
    benchmark.time( reset1,
                    "CPU->GPU", copyAssignHostCuda,
                    "GPU->CPU", copyAssignCudaHost );
#endif


    auto setValueHost = [&]() {
        hostArray.setValue( 3.0 );
    };
    auto setValueCuda = [&]() {
        deviceArray.setValue( 3.0 );
    };
    benchmark.setOperation( "setValue", datasetSize );
    benchmark.time( reset1, "CPU", setValueHost );
#ifdef HAVE_CUDA
    benchmark.time( reset1, "GPU", setValueCuda );
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
    benchmark.time( resetSize1, "CPU", setSizeHost );
#ifdef HAVE_CUDA
    benchmark.time( resetSize1, "GPU", setSizeCuda );
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
    benchmark.time( setSize1, "CPU", resetSizeHost );
#ifdef HAVE_CUDA
    benchmark.time( setSize1, "GPU", resetSizeCuda );
#endif

    return true;
}

} // namespace benchmarks
} // namespace tnl
