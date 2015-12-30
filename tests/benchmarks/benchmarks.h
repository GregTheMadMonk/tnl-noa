#pragma once

#include <iostream>

#include <core/tnlTimerRT.h>

namespace tnl
{
namespace benchmarks
{

// TODO: add data member for error message
struct BenchmarkError {};

auto trueFunc = []() { return true; };
auto voidFunc = [](){};

template< typename ComputeFunction,
          typename CheckFunction,
          typename ResetFunction >
double
benchmarkSingle( const int & loops,
                 const double & datasetSize, // in GB
                 ComputeFunction compute,
                 // TODO: check that default argument works here
                 CheckFunction check = trueFunc,
                 ResetFunction reset = voidFunc )
{
    tnlTimerRT timer;
    timer.reset();

    for(int i = 0; i < loops; ++i) {
        timer.start();
        compute();
        timer.stop();

        if( ! check() )
            throw BenchmarkError();

        reset();
    }

    const double time = timer.getTime();
    const double bandwidth = datasetSize / time;
    std::cout << "bandwidth: " << bandwidth << " GB/sec, time: " << time << " sec." << std::endl;

    return time;
}

template< typename ComputeHostFunction,
          typename ComputeCudaFunction,
          typename CheckFunction,
          typename ResetFunction >
void
benchmarkCuda( const int & loops,
               const double & datasetSize, // in GB
               ComputeHostFunction computeHost,
               ComputeCudaFunction computeCuda,
               // TODO: check that default argument works here
               CheckFunction check = trueFunc,
               ResetFunction reset = voidFunc )
{
    tnlTimerRT timerHost, timerCuda;
    timerHost.reset();
    timerHost.stop();
    timerCuda.reset();
    timerCuda.stop();

    for(int i = 0; i < loops; ++i) {
        timerHost.start();
        computeHost();
        timerHost.stop();

        timerCuda.start();
        computeCuda();
        timerCuda.stop();

        if( ! check() )
            throw BenchmarkError();

        reset();
    }

    const double timeHost = timerHost.getTime();
    const double timeCuda = timerCuda.getTime();
    const double bandwidthHost = datasetSize / timeHost;
    const double bandwidthCuda = datasetSize / timeCuda;
    std::cout << "  CPU: bandwidth: " << bandwidthHost << " GB/sec, time: " << timeHost << " sec." << std::endl;
    std::cout << "  GPU: bandwidth: " << bandwidthCuda << " GB/sec, time: " << timeCuda << " sec." << std::endl;
    std::cout << "  CPU/GPU speedup: " << timeHost / timeCuda << std::endl;
}

} // namespace benchmarks
} // namespace tnl
