#pragma once

#include <iostream>
#include <iomanip>

#include <core/tnlTimerRT.h>

namespace tnl
{
namespace benchmarks
{

const double oneGB = 1024.0 * 1024.0 * 1024.0;

template< typename ComputeFunction,
          typename ResetFunction >
double
timeFunction( ComputeFunction compute,
              ResetFunction reset,
              const int & loops,
              const double & datasetSize, // in GB
              const double & baseTime, // in seconds (baseline for speedup calculation)
              const char* performer )
{
    // the timer is constructed zero-initialized and stopped
    tnlTimerRT timer;

    reset();
    for(int i = 0; i < loops; ++i) {
        // TODO: not necessary for host computations
        // Explicit synchronization of the CUDA device
#ifdef HAVE_CUDA
        cudaDeviceSynchronize();
#endif
        timer.start();
        compute();
#ifdef HAVE_CUDA
        cudaDeviceSynchronize();
#endif
        timer.stop();

        reset();
    }

    const double time = timer.getTime();
    const double bandwidth = datasetSize / time;

    using namespace std;
    cout << "  " << performer << ": bandwidth: "
         << setw( 8 ) << bandwidth << " GB/sec, time: "
         << setw( 8 ) << time << " sec, speedup: ";
    if( baseTime )
        cout << baseTime / time << endl;
    else
        cout << "N/A" << endl;

    return time;
}

// This specialization terminates the recursion
template< typename ResetFunction,
          typename ComputeFunction >
inline void
benchmarkNextOperation( const double & datasetSize,
                        const int & loops,
                        ResetFunction reset,
                        const double & baseTime,
                        const char* performer,
                        ComputeFunction compute )
{
    timeFunction( compute, reset, loops, datasetSize, baseTime, performer );
}

// Recursive template function to deal with benchmarks involving multiple computations
template< typename ResetFunction,
          typename ComputeFunction,
          typename... NextComputations >
inline void
benchmarkNextOperation( const double & datasetSize,
                        const int & loops,
                        ResetFunction reset,
                        const double & baseTime,
                        const char* performer,
                        ComputeFunction compute,
                        NextComputations & ... nextComputations )
{
    benchmarkNextOperation( datasetSize, loops, reset, baseTime, performer, compute );
    benchmarkNextOperation( datasetSize, loops, reset, baseTime, nextComputations... );
}

// Main function for benchmarking
template< typename ResetFunction,
          typename ComputeFunction,
          typename... NextComputations >
void
benchmarkOperation( const char* operation,
                    const double & datasetSize,
                    const int & loops,
                    ResetFunction reset,
                    const char* performer,
                    ComputeFunction computeBase,
                    NextComputations... nextComputations )
{
    cout << "Benchmarking " << operation << ":" << endl;
    double baseTime = timeFunction( computeBase, reset, loops, datasetSize, 0.0, performer );
    benchmarkNextOperation( datasetSize, loops, reset, baseTime, nextComputations... );
    std::cout << std::endl;
}

} // namespace benchmarks
} // namespace tnl
