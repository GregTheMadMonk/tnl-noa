#pragma once

#include "benchmarks.h"

#include <core/vectors/tnlVector.h>

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
 
 
    // check functions
    auto compare1 = [&]() {
        return hostVector == deviceVector;
    };
    auto compare2 = [&]() {
        return hostVector2 == deviceVector2;
    };
    auto compare12 = [&]() {
        return compare1() && compare2();
    };
    auto compareScalars = [&]() {
        return resultHost == resultDevice;
    };
 
    // reset functions
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
 
    cout << "Benchmarking CPU-GPU memory transfer:" << endl;
    auto copyAssign = [&]() {
        deviceVector = hostVector;
    };
    cout << "  ";
    benchmarkSingle( loops, datasetSize, copyAssign, compare1, reset1 );
     
 
    cout << "Benchmarking vector addition:" << endl;
    auto addVectorHost = [&]() {
        hostVector.addVector( hostVector2 );
    };
    auto addVectorCuda = [&]() {
        deviceVector.addVector( deviceVector2 );
    };
    benchmarkCuda( loops, 3 * datasetSize, addVectorHost, addVectorCuda, compare1, reset1 );
 
 
    cout << "Benchmarking scalar product:" << endl;
    auto scalarProductHost = [&]() {
        resultHost = hostVector.scalarProduct( hostVector2 );
    };
    auto scalarProductCuda = [&]() {
        resultDevice = deviceVector.scalarProduct( deviceVector2 );
    };
    benchmarkCuda( loops, 2 * datasetSize, scalarProductHost, scalarProductCuda, compareScalars, voidFunc );

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

    cout << "Benchmarking L2 norm: " << endl;
    auto l2normHost = [&]() {
        resultHost = hostVector.lpNorm( 2.0 );
    };
    auto l2normCuda = [&]() {
        resultDevice = deviceVector.lpNorm( 2.0 );
    };
    benchmarkCuda( loops, datasetSize, l2normHost, l2normCuda, compareScalars, voidFunc );
 
 
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
