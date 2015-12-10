/***************************************************************************
                          tnl-benchmarks.h  -  description
                             -------------------
    begin                : Jan 27, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLCUDABENCHMARKS_H_
#define TNLCUDBENCHMARKS_H_

#include <tnlConfig.h>
#include <core/vectors/tnlVector.h>
#include <core/tnlTimerRT.h>

#ifdef HAVE_CUBLAS
//#include <cublas.h>
#endif    



int main( int argc, char* argv[] )
{
#ifdef HAVE_CUDA
   
   typedef double Real;
   typedef tnlVector< Real, tnlHost > HostVector;
   typedef tnlVector< Real, tnlCuda > CudaVector;

   
   /****
    * The first argument of this program is the size od data set to be reduced.
    * If no argument is given we use hardcoded default value.
    */
   int size = 1 << 22;
   if( argc > 1 )
      size = atoi( argv[ 1 ] );
   int loops = 10;
   if( argc > 2 )
      loops = atoi( argv[ 2 ] );
   
   
   
   const double oneGB = 1024.0 * 1024.0 * 1024.0;
   double datasetSize = ( double ) ( loops * size ) * sizeof( Real ) / oneGB;
   
   HostVector hostVector, hostVector2;
   CudaVector deviceVector, deviceVector2;
   hostVector.setSize( size );
   if( ! deviceVector.setSize( size ) )
      return EXIT_FAILURE;
   hostVector2.setLike( hostVector );
   if( ! deviceVector2.setLike( deviceVector ) )
      return EXIT_FAILURE;

   hostVector.setValue( 1.0 );
   deviceVector.setValue( 1.0 );
   hostVector2.setValue( 1.0 );
   deviceVector2.setValue( 1.0 );

   tnlTimerRT timer;
   double bandwidth( 0.0 );

   /*   
   cout << "Benchmarking CPU-GPU memory bandwidth: ";
   timer.reset();
   timer.start();
   for( int i = 0; i < loops; i++ )
     deviceVector = hostVector;
   timer.stop();    
   bandwidth = datasetSize / timer.getTime();
   cout << bandwidth << " GB/sec." << endl;
    
   cout << "Benchmarking vector addition on CPU: ";
   timer.reset();
   timer.start();
   for( int i = 0; i < loops; i++ )
     hostVector.addVector( hostVector2 );
   timer.stop();
   bandwidth = 2 * datasetSize / timer.getTime();
   cout << bandwidth << " GB/sec." << endl;
    
    cout << "Benchmarking vector addition on GPU: ";
    timer.reset();
    timer.start();
    for( int i = 0; i < loops; i++ )
      deviceVector.addVector( deviceVector2 );
    cudaThreadSynchronize();
    timer.stop();
    bandwidth = 3 * datasetSize / timer.getTime();
    cout << bandwidth << " GB/sec." << endl;
    */

   Real resultHost, resultDevice, timeHost, timeDevice;

   cout << "Benchmarking scalar product on CPU: ";
   timer.reset();
   timer.start();
   for( int i = 0; i < loops; i++ )
     resultHost = hostVector.scalarProduct( hostVector2 );
   timer.stop();
   timeHost = timer.getTime();
   bandwidth = 2 * datasetSize / timer.getTime();
   cout << "bandwidth: " << bandwidth << " GB/sec, time: " << timer.getTime() << " sec." << endl;
    
   cout << "Benchmarking scalar product on GPU: ";
   timer.reset();
   timer.start();
   for( int i = 0; i < loops; i++ )
      resultDevice = deviceVector.scalarProduct( deviceVector2 );
   timer.stop();
   timeDevice = timer.getTime();
   bandwidth = 2 * datasetSize / timer.getTime();
   cout << "bandwidth: " << bandwidth << " GB/sec, time: " << timer.getTime() << " sec." << endl;
   cout << "CPU/GPU speedup: " << timeHost / timeDevice << endl;

   if( resultHost != resultDevice )
   {
      cerr << "Error. " << resultHost << " != " << resultDevice << endl;
      //return EXIT_FAILURE;
   }

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
   cout << "Time: " << timer.getTime() << " bandwidth: " << bandwidth << " GB/sec." << endl;
#endif    
#endif
   

   cout << "Benchmarking lpNorm on CPU: ";
   timer.reset();
   timer.start();
   for( int i = 0; i < loops; i++ )
     resultHost = hostVector.lpNorm( 2.0 );
   timer.stop();
   timeHost = timer.getTime();
   bandwidth = datasetSize / timer.getTime();
   cout << "bandwidth: " << bandwidth << " GB/sec, time: " << timer.getTime() << " sec." << endl;
    
   cout << "Benchmarking lpNorm on GPU: ";
   timer.reset();
   timer.start();
   for( int i = 0; i < loops; i++ )
      resultDevice = deviceVector.lpNorm( 2.0 );
   timer.stop();
   timeDevice = timer.getTime();
   bandwidth = datasetSize / timer.getTime();
   cout << "bandwidth: " << bandwidth << " GB/sec, time: " << timer.getTime() << " sec." << endl;
   cout << "CPU/GPU speedup: " << timeHost / timeDevice << endl;

   if( resultHost != resultDevice )
   {
      cerr << "Error. " << resultHost << " != " << resultDevice << endl;
      //return EXIT_FAILURE;
   }


   cout << "Benchmarking prefix-sum on CPU: ";
   timer.reset();
   timer.start();
   hostVector.computePrefixSum();
   timer.stop();
   timeHost = timer.getTime();
   bandwidth = 2 * datasetSize / loops / timer.getTime();
   cout << "bandwidth: " << bandwidth << " GB/sec, time: " << timer.getTime() << " sec." << endl;
   
   cout << "Benchmarking prefix-sum on GPU: ";
   timer.reset();
   timer.start();
   deviceVector.computePrefixSum();
   timer.stop();
   timeDevice = timer.getTime();
   bandwidth = 2 * datasetSize / loops / timer.getTime();
   cout << "bandwidth: " << bandwidth << " GB/sec, time: " << timer.getTime() << " sec." << endl;
   cout << "CPU/GPU speedup: " << timeHost / timeDevice << endl;

   for( int i = 0; i < size; i++ )
      if( hostVector.getElement( i ) != deviceVector.getElement( i ) )
      {
         cerr << "Error in prefix sum at position " << i << ":  " << hostVector.getElement( i ) << " != " << deviceVector.getElement( i ) << endl;
      }

   return EXIT_SUCCESS;
}

#endif /* TNLCUDABENCHMARKS_H_ */
