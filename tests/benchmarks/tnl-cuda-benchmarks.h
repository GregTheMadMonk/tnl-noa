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

#ifndef TNLBENCHMARKS_H_
#define TNLBENCHMARKS_H_

#include <core/mfuncs.h>
#include <core/tnlTimerCPU.h>
#include <../tests/unit-tests/core/tnl-cuda-kernels.h>
#include <core/low-level/cuda-long-vector-kernels.h>

template< class T >
bool transferBenchmark( const int size,
                        double& host_to_host_band_width,
                        double& host_to_device_band_width,
                        double& device_to_host_band_width,
                        double& device_to_device_band_width )
{

  tnlVector< T > host_vector( "transferBenchmark:host-vector", size );
  tnlVector< T > host_vector2( "transferBenchmark:host-vector-2", size );
  tnlVector< T, tnlCuda > device_vector( "transferBenchmark:device-vector", size );
  tnlVector< T, tnlCuda > device_vector2( "transferBenchmark:device-vector-2", size );

   for( int i = 0; i < size; i ++ )
      host_vector[ i ] = i + 1;

   const long int cycles = 100;
   long int bytes = cycles * size * sizeof( int );
   long int mega_byte = 1 << 20;

   tnlTimerCPU timer;
   timer. Reset();
   for( int i = 0; i < cycles; i ++ )
      host_vector2 = host_vector;

   double time = timer. getTime();
   double giga_byte = ( double ) ( 1 << 30 );
   host_to_host_band_width = bytes / giga_byte / time;

   cout << "Transfering " << bytes / mega_byte << " MB from HOST to HOST took " << time << " seconds. Bandwidth is " << host_to_host_band_width << " GB/s." << endl;

   timer. Reset();
   for( int i = 0; i < cycles; i ++ )
      device_vector = host_vector;

   time = timer. getTime();
   host_to_device_band_width = bytes / giga_byte / time;

   cout << "Transfering " << bytes / mega_byte << " MB from HOST to DEVICE took " << time << " seconds. Bandwidth is " << host_to_device_band_width << " GB/s." << endl;

   timer. Reset();
   for( int i = 0; i < cycles; i ++ )
      host_vector2 = device_vector;

   time = timer. getTime();
   device_to_host_band_width = bytes / giga_byte / time;

   cout << "Transfering " << bytes / mega_byte << " MB from DEVICE to HOST took " << time << " seconds. Bandwidth is " << device_to_host_band_width << " GB/s." << endl;

   timer. Reset();
   for( int i = 0; i < cycles; i ++ )
      device_vector2 = device_vector;


   time = timer. getTime();

   // Since we read and write tha data back we process twice as many bytes.
   bytes *= 2;
   device_to_device_band_width = bytes / giga_byte / time;

   cout << "Transfering " << bytes / mega_byte << " MB from DEVICE to DEVICE took " << time << " seconds. Bandwidth is " << device_to_device_band_width << " GB/s." << endl;
}

template< class T >
void tnlCPUReductionSum( const tnlVector< T >& host_vector,
                         T& sum )
{
   const T* data = host_vector. getData();
   const int size = host_vector. getSize();
   sum = 0.0;
   for( int i = 0; i < size; i ++ )
      sum += data[ i ];
};

template< class T >
void tnlCPUReductionMin( const tnlVector< T >& host_vector,
                         T& min )
{
   const T* data = host_vector. getData();
   const int size = host_vector. getSize();
   //tnlAssert( data );
   min = data[ 0 ];
   for( int i = 1; i < size; i ++ )
      min = :: Min( min,  data[ i ] );
};

template< class T >
void tnlCPUReductionMax( const tnlVector< T >& host_vector,
                         T& max )
{
   const T* data = host_vector. getData();
   const int size = host_vector. getSize();
   //tnlAssert( data );
   max = data[ 0 ];
   for( int i = 1; i < size; i ++ )
      max = :: Max( max,  data[ i ] );
};

template< class T >
void reductionBenchmark( const int size,
                         const int algorithm )
{
   tnlVector< T > host_vector( "reductionBenchmark:host-vector", size );
   tnlVector< T, tnlCuda > device_vector( "reductionBenchmark:device-vector", size );
   tnlVector< T, tnlCuda > device_aux( "reductionBenchmark:device-aux", size / 2 );

   for( int i = 0; i < size; i ++ )
      host_vector[ i ] = i + 1;

   device_vector = host_vector;

   T sum, min, max;
   const long int reducing_cycles( 1 );

   tnlTimerCPU timer;
   timer. Reset();
   for( int i = 0; i < reducing_cycles; i ++ )
   {
      switch( algorithm )
      {
         case 0:  // reduction on CPU
            tnlCPUReductionSum( host_vector, sum );
            tnlCPUReductionMin( host_vector, sum );
            tnlCPUReductionMax( host_vector, sum );
#ifdef HAVE_CUDA
         case 1:
            tnlCUDASimpleReduction1< T, tnlParallelReductionSum >( size,
                                                                   device_vector. getData(),
                                                                   sum,
                                                                   device_aux. getData() );
            tnlCUDASimpleReduction1< T, tnlParallelReductionMin >( size,
                                                                   device_vector. getData(),
                                                                   min,
                                                                   device_aux. getData() );
            tnlCUDASimpleReduction1< T, tnlParallelReductionMax >( size,
                                                                   device_vector. getData(),
                                                                   max,
                                                                   device_aux. getData() );
            break;
         case 2:
            tnlCUDASimpleReduction2< T, tnlParallelReductionSum >( size,
                                                                   device_vector. getData(),
                                                                   sum,
                                                                   device_aux. getData() );
            tnlCUDASimpleReduction2< T, tnlParallelReductionMin >( size,
                                                                   device_vector. getData(),
                                                                   min,
                                                                   device_aux. getData() );
            tnlCUDASimpleReduction2< T, tnlParallelReductionMax >( size,
                                                                   device_vector. getData(),
                                                                   max,
                                                                   device_aux. getData() );
            break;
         case 3:
            tnlCUDASimpleReduction3< T, tnlParallelReductionSum >( size,
                                                                   device_vector. getData(),
                                                                   sum,
                                                                   device_aux. getData() );
            tnlCUDASimpleReduction3< T, tnlParallelReductionMin >( size,
                                                                   device_vector. getData(),
                                                                   min,
                                                                   device_aux. getData() );
            tnlCUDASimpleReduction3< T, tnlParallelReductionMax >( size,
                                                                   device_vector. getData(),
                                                                   max,
                                                                   device_aux. getData() );
            break;
         case 4:
            tnlCUDASimpleReduction4< T, tnlParallelReductionSum >( size,
                                                                   device_vector. getData(),
                                                                   sum,
                                                                   device_aux. getData() );
            tnlCUDASimpleReduction4< T, tnlParallelReductionMin >( size,
                                                                   device_vector. getData(),
                                                                   min,
                                                                   device_aux. getData() );
            tnlCUDASimpleReduction4< T, tnlParallelReductionMax >( size,
                                                                   device_vector. getData(),
                                                                   max,
                                                                   device_aux. getData() );
            break;
         case 5:
            tnlCUDASimpleReduction5< T, tnlParallelReductionSum >( size,
                                                                   device_vector. getData(),
                                                                   sum,
                                                                   device_aux. getData() );
            tnlCUDASimpleReduction5< T, tnlParallelReductionMin >( size,
                                                                   device_vector. getData(),
                                                                   min,
                                                                   device_aux. getData() );
            tnlCUDASimpleReduction5< T, tnlParallelReductionMax >( size,
                                                                   device_vector. getData(),
                                                                   max,
                                                                   device_aux. getData() );
            break;
         default:
            reductionOnCudaDevice< T, T, int, tnlParallelReductionSum >( size,
                                                                              device_vector. getData(),
                                                                              NULL,
                                                                              sum,
                                                                              0.0,
                                                                              device_aux. getData() );
            reductionOnCudaDevice< T, T, int, tnlParallelReductionMin >( size,
                                                                              device_vector. getData(),
                                                                              NULL,
                                                                              min,
                                                                              0.0,
                                                                              device_aux. getData() );
            reductionOnCudaDevice< T, T, int, tnlParallelReductionMax >( size,
                                                                              device_vector. getData(),
                                                                              NULL,
                                                                              max,
                                                                              0.0,
                                                                              device_aux. getData() );
#endif

      }
   }
   const double time = timer. getTime();
   double giga_byte = ( double ) ( 1 << 30 );
   long int mega_byte = 1 << 20;
   long int bytes_reduced = size * sizeof( T ) * reducing_cycles * 3;
   const double reduction_band_width = bytes_reduced / giga_byte / time;

   cout << "Reducing " << bytes_reduced / mega_byte
        << " MB on DEVICE using algorithm " << algorithm
        << " took " << time
        << " seconds. Bandwidth is " << reduction_band_width
        << " GB/s." << endl;
}

#endif /* TNLBENCHMARKS_H_ */