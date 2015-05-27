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

template< class T >
bool transferBenchmark( const int size,
                        double& host_to_host_band_width,
                        double& host_to_device_band_width,
                        double& device_to_host_band_width,
                        double& device_to_device_band_width )
{

  tnlVector< T > host_vector( "transferBenchmark:host-vector", size );
  tnlVector< T > host_vector2( "transferBenchmark:host-vector-2", size );
  tnlVectorCUDA< T > device_vector( "transferBenchmark:device-vector", size );
  tnlVectorCUDA< T > device_vector2( "transferBenchmark:device-vector-2", size );

   for( int i = 0; i < size; i ++ )
      host_vector[ i ] = i + 1;

   const long int cycles = 100;
   long int bytes = cycles * size * sizeof( int );
   long int mega_byte = 1 << 20;

   tnlTimerRT timer;
   timer. Reset();
   for( int i = 0; i < cycles; i ++ )
      if( ! host_vector2. copyFrom( host_vector ) )
         return false;
   double time = timer. getTime();
   double giga_byte = ( double ) ( 1 << 30 );
   host_to_host_band_width = bytes / giga_byte / time;

   cout << "Transfering " << bytes / mega_byte << " MB from HOST to HOST took " << time << " seconds. Bandwidth is " << host_to_host_band_width << " GB/s." << endl;

   timer. Reset();
   for( int i = 0; i < cycles; i ++ )
      if( ! device_vector. copyFrom( host_vector ) )
         return false;
   time = timer. getTime();
   host_to_device_band_width = bytes / giga_byte / time;

   cout << "Transfering " << bytes / mega_byte << " MB from HOST to DEVICE took " << time << " seconds. Bandwidth is " << host_to_device_band_width << " GB/s." << endl;

   timer. Reset();
   for( int i = 0; i < cycles; i ++ )
      if( ! host_vector2. copyFrom( device_vector ) )
         return false;
   time = timer. getTime();
   device_to_host_band_width = bytes / giga_byte / time;

   cout << "Transfering " << bytes / mega_byte << " MB from DEVICE to HOST took " << time << " seconds. Bandwidth is " << device_to_host_band_width << " GB/s." << endl;

   timer. Reset();
   for( int i = 0; i < cycles; i ++ )
      if( ! device_vector2. copyFrom( device_vector ) )
         return false;

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
   const T* data = host_vector. Data();
   const int size = host_vector. GetSize();
   sum = 0.0;
   for( int i = 0; i < size; i ++ )
      sum += data[ i ];
};

template< class T >
void tnlCPUReductionMin( const tnlVector< T >& host_vector,
                         T& min )
{
   const T* data = host_vector. Data();
   const int size = host_vector. GetSize();
   //tnlAssert( data );
   min = data[ 0 ];
   for( int i = 1; i < size; i ++ )
      min = :: Min( min,  data[ i ] );
};

template< class T >
void tnlCPUReductionMax( const tnlVector< T >& host_vector,
                         T& max )
{
   const T* data = host_vector. Data();
   const int size = host_vector. GetSize();
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
   tnlVectorCUDA< T > device_vector( "reductionBenchmark:device-vector", size );
   tnlVectorCUDA< T > device_aux( "reductionBenchmark:device-aux", size / 2 );

   for( int i = 0; i < size; i ++ )
      host_vector[ i ] = i + 1;

   device_vector. copyFrom( host_vector );

   T sum, min, max;
   const long int reducing_cycles( 10 );

   tnlTimerRT timer;
   timer. Reset();
   for( int i = 0; i < reducing_cycles; i ++ )
   {
      switch( algorithm )
      {
         case 0:  // reduction on CPU
            tnlCPUReductionSum( host_vector, sum );
            tnlCPUReductionMin( host_vector, sum );
            tnlCPUReductionMax( host_vector, sum );
         case 1:
            tnlCUDASimpleReduction1Sum( size,
                                        device_vector. Data(),
                                        sum,
                                        device_aux. Data() );
            tnlCUDASimpleReduction1Min( size,
                                        device_vector. Data(),
                                        min,
                                        device_aux. Data() );
            tnlCUDASimpleReduction1Max( size,
                                        device_vector. Data(),
                                        max,
                                        device_aux. Data() );
            break;
         case 2:
            tnlCUDASimpleReduction2Sum( size,
                                        device_vector. Data(),
                                        sum,
                                        device_aux. Data() );
            tnlCUDASimpleReduction2Min( size,
                                        device_vector. Data(),
                                        min,
                                        device_aux. Data() );
            tnlCUDASimpleReduction2Max( size,
                                        device_vector. Data(),
                                        max,
                                        device_aux. Data() );
            break;
         case 3:
            tnlCUDASimpleReduction3Sum( size,
                                        device_vector. Data(),
                                        sum,
                                        device_aux. Data() );
            tnlCUDASimpleReduction3Min( size,
                                        device_vector. Data(),
                                        min,
                                        device_aux. Data() );
            tnlCUDASimpleReduction3Max( size,
                                        device_vector. Data(),
                                        max,
                                        device_aux. Data() );
            break;
         case 4:
            tnlCUDASimpleReduction4Sum( size,
                                        device_vector. Data(),
                                        sum,
                                        device_aux. Data() );
            tnlCUDASimpleReduction4Min( size,
                                        device_vector. Data(),
                                        min,
                                        device_aux. Data() );
            tnlCUDASimpleReduction4Max( size,
                                        device_vector. Data(),
                                        max,
                                        device_aux. Data() );
            break;
         case 5:
            tnlCUDASimpleReduction5Sum( size,
                                        device_vector. Data(),
                                        sum,
                                        device_aux. Data() );
            tnlCUDASimpleReduction5Min( size,
                                        device_vector. Data(),
                                        min,
                                        device_aux. Data() );
            tnlCUDASimpleReduction5Max( size,
                                        device_vector. Data(),
                                        max,
                                        device_aux. Data() );
            break;
         default:
            tnlCUDAReductionSum( size,
                                 device_vector. Data(),
                                 sum,
                                 device_aux. Data() );
            tnlCUDAReductionMin( size,
                                 device_vector. Data(),
                                 min,
                                 device_aux. Data() );
            tnlCUDAReductionMax( size,
                                 device_vector. Data(),
                                 max,
                                 device_aux. Data() );

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
