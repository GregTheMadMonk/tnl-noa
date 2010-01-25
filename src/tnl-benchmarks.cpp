/***************************************************************************
                          tnl-benchmarks.cpp  -  description
                             -------------------
    begin                : Nov 25, 2010
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

#include <core/tnlTimerRT.h>
#include <core/tnlLongVector.h>
#include <core/tnlLongVectorCUDA.h>
#include <core/tnl-cuda-kernels.cu.h>

void reductionBenchmark( const int size,
                         tnlLongVectorCUDA< int >& device_vector,
                         const int algorithm )
{
   const long int seq_sum = size * ( size - 1 ) / 2;
   const long int seq_max = size;
   const long int seq_min = 1;

   int sum, min, max;
   const long int sorting_cycles( 10 );

   tnlTimerRT timer;
   timer. Reset();
   for( int i = 0; i < sorting_cycles; i ++ )
   {
      switch( algorithm )
      {
         case 1:
            tnlCUDASimpleReduction1Sum( size,
                                        device_vector. Data(),
                                        sum );
            tnlCUDASimpleReduction1Min( size,
                                        device_vector. Data(),
                                        min );
            tnlCUDASimpleReduction1Max( size,
                                        device_vector. Data(),
                                        max );
            break;
         case 2:
            tnlCUDASimpleReduction2Sum( size,
                                        device_vector. Data(),
                                        sum );
            tnlCUDASimpleReduction2Min( size,
                                        device_vector. Data(),
                                        min );
            tnlCUDASimpleReduction2Max( size,
                                        device_vector. Data(),
                                        max );
            break;
         case 3:
            tnlCUDASimpleReduction3Sum( size,
                                        device_vector. Data(),
                                        sum );
            tnlCUDASimpleReduction3Min( size,
                                        device_vector. Data(),
                                        min );
            tnlCUDASimpleReduction3Max( size,
                                        device_vector. Data(),
                                        max );
            break;
         case 4:
            tnlCUDASimpleReduction4Sum( size,
                                        device_vector. Data(),
                                        sum );
            tnlCUDASimpleReduction4Min( size,
                                        device_vector. Data(),
                                        min );
            tnlCUDASimpleReduction4Max( size,
                                        device_vector. Data(),
                                        max );
            break;
         case 5:
            tnlCUDASimpleReduction5Sum( size,
                                        device_vector. Data(),
                                        sum );
            tnlCUDASimpleReduction5Min( size,
                                        device_vector. Data(),
                                        min );
            tnlCUDASimpleReduction5Max( size,
                                        device_vector. Data(),
                                        max );
            break;

      }
   }
   const double time = timer. GetTime();
   double giga_byte = ( double ) ( 1 << 30 );
   long int mega_byte = 1 << 20;
   long int bytes_reduced = size * sizeof( int ) * sorting_cycles * 3;
   const double reduction_band_width = bytes_reduced / giga_byte / time;
   cout << "Reducing " << bytes_reduced / mega_byte
        << " MB on DEVICE using algorithm " << algorithm
        << " took " << time
        << " seconds. Bandwidth is " << reduction_band_width
        << " GB/s." << endl;

}

int main( int argc, char* argv[] )
{
   cout << "Benchmarking memory bandwidth when transfering data ..." << endl;

   const long int size = 1 << 20;
   tnlLongVector< int > host_vector( size );
   tnlLongVector< int > host_vector2( size );
   tnlLongVectorCUDA< int > device_vector( size );
   tnlLongVectorCUDA< int > device_vector2( size );

   for( int i = 0; i < size; i ++ )
      host_vector[ i ] = i + 1;

   const long int cycles = 100;
   long int bytes = cycles * size * sizeof( int );
   long int mega_byte = 1 << 20;

   tnlTimerRT timer;
   timer. Reset();
   for( int i = 0; i < cycles; i ++ )
      host_vector2. copyFrom( host_vector );
   double time = timer. GetTime();
   double giga_byte = ( double ) ( 1 << 30 );
   const double host_to_host_band_width = bytes / giga_byte / time;

   cout << "Transfering " << bytes / mega_byte << " MB from HOST to HOST took " << time << " seconds. Bandwidth is " << host_to_host_band_width << " GB/s." << endl;

   timer. Reset();
   for( int i = 0; i < cycles; i ++ )
         device_vector. copyFrom( host_vector );
   time = timer. GetTime();
   const double host_to_device_band_width = bytes / giga_byte / time;

   cout << "Transfering " << bytes / mega_byte << " MB from HOST to DEVICE took " << time << " seconds. Bandwidth is " << host_to_device_band_width << " GB/s." << endl;

   timer. Reset();
   for( int i = 0; i < cycles; i ++ )
         host_vector2. copyFrom( device_vector );
   time = timer. GetTime();
   const double device_to_host_band_width = bytes / giga_byte / time;

   cout << "Transfering " << bytes / mega_byte << " MB from DEVICE to HOST took " << time << " seconds. Bandwidth is " << device_to_host_band_width << " GB/s." << endl;

   timer. Reset();
   for( int i = 0; i < cycles; i ++ )
      if( ! device_vector2. copyFrom( device_vector ) )
         return false;

   time = timer. GetTime();
   const double device_to_device_band_width = bytes / giga_byte / time;

   cout << "Transfering " << bytes / mega_byte << " MB from DEVICE to DEVICE took " << time << " seconds. Bandwidth is " << device_to_device_band_width << " GB/s." << endl;

   for( int i = 1; i <= 4; i ++ )
      reductionBenchmark( size, device_vector, i );

}
