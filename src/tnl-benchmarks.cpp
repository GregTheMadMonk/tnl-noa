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


int main( int argc, char* argv[] )
{
   cout << "Benchmarking memory bandwidth when transfering data ..." << endl;

   const long int size = 1 << 24;
   tnlLongVector< int > host_vector( size );
   tnlLongVector< int > host_vector2( size );
   tnlLongVectorCUDA< int > device_vector( size );
   tnlLongVectorCUDA< int > device_vector2( size );
   const long int cycles = 100;
   long int bytes = cycles * size * sizeof( int );

   tnlTimerRT timer;
   timer. Reset();
   for( int i = 0; i < cycles; i ++ )
      host_vector2. copyFrom( host_vector );
   double time = timer. GetTime();
   double giga_byte = ( double ) ( 1 << 30 );
   const double host_to_host_band_width = bytes / giga_byte / time;

   cout << "Transfering " << bytes << " bytes from HOST to HOST took " << time << " seconds. Bandwidth is " << host_to_host_band_width << " GB/s." << endl;

   timer. Reset();
   for( int i = 0; i < cycles; i ++ )
         device_vector. copyFrom( host_vector );
   time = timer. GetTime();
   const double host_to_device_band_width = bytes / giga_byte / time;

   cout << "Transfering " << bytes << " bytes from HOST to DEVICE took " << time << " seconds. Bandwidth is " << host_to_device_band_width << " GB/s." << endl;

   timer. Reset();
   for( int i = 0; i < cycles; i ++ )
         host_vector. copyFrom( device_vector );
   time = timer. GetTime();
   const double device_to_host_band_width = bytes / giga_byte / time;

   cout << "Transfering " << bytes << " bytes from DEVICE to HOST took " << time << " seconds. Bandwidth is " << device_to_host_band_width << " GB/s." << endl;

   timer. Reset();
   for( int i = 0; i < cycles; i ++ )
      if( ! device_vector2. copyFrom( device_vector ) )
         return false;

   time = timer. GetTime();
   const double device_to_device_band_width = bytes / giga_byte / time;

   cout << "Transfering " << bytes << " bytes from DEVICE to DEVICE took " << time << " seconds. Bandwidth is " << device_to_device_band_width << " GB/s." << endl;

}
