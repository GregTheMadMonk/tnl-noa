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
#include <tnl-benchmarks.h>


int main( int argc, char* argv[] )
{
   cout << "Benchmarking memory bandwidth when transfering int ..." << endl;

   const int size = 1 << 22;
   double host_to_host_band_width;
   double host_to_device_band_width;
   double device_to_host_band_width;
   double device_to_device_band_width;

   transferBenchmark< int >( size,
                             host_to_host_band_width,
                             host_to_device_band_width,
                             device_to_host_band_width,
                             device_to_device_band_width );


   cout << "Benchmarking reduction of int ..." << endl;
   for( int i = 0; i <= 6; i ++ )
      reductionBenchmark< int >( size, i );

   cout << "Benchmarking reduction of float ..." << endl;
   for( int i = 0; i <= 6; i ++ )
      reductionBenchmark< float >( size, i );

   cout << "Benchmarking reduction of double ..." << endl;
   for( int i = 0; i <= 6; i ++ )
      reductionBenchmark< double >( size / 2, i );

   return EXIT_SUCCESS;
}
