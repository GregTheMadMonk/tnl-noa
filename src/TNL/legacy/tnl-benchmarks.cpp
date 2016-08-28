/***************************************************************************
                          tnl-benchmarks.cpp  -  description
                             -------------------
    begin                : Nov 25, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/TimerRT.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorCUDA.h>
#include <TNL/tnl-cuda-kernels.cu.h>
#include <TNL/tnl-benchmarks.h>


int main( int argc, char* argv[] )
{
   std::cout << "Benchmarking memory bandwidth when transfering int ..." << std::endl;

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


   std::cout << "Benchmarking reduction of int ..." << std::endl;
   for( int i = 0; i <= 6; i ++ )
      reductionBenchmark< int >( size, i );

   std::cout << "Benchmarking reduction of float ..." << std::endl;
   for( int i = 0; i <= 6; i ++ )
      reductionBenchmark< float >( size, i );

   std::cout << "Benchmarking reduction of double ..." << std::endl;
   for( int i = 0; i <= 6; i ++ )
      reductionBenchmark< double >( size / 2, i );

   return EXIT_SUCCESS;
}
