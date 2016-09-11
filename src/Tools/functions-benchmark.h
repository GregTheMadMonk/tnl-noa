/***************************************************************************
                          functions-benchmark.h  -  description
                             -------------------
    begin                : Jul 4, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef FUNCTIONSBENCHMARK_H_
#define FUNCTIONSBENCHMARK_H_

#include <iostream>
#include <math.h>

#include <TNL/TimerRT.h>
#include <TNL/TimerCPU.h>

using namespace TNL;

template< typename REAL > void benchmarkAddition( long int loops )
{
   std::cout << "Benchmarking addition on CPU ( " << loops << " loops ) ... " << std::flush;
   TimerCPU cpu_timer;

   REAL a1 = 1.2;
   REAL a2 = 1.2;
   REAL a3 = 1.2;
   REAL a4 = 1.2;
   for( long int i = 0; i < loops; i ++ )
   {
      a1 += REAL( 0.1 );
      a2 += REAL( 0.1 );
      a3 += REAL( 0.1 );
      a4 += REAL( 0.1 );
   }

   double cpu_time = cpu_timer. getTime();
   std::cout << " ( " << a1 + a2 + a3 + a4 << " ) " <<  cpu_time << "secs. " << 4.0 * ( ( double ) loops ) / cpu_time * 1.0e-9 << " GFLOPS." << std::endl;
}

template< typename REAL > void benchmarkMultiplication( const long int loops )
{
   std::cout << "Benchmarking multiplication on CPU ( " << loops << " loops ) ... " << std::flush;
   TimerCPU cpu_timer;

   REAL a1 = 1.0e9;
   REAL a2 = 1.0e9;
   REAL a3 = 1.0e9;
   REAL a4 = 1.0e9;
   for( long int i = 0; i < loops; i ++ )
   {
      {
         a1 *= REAL( 0.99 );
         a2 *= REAL( 0.99 );
         a3 *= REAL( 0.99 );
         a4 *= REAL( 0.99 );
         if( a1 < REAL( 0.01 ) ) a1 = a2 = a3 = a4 = REAL( 1.0e9 );
      }
   }

   double cpu_time = cpu_timer. getTime();
   std::cout << " ( " << a1 * a2 * a3 * a4 << " ) " <<  cpu_time << "secs. " << 4.0 * ( ( double ) loops ) / cpu_time * 1.0e-9 << " GFLOPS." << std::endl;
}

template< typename REAL > void benchmarkDivision( long int loops )
{
   std::cout << "Benchmarking division on CPU ( " << loops << " loops ) ... " << std::flush;
   TimerCPU cpu_timer;

   REAL a1( 1.0e9 );
   REAL a2( 1.0e9 );
   REAL a3( 1.0e9 );
   REAL a4( 1.0e9 );
   for( long int i = 0; i < loops; i ++ )
   {
      a1 /= REAL( 1.1 );
      a2 /= REAL( 1.1 );
      a3 /= REAL( 1.1 );
      a4 /= REAL( 1.1 );
      if( a1 < REAL( 0.01 ) ) a1 = a2 = a3 = a4 = REAL( 1.0e9 );
   }

   double cpu_time = cpu_timer. getTime();
   std::cout << " ( " << a1 / a2 / a3 / a4 << " ) " << cpu_time << "secs. " << 4.0 * ( ( double ) loops / 2 ) / cpu_time * 1.0e-9 << " GFLOPS." << std::endl;
}

template< typename REAL > void benchmarkSqrt( long int loops )
{
   std::cout << "Benchmarking sqrt on CPU ( " << loops << " loops ) ... " << std::flush;
   TimerCPU cpu_timer;

   REAL a1( 1.0e9 );
   REAL a2( 1.0e9 );
   REAL a3( 1.0e9 );
   REAL a4( 1.0e9 );
   for( long int i = 0; i < loops; i ++ )
   {
      a1 = ::sqrt( a1 );
      a2 = ::sqrt( a2 );
      a3 = ::sqrt( a3 );
      a4 = ::sqrt( a4 );
      if( a1 < REAL( 100.0 ) ) a1 = a2 = a3 = a4 = REAL( 1.0e9 );
   }

   double cpu_time = cpu_timer. getTime();
   std::cout << " ( " << a1 + a2 + a3 + a4 << " ) " << cpu_time << "secs. " << 4.0 * ( ( double ) loops / 2 ) / cpu_time * 1.0e-9 << " GFLOPS." << std::endl;
}

template< typename REAL > void benchmarkSin( long int loops )
{
   std::cout << "Benchmarking sin on CPU ( " << loops << " loops ) ... " << std::flush;
   TimerCPU cpu_timer;

   REAL a1( 1.0e9 );
   REAL a2( 1.0e9 );
   REAL a3( 1.0e9 );
   REAL a4( 1.0e9 );
   for( long int i = 0; i < loops; i ++ )
   {
      a1 = ::sin( a1 );
      a2 = ::sin( a2 );
      a3 = ::sin( a3 );
      a4 = ::sin( a4 );
   }

   double cpu_time = cpu_timer. getTime();
   std::cout << " ( " << a1 + a2 + a3 + a4 << " ) " << cpu_time << "secs. " << 4.0 * ( ( double ) loops ) / cpu_time * 1.0e-9 << " GFLOPS." << std::endl;
}

template< typename REAL > void benchmarkExp( long int loops )
{
   std::cout << "Benchmarking exp on CPU ( " << loops << " loops ) ... " << std::flush;
   TimerCPU cpu_timer;

   REAL a1( 1.1 );
   REAL a2( 1.1 );
   REAL a3( 1.1 );
   REAL a4( 1.1 );
   for( long int i = 0; i < loops; i ++ )
   {
      a1 = exp( a1 );
      a2 = exp( a2 );
      a3 = exp( a3 );
      a4 = exp( a4 );
      if( a1 > REAL( 1.0e9 ) ) a1 = a2 = a3 = a4 = REAL( 1.1 );
   }

   double cpu_time = cpu_timer. getTime();
   std::cout << " ( " << a1 + a2 + a3 + a4 << " ) " << cpu_time << "secs. " << 4.0 * ( ( double ) loops) / cpu_time * 1.0e-9 << " GFLOPS." << std::endl;
}

template< typename REAL > void benchmarkPow( long int loops )
{
   std::cout << "Benchmarking pow on CPU ( " << loops << " loops ) ... " << std::flush;
   TimerCPU cpu_timer;

   REAL a1( 1.0e9 );
   REAL a2( 1.0e9 );
   REAL a3( 1.0e9 );
   REAL a4( 1.0e9 );
   for( long int i = 0; i < loops; i ++ )
   {
      a1 = ::pow( a1, REAL( 0.9 ) );
      a2 = ::pow( a2, REAL( 0.9 ) );
      a3 = ::pow( a3, REAL( 0.9 ) );
      a4 = ::pow( a4, REAL( 0.9 ) );
      if( a1 < REAL( 1.0 ) ) a1 = a2 = a3 = a4 = REAL( 1.0e9 );
   }

   double cpu_time = cpu_timer. getTime();
   std::cout << " ( " << a1 + a2 + a3 + a4 << " ) " << cpu_time << "secs. " << 4.0 * ( ( double ) loops) / cpu_time * 1.0e-9 << " GFLOPS." << std::endl;
}


#endif /* FUNCTIONSBENCHMARK_H_ */
