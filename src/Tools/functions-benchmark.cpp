/***************************************************************************
                          functions-benchmark.cpp  -  description
                             -------------------
    begin                : Jul 4, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include "functions-benchmark.h"

int main( int argc, char* argv[] )
{
   const long int loops = 1 << 24;

   std::cout << "Runnning benchmarks in single precision on CPU ... " << std::endl;
   benchmarkAddition< float >( loops );
   benchmarkMultiplication< float >( loops );
   benchmarkDivision< float >( loops );
   benchmarkSqrt< float >( loops );
   benchmarkSin< float >( loops );
   benchmarkExp< float >( loops );
   benchmarkPow< float >( loops );

   std::cout << "Runnning benchmarks in double precision on CPU ... " << std::endl;
   benchmarkAddition< double >( loops );
   benchmarkMultiplication< float >( loops );
   benchmarkDivision< double >( loops );
   benchmarkSqrt< double >( loops );
   benchmarkSin< double >( loops );
   benchmarkExp< double >( loops );
   benchmarkPow< double >( loops );



   return EXIT_SUCCESS;
}
