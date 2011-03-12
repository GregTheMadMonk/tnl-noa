/***************************************************************************
                          functions-benchmark.cpp  -  description
                             -------------------
    begin                : Jul 4, 2010
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
#include <core/tnlTimerCPU.h>
#include "functions-benchmark.h"

int main( int argc, char* argv[] )
{
   const long int loops = 1 << 24;

   cout << "Runnning benchmarks in single precision on CPU ... " << endl;
   benchmarkAddition< float >( loops );
   benchmarkMultiplication< float >( loops );
   benchmarkDivision< float >( loops );
   benchmarkSqrt< float >( loops );
   benchmarkSin< float >( loops );
   benchmarkExp< float >( loops );
   benchmarkPow< float >( loops );

   cout << "Runnning benchmarks in double precision on CPU ... " << endl;
   benchmarkAddition< double >( loops );
   benchmarkMultiplication< float >( loops );
   benchmarkDivision< double >( loops );
   benchmarkSqrt< double >( loops );
   benchmarkSin< double >( loops );
   benchmarkExp< double >( loops );
   benchmarkPow< double >( loops );



   return EXIT_SUCCESS;
}
