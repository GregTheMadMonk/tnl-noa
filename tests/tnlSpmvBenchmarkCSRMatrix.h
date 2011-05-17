/***************************************************************************
                          tnlSpmvBenchmarkCSRMatrix.h  -  description
                             -------------------
    begin                : May 15, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
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

#ifndef TNLSPMVBENCHMARKCSRMATRIX_H_
#define TNLSPMVBENCHMARKCSRMATRIX_H_

#include <tnlSpmvBenchmark.h>
#include <matrix/tnlCSRMatrix.h>

template< typename Real, typename Index>
class tnlSpmvBenchmarkCSRMatrix : public tnlSpmvBenchmark< Real, tnlHost, Index, tnlCSRMatrix >
{
   public:

   bool setup( const tnlCSRMatrix< Real, tnlHost, Index >& matrix );

   void tearDown();

   void writeProgress() const;
};

template< typename Real, typename Index>
bool tnlSpmvBenchmarkCSRMatrix< Real, Index > :: setup( const tnlCSRMatrix< Real, tnlHost, Index >& matrix )
{
   this -> matrix = matrix;
   return true;
}

template< typename Real, typename Index>
void tnlSpmvBenchmarkCSRMatrix< Real, Index > :: tearDown()
{
   this -> matrix. setSize( 0 );
}

template< typename Real,
          typename Index >
void tnlSpmvBenchmarkCSRMatrix< Real, Index > :: writeProgress() const
{
   cout << left << setw( this -> formatColumnWidth ) << "CSR";
   //   cout << left << setw( 25 ) << matrixFormat << setw( 5 ) << cudaBlockSize;
   cout << right << setw( this -> timeColumnWidth ) << setprecision( 2 ) << this -> getTime()
        << right << setw( this -> iterationsColumnWidth ) << this -> getIterations()
        << right << setw( this -> gflopsColumnWidth ) << setprecision( 2 ) << this -> getGflops();
   if( this -> getBenchmarkWasSuccesful() )
        cout << right << setw( this -> benchmarkStatusColumnWidth ) << "OK ";
   else
        cout << right << setw( this -> benchmarkStatusColumnWidth ) << "FAILED ";
   cout << endl;
}

#endif /* TNLSPMVBENCHMARKCSRMATRIX_H_ */
