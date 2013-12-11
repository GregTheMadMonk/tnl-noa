/***************************************************************************
                          tnlSpmvBenchmarkCusparseCSRMatrix.h  -  description
                             -------------------
    begin                : Feb 16, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
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

#ifndef TNLSPMVBENCHMARKCUSPARSECSRMATRIX_H_
#define TNLSPMVBENCHMARKCUSPARSECSRMATRIX_H_

#include "tnlSpmvBenchmark.h"
#include <tnlConfig.h>
#include <legacy/matrices/tnlCusparseCSRMatrix.h>

template< typename Real, typename Index>
class tnlSpmvBenchmarkCusparseCSRMatrix : public tnlSpmvBenchmark< Real, tnlCuda, Index, tnlCusparseCSRMatrix >
{
   public:
   tnlSpmvBenchmarkCusparseCSRMatrix();

   bool setup( const tnlCSRMatrix< Real, tnlHost, Index >& matrix );

   void tearDown();

   Index getArtificialZeros() const;

   void writeProgress() const;

   void writeToLogTable( ostream& logFile,
                         const double& csrGflops,
                         const tnlString& inputMtxFile,
                         const tnlCSRMatrix< Real, tnlHost, Index >& csrMatrix,
                         bool writeMatrixInfo  ) const;

   void setNonzeroElements( const Index nonzeroElements );
};

template< typename Real, typename Index>
bool tnlSpmvBenchmarkCusparseCSRMatrix< Real, Index > :: setup( const tnlCSRMatrix< Real, tnlHost, Index >& matrix )
{
   if( ! this -> matrix. copyFrom( matrix ) )
      return false;
   this -> setupOk = true;
   return true;
}

template< typename Real,
          typename Index>
void tnlSpmvBenchmarkCusparseCSRMatrix< Real, Index > :: tearDown()
{
   this -> matrix. reset();
}

template< typename Real,
          typename Index>
Index tnlSpmvBenchmarkCusparseCSRMatrix< Real, Index > :: getArtificialZeros() const
{
   return 0;
}

template< typename Real,
          typename Index >
void tnlSpmvBenchmarkCusparseCSRMatrix< Real, Index > :: writeProgress() const
{
   cout << left << setw( this -> formatColumnWidth ) << "Cusparse";
   //   cout << left << setw( 25 ) << matrixFormat << setw( 5 ) << cudaBlockSize;
   cout << right << setw( this -> timeColumnWidth ) << setprecision( 2 ) << this -> getTime()
        << right << setw( this -> iterationsColumnWidth ) << this -> getIterations()
        << right << setw( this -> gflopsColumnWidth ) << setprecision( 2 ) << this -> getGflops();
   if( this -> getBenchmarkWasSuccesful() )
        cout << right << setw( this -> benchmarkStatusColumnWidth ) << "OK ";
   else
        cout << right << setw( this -> benchmarkStatusColumnWidth ) << "  FAILED - maxError is " << this -> maxError << ". ";
#ifndef HAVE_CUSP
   cout << "CUSPARSE library is missing.";
#endif
   cout << endl;
}

template< typename Real,
          typename Index >
tnlSpmvBenchmarkCusparseCSRMatrix< Real, Index > :: tnlSpmvBenchmarkCusparseCSRMatrix()
{

}

template< typename Real,
          typename Index >
void tnlSpmvBenchmarkCusparseCSRMatrix< Real, Index > :: writeToLogTable( ostream& logFile,
                                                                       const double& csrGflops,
                                                                       const tnlString& inputMtxFile,
                                                                       const tnlCSRMatrix< Real, tnlHost, Index >& csrMatrix,
                                                                       bool writeMatrixInfo  ) const
{
   if( this -> getBenchmarkWasSuccesful() )
   {
      double speedUp = this -> getGflops() / csrGflops;
      tnlString bgColor = this -> getBgColorBySpeedUp( speedUp );
      logFile << "             <td bgcolor=" << bgColor << ">" << this -> getTime() << "</td>" << endl;
      logFile << "             <td bgcolor=" << bgColor << ">" << this -> getGflops() << "</td>" << endl;

      logFile << "             <td bgcolor=" << bgColor << "> " << speedUp << "</td>" << endl;
   }
   else
   {
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
   }
}

#endif /* TNLSPMVBENCHMARKCUSPARSECSRMATRIX_H_ */
