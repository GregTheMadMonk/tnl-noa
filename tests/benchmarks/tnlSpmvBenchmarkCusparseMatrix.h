/***************************************************************************
                          tnlSpmvBenchmarkCusparseMatrix.h  -  description
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

#ifndef TNLSPMVBENCHMARKCUSPARSEMATRIX_H_
#define TNLSPMVBENCHMARKCUSPARSEMATRIX_H_

#include "tnlSpmvBenchmark.h"
#include <tnlConfig.h>
#ifdef HAVE_CUSPARSE
   #include <cusparse.h>
#endif


template< typename Real, typename Index>
class tnlSpmvBenchmarkCusparseMatrix : public tnlSpmvBenchmark< Real, tnlHost, Index, tnlCSRMatrix >
{
   public:

   void setFileName( const tnlString& fileName );

   bool setup( const tnlCSRMatrix< Real, tnlHost, Index >& matrix );

   void tearDown();

   void runBenchmark( const tnlLongVector< Real, tnlHost, Index >& x,
                      const tnlLongVector< Real, tnlHost, Index >& refB,
                      bool verbose );

   void writeProgress() const;

   void writeToLogTable( ostream& logFile,
                         const double& csrGflops,
                         const tnlString& inputMtxFile,
                         const tnlCSRMatrix< Real, tnlHost, Index >& csrMatrix,
                         bool writeMatrixInfo  ) const;

   void setNonzeroElements( const Index nonzeroElements );

   protected:

   tnlString fileName;

   Index nonzeroElements;
};

template< typename Real, typename Index>
void tnlSpmvBenchmarkCusparseMatrix< Real, Index > :: setFileName( const tnlString& fileName )
{
   this -> fileName = fileName;
}

template< typename Real, typename Index>
bool tnlSpmvBenchmarkCusparseMatrix< Real, Index > :: setup( const tnlCSRMatrix< Real, tnlHost, Index >& matrix )
{
   return true;
}

template< typename Real,
          typename Index>
void tnlSpmvBenchmarkCusparseMatrix< Real, Index > :: tearDown()
{

}

template< typename Real,
          typename Index>
void tnlSpmvBenchmarkCusparseMatrix< Real, Index > :: runBenchmark( const tnlLongVector< Real, tnlHost, Index >& _x,
                                                                    const tnlLongVector< Real, tnlHost, Index >& refB,
                                                                    bool verbose )
{
   this -> benchmarkWasSuccesful = false;
#ifdef HAVE_CUSPARSE
   try
   {
#else
   this -> benchmarkWasSuccesful = false;
#endif
   writeProgress();
}

template< typename Real,
          typename Index >
void tnlSpmvBenchmarkCusparseMatrix< Real, Index > :: writeProgress() const
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
void tnlSpmvBenchmarkCusparseMatrix< Real, Index > :: writeToLogTable( ostream& logFile,
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


template< typename Real,
          typename Index >
void tnlSpmvBenchmarkCusparseMatrix< Real, Index > :: setNonzeroElements( const Index nonzeroElements )
{
   this -> nonzeroElements = nonzeroElements;
}

#endif /* TNLSPMVBENCHMARKCUSPARSEMATRIX_H_ */
