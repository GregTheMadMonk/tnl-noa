/***************************************************************************
                          tnlSpmvBenchmarkCSRMatrix.h  -  description
                             -------------------
    begin                : May 15, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLSPMVBENCHMARKCSRMATRIX_H_
#define TNLSPMVBENCHMARKCSRMATRIX_H_

#include "tnlSpmvBenchmark.h"
#include <matrices/tnlCSRMatrix.h>

template< typename Real, typename Index>
class tnlSpmvBenchmarkCSRMatrix : public tnlSpmvBenchmark< Real, tnlHost, Index, tnlCSRMatrix >
{
   public:

   bool setup( const tnlCSRMatrix< Real, tnlHost, Index >& matrix );

   void tearDown();

   void writeProgress() const;

   void writeToLogTable( ostream& logFile,
                         const double& csrGflops,
                         const tnlString& inputMtxFile,
                         const tnlCSRMatrix< Real, tnlHost, Index >& csrMatrix,
                         bool writeMatrixInfo  ) const;
   Real getForwardBackwardDifference() const;

   protected:

   /*!***
    * This measures the difference between SpMV result when used forward or bakward
    * matrix columns ordering.
    */
   Real forwardBackwardDifference;
};

template< typename Real, typename Index>
bool tnlSpmvBenchmarkCSRMatrix< Real, Index > :: setup( const tnlCSRMatrix< Real, tnlHost, Index >& matrix )
{
   this->matrix = matrix;

   const Index size = matrix. getSize();
   tnlVector< Real, tnlHost > refX( "ref-x", size ), refB( "ref-b", size), backwardRefB( "backwardRef-b", size);
   refX. setValue( 1.0 );
   this->matrix. vectorProduct( refX, refB );
   this->matrix. setBackwardSpMV( true );
   this->matrix. vectorProduct( refX, backwardRefB );
   this->matrix. setBackwardSpMV( false );
   Real error( 0.0 ), maxError( 0.0 );
   for( Index j = 0; j < refB. getSize(); j ++ )
   {
      if( refB[ j ] != 0.0 && backwardRefB[ j ] != 0.0 )
         error = ( Real ) fabs( refB[ j ] - backwardRefB[ j ] ) / Min( ( Real ) fabs( refB[ j ] ), ( Real ) fabs( backwardRefB[ j ] ) );
      else
         error = Max( ( Real ) fabs( refB[ j ] ), ( Real ) fabs( backwardRefB[ j ] ) );
      maxError = Max( error, maxError );
   }
   forwardBackwardDifference = maxError;
   this->setupOk = true;
   return true;
}

template< typename Real, typename Index>
void tnlSpmvBenchmarkCSRMatrix< Real, Index > :: tearDown()
{
   this->matrix. setSize( 0 );
}

template< typename Real,
          typename Index >
void tnlSpmvBenchmarkCSRMatrix< Real, Index > :: writeProgress() const
{
   cout << left << setw( this->formatColumnWidth ) << "CSR";
   //   cout << left << setw( 25 ) << matrixFormat << setw( 5 ) << cudaBlockSize;
   cout << right << setw( this->timeColumnWidth ) << setprecision( 2 ) << this->getTime()
        << right << setw( this->iterationsColumnWidth ) << this->getIterations()
        << right << setw( this->gflopsColumnWidth ) << setprecision( 2 ) << this->getGflops();
   if( this->getBenchmarkWasSuccesful() )
        cout << right << setw( this->benchmarkStatusColumnWidth ) << " OK - SpMV diff. " << getForwardBackwardDifference();
   else
        cout << right << setw( this->benchmarkStatusColumnWidth ) << " FAILED ";
   cout << endl;
}

template< typename Real,
          typename Index >
void tnlSpmvBenchmarkCSRMatrix< Real, Index > :: writeToLogTable( ostream& logFile,
                                                                  const double& csrGflops,
                                                                  const tnlString& inputMtxFile,
                                                                  const tnlCSRMatrix< Real, tnlHost, Index >& csrMatrix,
                                                                  bool writeMatrixInfo  ) const
{
   if( this->getBenchmarkWasSuccesful() )
   {
      logFile << "             <td> " << this->getTime() << "</font></td>" << endl;
      logFile << "             <td> " << this->getGflops() << "</td>" << endl;
   }
   else
   {
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
   }
}

template< typename Real,
          typename Index >
Real tnlSpmvBenchmarkCSRMatrix< Real, Index > :: getForwardBackwardDifference() const
{
   return forwardBackwardDifference;
}

#endif /* TNLSPMVBENCHMARKCSRMATRIX_H_ */
