/***************************************************************************
                          tnlSpmvBenchmarkCSR.h  -  description
                             -------------------
    begin                : May 15, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLSPMVBENCHMARKCSRMATRIX_H_
#define TNLSPMVBENCHMARKCSRMATRIX_H_

#include "tnlSpmvBenchmark.h"
#include <TNL/Matrices/CSR.h>

template< typename Real, typename Index>
class tnlSpmvBenchmarkCSR : public tnlSpmvBenchmark< Real, Devices::Host, Index, CSR >
{
   public:

   bool setup( const CSR< Real, Devices::Host, Index >& matrix );

   void tearDown();

   void writeProgress() const;

   void writeToLogTable( std::ostream& logFile,
                         const double& csrGflops,
                         const String& inputMtxFile,
                         const CSR< Real, Devices::Host, Index >& csrMatrix,
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
bool tnlSpmvBenchmarkCSR< Real, Index > :: setup( const CSR< Real, Devices::Host, Index >& matrix )
{
   this->matrix = matrix;

   const Index size = matrix. getSize();
   Vector< Real, Devices::Host > refX( "ref-x", size ), refB( "ref-b", size), backwardRefB( "backwardRef-b", size);
   refX. setValue( 1.0 );
   this->matrix. vectorProduct( refX, refB );
   this->matrix. setBackwardSpMV( true );
   this->matrix. vectorProduct( refX, backwardRefB );
   this->matrix. setBackwardSpMV( false );
   Real error( 0.0 ), maxError( 0.0 );
   for( Index j = 0; j < refB. getSize(); j ++ )
   {
      if( refB[ j ] != 0.0 && backwardRefB[ j ] != 0.0 )
         error = ( Real ) fabs( refB[ j ] - backwardRefB[ j ] ) / min( ( Real ) fabs( refB[ j ] ), ( Real ) fabs( backwardRefB[ j ] ) );
      else
         error = max( ( Real ) fabs( refB[ j ] ), ( Real ) fabs( backwardRefB[ j ] ) );
      maxError = max( error, maxError );
   }
   forwardBackwardDifference = maxError;
   this->setupOk = true;
   return true;
}

template< typename Real, typename Index>
void tnlSpmvBenchmarkCSR< Real, Index > :: tearDown()
{
   this->matrix. setSize( 0 );
}

template< typename Real,
          typename Index >
void tnlSpmvBenchmarkCSR< Real, Index > :: writeProgress() const
{
  std::cout << left << std::setw( this->formatColumnWidth ) << "CSR";
   //  std::cout << left << std::setw( 25 ) << matrixFormat << std::setw( 5 ) << cudaBlockSize;
  std::cout << right << std::setw( this->timeColumnWidth ) << std::setprecision( 2 ) << this->getTime()
        << right << std::setw( this->iterationsColumnWidth ) << this->getIterations()
        << right << std::setw( this->gflopsColumnWidth ) << std::setprecision( 2 ) << this->getGflops();
   if( this->getBenchmarkWasSuccesful() )
       std::cout << right << std::setw( this->benchmarkStatusColumnWidth ) << " OK - SpMV diff. " << getForwardBackwardDifference();
   else
       std::cout << right << std::setw( this->benchmarkStatusColumnWidth ) << " FAILED ";
  std::cout << std::endl;
}

template< typename Real,
          typename Index >
void tnlSpmvBenchmarkCSR< Real, Index > :: writeToLogTable( std::ostream& logFile,
                                                                  const double& csrGflops,
                                                                  const String& inputMtxFile,
                                                                  const CSR< Real, Devices::Host, Index >& csrMatrix,
                                                                  bool writeMatrixInfo  ) const
{
   if( this->getBenchmarkWasSuccesful() )
   {
      logFile << "             <td> " << this->getTime() << "</font></td>" << std::endl;
      logFile << "             <td> " << this->getGflops() << "</td>" << std::endl;
   }
   else
   {
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << std::endl;
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << std::endl;
   }
}

template< typename Real,
          typename Index >
Real tnlSpmvBenchmarkCSR< Real, Index > :: getForwardBackwardDifference() const
{
   return forwardBackwardDifference;
}

#endif /* TNLSPMVBENCHMARKCSRMATRIX_H_ */
