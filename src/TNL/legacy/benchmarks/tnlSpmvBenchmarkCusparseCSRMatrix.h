/***************************************************************************
                          tnlSpmvBenchmarkCusparseCSR.h  -  description
                             -------------------
    begin                : Feb 16, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLSPMVBENCHMARKCUSPARSECSRMATRIX_H_
#define TNLSPMVBENCHMARKCUSPARSECSRMATRIX_H_

#include "tnlSpmvBenchmark.h"
#include <TNL/tnlConfig.h>
#include <TNL/legacy/matrices/tnlCusparseCSR.h>

template< typename Real, typename Index>
class tnlSpmvBenchmarkCusparseCSR : public tnlSpmvBenchmark< Real, Devices::Cuda, Index, tnlCusparseCSR >
{
   public:
   tnlSpmvBenchmarkCusparseCSR();

   bool setup( const CSR< Real, Devices::Host, Index >& matrix );

   void tearDown();

   Index getArtificialZeros() const;

   void writeProgress() const;

   void writeToLogTable( std::ostream& logFile,
                         const double& csrGflops,
                         const String& inputMtxFile,
                         const CSR< Real, Devices::Host, Index >& csrMatrix,
                         bool writeMatrixInfo  ) const;

   void setNonzeroElements( const Index nonzeroElements );
};

template< typename Real, typename Index>
bool tnlSpmvBenchmarkCusparseCSR< Real, Index > :: setup( const CSR< Real, Devices::Host, Index >& matrix )
{
   if( ! this->matrix. copyFrom( matrix ) )
      return false;
   this->setupOk = true;
   return true;
}

template< typename Real,
          typename Index>
void tnlSpmvBenchmarkCusparseCSR< Real, Index > :: tearDown()
{
   this->matrix. reset();
}

template< typename Real,
          typename Index>
Index tnlSpmvBenchmarkCusparseCSR< Real, Index > :: getArtificialZeros() const
{
   return 0;
}

template< typename Real,
          typename Index >
void tnlSpmvBenchmarkCusparseCSR< Real, Index > :: writeProgress() const
{
  std::cout << left << std::setw( this->formatColumnWidth ) << "Cusparse";
   //  std::cout << left << std::setw( 25 ) << matrixFormat << std::setw( 5 ) << cudaBlockSize;
  std::cout << right << std::setw( this->timeColumnWidth ) << std::setprecision( 2 ) << this->getTime()
        << right << std::setw( this->iterationsColumnWidth ) << this->getIterations()
        << right << std::setw( this->gflopsColumnWidth ) << std::setprecision( 2 ) << this->getGflops();
   if( this->getBenchmarkWasSuccesful() )
       std::cout << right << std::setw( this->benchmarkStatusColumnWidth ) << "OK ";
   else
       std::cout << right << std::setw( this->benchmarkStatusColumnWidth ) << "  FAILED - maxError is " << this->maxError << ". ";
#ifndef HAVE_CUSP
  std::cout << "CUSPARSE library is missing.";
#endif
  std::cout << std::endl;
}

template< typename Real,
          typename Index >
tnlSpmvBenchmarkCusparseCSR< Real, Index > :: tnlSpmvBenchmarkCusparseCSR()
{

}

template< typename Real,
          typename Index >
void tnlSpmvBenchmarkCusparseCSR< Real, Index > :: writeToLogTable( std::ostream& logFile,
                                                                       const double& csrGflops,
                                                                       const String& inputMtxFile,
                                                                       const CSR< Real, Devices::Host, Index >& csrMatrix,
                                                                       bool writeMatrixInfo  ) const
{
   if( this->getBenchmarkWasSuccesful() )
   {
      double speedUp = this->getGflops() / csrGflops;
      String bgColor = this->getBgColorBySpeedUp( speedUp );
      logFile << "             <td bgcolor=" << bgColor << ">" << this->getTime() << "</td>" << std::endl;
      logFile << "             <td bgcolor=" << bgColor << ">" << this->getGflops() << "</td>" << std::endl;

      logFile << "             <td bgcolor=" << bgColor << "> " << speedUp << "</td>" << std::endl;
   }
   else
   {
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << std::endl;
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << std::endl;
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << std::endl;
   }
}

#endif /* TNLSPMVBENCHMARKCUSPARSECSRMATRIX_H_ */