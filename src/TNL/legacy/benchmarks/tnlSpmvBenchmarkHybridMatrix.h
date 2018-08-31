/***************************************************************************
                          tnlSpmvBenchmarkHybridMatrix.h  -  description
                             -------------------
    begin                : May 15, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLSPMVBENCHMARKHYBRIDMATRIX_H_
#define TNLSPMVBENCHMARKHYBRIDMATRIX_H_

#include "tnlSpmvBenchmark.h"
#include <TNL/tnlConfig.h>
#ifdef HAVE_CUSP
   #include <cusp/hyb_matrix.h>
   #include <cusp/io/matrix_market.h>
   #include <cusp/multiply.h>
   #include <cusp/print.h>
#endif


template< typename Real, typename Index>
class tnlSpmvBenchmarkHybridMatrix : public tnlSpmvBenchmark< Real, Devices::Host, Index, CSR >
{
   public:

   void setFileName( const String& fileName );

   bool setup( const CSR< Real, Devices::Host, Index >& matrix );

   void tearDown();

   void runBenchmark( const Vector< Real, Devices::Host, Index >& x,
                      const Vector< Real, Devices::Host, Index >& refB,
                      bool verbose );

   void writeProgress() const;

   void writeToLogTable( std::ostream& logFile,
                         const double& csrGflops,
                         const String& inputMtxFile,
                         const CSR< Real, Devices::Host, Index >& csrMatrix,
                         bool writeMatrixInfo  ) const;

   void setNonzeroElements( const Index nonzeroElements );

   protected:

   String fileName;

   Index nonzeroElements;
};

template< typename Real, typename Index>
void tnlSpmvBenchmarkHybridMatrix< Real, Index > :: setFileName( const String& fileName )
{
   this->fileName = fileName;
}

template< typename Real, typename Index>
bool tnlSpmvBenchmarkHybridMatrix< Real, Index > :: setup( const CSR< Real, Devices::Host, Index >& matrix )
{
   return true;
}

template< typename Real,
          typename Index>
void tnlSpmvBenchmarkHybridMatrix< Real, Index > :: tearDown()
{

}

template< typename Real,
          typename Index>
void tnlSpmvBenchmarkHybridMatrix< Real, Index > :: runBenchmark( const Vector< Real, Devices::Host, Index >& _x,
                                                                  const Vector< Real, Devices::Host, Index >& refB,
                                                                  bool verbose )
{
   this->benchmarkWasSuccesful = false;
#ifdef HAVE_CUSP
   try
   {
      // create an empty sparse matrix structure (HYB format)
      cusp::hyb_matrix< Index, Real, cusp::device_memory > A;

      // load a matrix stored in MatrixMarket format
      cusp::io::read_matrix_market_file( A, this->fileName. getString() );

      // allocate storage for solution (x) and right hand side (b)
      cusp::array1d< Real, cusp::host_memory > host_x( A.num_rows, 1 );
      cusp::array1d< Real, cusp::device_memory > x( A.num_rows, 1 );
      cusp::array1d< Real, cusp::device_memory > b( A.num_rows, 0 );

      for( Index j = 0; j < refB. getSize(); j ++ )
         host_x[ j ] = _x[ j ];

      x = host_x;

      TimerRT rt_timer;
      rt_timer. Reset();

      this->iterations = 0;
      //while( rt_timer. getTime() < time )
      {
         for( int i = 0; i < this->maxIterations; i ++ )
         {
            cusp :: multiply( A, x, b );
            cudaThreadSynchronize();
            this->iterations ++;
         }
      }
      this->time = rt_timer. getTime();

      cusp::array1d< Real, cusp::host_memory > host_b( b );
      host_b = b;

      for( Index j = 0; j < refB. getSize(); j ++ )
      {
         //f << refB[ j ] << " - " << host_b[ j ] << " = "  << refB[ j ] - host_b[ j ] <<  std::endl;
         if( refB[ j ] != 0.0 )
            this->maxError = max( this->maxError, ( Real ) fabs( refB[ j ] - host_b[ j ] ) /  ( Real ) fabs( refB[ j ] ) );
         else
            this->maxError = max( this->maxError, ( Real ) fabs( refB[ j ] ) );
      }
      //if( this->maxError < 1.0 )
         this->benchmarkWasSuccesful = true;
      //else
      //   this->benchmarkWasSuccesful = false;


      double flops = 2.0 * this->iterations * this->nonzeroElements;
      this->gflops = flops / this->time * 1.0e-9;

   }
   catch( std::bad_alloc )
   {
      writeProgress();
      return;
   }
#else
   this->benchmarkWasSuccesful = false;
#endif
   writeProgress();
}

template< typename Real,
          typename Index >
void tnlSpmvBenchmarkHybridMatrix< Real, Index > :: writeProgress() const
{
  std::cout << left << std::setw( this->formatColumnWidth ) << "Hybrid";
   //  std::cout << left << std::setw( 25 ) << matrixFormat << std::setw( 5 ) << cudaBlockSize;
  std::cout << right << std::setw( this->timeColumnWidth ) << std::setprecision( 2 ) << this->getTime()
        << right << std::setw( this->iterationsColumnWidth ) << this->getIterations()
        << right << std::setw( this->gflopsColumnWidth ) << std::setprecision( 2 ) << this->getGflops();
   if( this->getBenchmarkWasSuccesful() )
       std::cout << right << std::setw( this->benchmarkStatusColumnWidth ) << "OK ";
   else
       std::cout << right << std::setw( this->benchmarkStatusColumnWidth ) << "  FAILED - maxError is " << this->maxError << ". ";
#ifndef HAVE_CUSP
  std::cout << "CUSP library is missing.";
#endif
  std::cout << std::endl;
}

template< typename Real,
          typename Index >
void tnlSpmvBenchmarkHybridMatrix< Real, Index > :: writeToLogTable( std::ostream& logFile,
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


template< typename Real,
          typename Index >
void tnlSpmvBenchmarkHybridMatrix< Real, Index > :: setNonzeroElements( const Index nonzeroElements )
{
   this->nonzeroElements = nonzeroElements;
}

#endif /* TNLSPMVBENCHMARKHYBRIDMATRIX_H_ */
