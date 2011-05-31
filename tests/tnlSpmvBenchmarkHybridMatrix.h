/***************************************************************************
                          tnlSpmvBenchmarkHybridMatrix.h  -  description
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

#ifndef TNLSPMVBENCHMARKHYBRIDMATRIX_H_
#define TNLSPMVBENCHMARKHYBRIDMATRIX_H_

#include <tnlSpmvBenchmark.h>
#ifdef HAVE_CUSP && HAVE_CUDA
   #include <hyb_matrix.h>
   #include <io/matrix_market.h>
   #include <multiply.h>
   #include <print.h>
#endif


template< typename Real, typename Index>
class tnlSpmvBenchmarkHybridMatrix : public tnlSpmvBenchmark< Real, tnlHost, Index, tnlCSRMatrix >
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
void tnlSpmvBenchmarkHybridMatrix< Real, Index > :: setFileName( const tnlString& fileName )
{
   this -> fileName = fileName;
}

template< typename Real, typename Index>
bool tnlSpmvBenchmarkHybridMatrix< Real, Index > :: setup( const tnlCSRMatrix< Real, tnlHost, Index >& matrix )
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
void tnlSpmvBenchmarkHybridMatrix< Real, Index > :: runBenchmark( const tnlLongVector< Real, tnlHost, Index >& _x,
                                                                  const tnlLongVector< Real, tnlHost, Index >& refB,
                                                                  bool verbose )
{
#ifdef HAVE_CUSP
   // create an empty sparse matrix structure (HYB format)
   cusp::hyb_matrix< Index, Real, cusp::device_memory > A;

   // load a matrix stored in MatrixMarket format
   cusp::io::read_matrix_market_file( A, this -> fileName. getString() );

   // allocate storage for solution (x) and right hand side (b)
   cusp::array1d< Real, cusp::device_memory > x( A.num_rows, 1 );
   cusp::array1d< Real, cusp::device_memory > b( A.num_rows, 0 );

   tnlTimerRT rt_timer;
   rt_timer. Reset();

   int iterations( 0 );
   //while( rt_timer. GetTime() < time )
   {
      for( int i = 0; i < this -> maxIterations; i ++ )
      {
         cusp :: multiply( A, x, b );
         this -> iterations ++;
      }
   }

   cusp::array1d< Real, cusp::host_memory > host_b( b );
   host_b = b;

   for( Index j = 0; j < refB. getSize(); j ++ )
   {
      //f << refB[ j ] << " - " << host_b[ j ] << " = "  << refB[ j ] - host_b[ j ] <<  endl;
      if( refB[ j ] != 0.0 )
         this -> maxError = Max( this -> maxError, ( Real ) fabs( refB[ j ] - host_b[ j ] ) /  ( Real ) fabs( refB[ j ] ) );
      else
         this -> maxError = Max( this -> maxError, ( Real ) fabs( refB[ j ] ) );
   }
   if( this -> maxError < tnlSpmvBenchmarkPrecision( this -> maxError ) )
      this -> benchmarkWasSuccesful = true;
   else
      this -> benchmarkWasSuccesful = false;

   this -> time = rt_timer. GetTime();
   double flops = 2.0 * iterations * this -> nonzeroElements;
   this -> gflops = flops / this -> time * 1.0e-9;
#else
   this -> benchmarkWasSuccesful = false;
#endif
   writeProgress();
}

template< typename Real,
          typename Index >
void tnlSpmvBenchmarkHybridMatrix< Real, Index > :: writeProgress() const
{
   cout << left << setw( this -> formatColumnWidth ) << "Hybrid";
   //   cout << left << setw( 25 ) << matrixFormat << setw( 5 ) << cudaBlockSize;
   cout << right << setw( this -> timeColumnWidth ) << setprecision( 2 ) << this -> getTime()
        << right << setw( this -> iterationsColumnWidth ) << this -> getIterations()
        << right << setw( this -> gflopsColumnWidth ) << setprecision( 2 ) << this -> getGflops();
   if( this -> getBenchmarkWasSuccesful() )
        cout << right << setw( this -> benchmarkStatusColumnWidth ) << "OK ";
   else
        cout << right << setw( this -> benchmarkStatusColumnWidth ) << "  FAILED - maxError is " << this -> maxError << ". ";
#ifndef HAVE_CUSP
   cout << "CUSP library is missing.";
#endif
   cout << endl;
}

template< typename Real,
          typename Index >
void tnlSpmvBenchmarkHybridMatrix< Real, Index > :: writeToLogTable( ostream& logFile,
                                                                     const double& csrGflops,
                                                                     const tnlString& inputMtxFile,
                                                                     const tnlCSRMatrix< Real, tnlHost, Index >& csrMatrix,
                                                                     bool writeMatrixInfo  ) const
{
   if( this -> getBenchmarkWasSuccesful() )
   {
      logFile << "             <td> " << this -> getGflops() << "</td>" << endl;
      double speedUp = this -> getGflops() / csrGflops;
      logFile << "             <td bgcolor=" << this -> getBgColorBySpeedUp( speedUp ) << "> " << speedUp << "</td>" << endl;
   }
   else
   {
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << endl;

   }
}


template< typename Real,
          typename Index >
void tnlSpmvBenchmarkHybridMatrix< Real, Index > :: setNonzeroElements( const Index nonzeroElements )
{
   this -> nonzeroElements = nonzeroElements;
}

#endif /* TNLSPMVBENCHMARKHYBRIDMATRIX_H_ */
