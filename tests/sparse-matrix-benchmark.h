/***************************************************************************
                          sparse-matrix-benchmark.h  -  description
                             -------------------
    begin                : Jul 27, 2010
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

#ifndef SPARSEMATRIXBENCHMARK_H_
#define SPARSEMATRIXBENCHMARK_H_

//#define HAVE_CUDA

#include <fstream>
#include <iomanip>
#include <matrix/tnlCSRMatrix.h>
#include <matrix/tnlAdaptiveRgCSRMatrix.h>
#include <matrix/tnlRgCSRMatrix.h>
#include <matrix/tnlFastCSRMatrix.h>
#include <matrix/tnlFastRgCSRMatrix.h>
#include <matrix/tnlFastRgCSRMatrixCUDA.h>
#include <matrix/tnlEllpackMatrix.h>
#include <matrix/tnlEllpackMatrixCUDA.h>
#include <core/tnlTimerRT.h>
#include <core/mfuncs.h>
#include <config.h>



#ifdef HAVE_CUSP
#include <cusp-test.h>
#endif

using namespace std;

template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
class tnlSpmvBenchmark
{
   public:

   tnlSpmvBenchmark();

   virtual bool setup( const tnlCSRMatrix< Real, tnlHost, Index >& matrix ) = 0;

   void runBenchmark( const tnlLongVector< Real, Device, Index >& x,
                      const tnlLongVector< Real, tnlHost, Index >& refB );

   virtual void tearDown() = 0;


   void writeProgress( const tnlString& matrixFormat,
                       const int cudaBlockSize,
                       const double& time,
                       const int iterations,
                       const double& gflops,
                       bool check,
                       const tnlString& info );

   //void writeLog( ostream& str ) const = 0;

   bool getBenchmarkWasSuccesful() const;

   double getGflops() const;

   double getTime() const;

   int getIterations() const;

   Index getArtificialZeros() const;

   protected:

   bool benchmarkWasSuccesful;

   double gflops;

   double time;

   int iterations;

   Index artificialZeros;

   Matrix< Real, Device, Index > matrix;
};

template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
tnlSpmvBenchmark< Real, Device, Index, Matrix > :: tnlSpmvBenchmark()
   : benchmarkWasSuccesful( false ),
     gflops( 0.0 ),
     time( 0.0 ),
     iterations( 0.0 ),
     artificialZeros( 0 )
{

}

template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
bool tnlSpmvBenchmark< Real, Device, Index, Matrix > :: getBenchmarkWasSuccesful() const
{
   return this -> benchmarkWasSuccesful;
}

template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
double tnlSpmvBenchmark< Real, Device, Index, Matrix > :: getGflops() const
{
   return this -> gflops;
}

template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
double tnlSpmvBenchmark< Real, Device, Index, Matrix > :: getTime() const
{
   return this -> time;
}

template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
int tnlSpmvBenchmark< Real, Device, Index, Matrix > :: getIterations() const
{
   return this -> iterations;
}


template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
Index tnlSpmvBenchmark< Real, Device, Index, Matrix > :: getArtificialZeros() const
{
   return this -> artificialZeros;
}

template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
void tnlSpmvBenchmark< Real, Device, Index, Matrix > :: writeProgress( const tnlString& matrixFormat,
                                                                       const int cudaBlockSize,
                                                                       const double& time,
                                                                       const int iterations,
                                                                       const double& gflops,
                                                                       bool check,
                                                                       const tnlString& info )
{
   if( ! cudaBlockSize )
      cout << left << setw( 30 ) << matrixFormat;
   else
      cout << left << setw( 25 ) << matrixFormat << setw( 5 ) << cudaBlockSize;
   cout << right << setw( 12 ) << setprecision( 2 ) << time
        << right << setw( 15 ) << iterations
        << right << setw( 12 ) << setprecision( 2 ) << gflops;
   if( check )
        cout << left << setw( 12 ) << "   OK  ";
   else
        cout << left << setw( 12 ) << "FAILED ";
   cout << info << endl;
}


template< typename Real,
          tnlDevice Device,
          typename Index,
          template< typename Real, tnlDevice Device, typename Index > class Matrix >
void tnlSpmvBenchmark< Real, Device, Index, Matrix > :: runBenchmark( const tnlLongVector< Real, Device, Index >& x,
                                                                      const tnlLongVector< Real, tnlHost, Index >& refB )
{
   benchmarkWasSuccesful = false;
   tnlLongVector< Real, Device, Index > b( "tnlSpmvBenchmark< Real, Device, Index, Matrix > :: runBenchmark : b" );
   if( ! b. setSize( refB. getSize() ) )
      return;
   tnlTimerRT rt_timer;
   rt_timer. Reset();

   {
      for( int i = 0; i < 10; i ++ )
         matrix. vectorProduct( x, b );
      iterations += 10;
   }

   Real maxErr( 0.0 );
   for( Index j = 0; j < refB. getSize(); j ++ )
   {
      //f << refB[ j ] << " - " << host_b[ j ] << " = "  << refB[ j ] - host_b[ j ] <<  endl;
      if( refB[ j ] != 0.0 )
         maxErr = Max( maxErr, ( Real ) fabs( refB[ j ] - b[ j ] ) /  ( Real ) fabs( refB[ j ] ) );
      else
         maxErr = Max( maxErr, ( Real ) fabs( refB[ j ] ) );
   }

   time = rt_timer. GetTime();
   double flops = 2.0 * iterations * matrix. getNonzeroElements();
   gflops = flops / time * 1.0e-9;
   artificialZeros = matrix. getArtificialZeros();
   benchmarkWasSuccesful = true;
}



template< typename Real, typename Index>
class tnlSpmvBenchmarkCSRFormat : public tnlSpmvBenchmark< Real, tnlHost, Index, tnlCSRMatrix >
{
   public:

   bool setup( const tnlCSRMatrix< Real, tnlHost, Index >& matrix ) = 0;

   void tearDown();

};

template< typename Real, typename Index>
bool tnlSpmvBenchmarkCSRFormat< Real, Index > :: setup( const tnlCSRMatrix< Real, tnlHost, Index >& matrix )
{
   this -> matrix = matrix;
}

template< typename Real, typename Index>
void tnlSpmvBenchmarkCSRFormat< Real, Index > :: tearDown()
{
   this -> matrix. setSize( 0 );
}

template< class Real >
bool benchmarkMatrix( const tnlString& input_file,
                      const tnlString& input_mtx_file,
                      const tnlString& logFileName,
                      int verbose )
{
   /****
    * Write teminal table header
    */
   if( verbose )
      cout << left << setw( 25 ) << "MATRIX FORMAT"
           << left << setw( 5 ) << "BLOCK"
           << right << setw( 12 ) << "TIME"
           << right << setw( 15 ) << "ITERATIONS"
           << right << setw( 12 ) << "GFLOPS"
           << right << setw( 12 ) << "CHECK"
           << left << setw( 20 ) << " INFO" << endl
           << setfill( '-' ) << setw( 105 ) << "--" << endl
           << setfill( ' ');


   /****
    * Read the CSR matrix ...
    */
   tnlCSRMatrix< Real > csrMatrix( "csr-matrix" );
   tnlFile binaryFile;
   if( ! binaryFile. open( input_file, tnlReadMode ) )
   {
      cerr << "I am not able to open the file " << input_file << "." << endl;
      return 1;
   }
   if( verbose )
      cout << "Reading the CSR matrix ... " << flush;
   if( ! csrMatrix. load( binaryFile ) )
   {
      cerr << "Unable to restore the CSR matrix." << endl;
      return false;
   }
   if( verbose )
      cout << " OK." << endl;
   binaryFile. close();

   /****
    * Check the real number of the non-zero elements
    */
   const long int nonzeroElements = csrMatrix. checkNonzeroElements();
   if( nonzeroElements != csrMatrix. getNonzeroElements() )
      cerr << "WARNING: The matrix reports " << csrMatrix. getNonzeroElements() << " but actually there are " << nonzeroElements << " non-zero elements." << endl;
   if( verbose )
      cout << "Matrix size: " << csrMatrix. getSize()
           << " Non-zero elements: " << nonzeroElements << endl;

   const long int size = csrMatrix. getSize();
   tnlLongVector< Real, tnlHost > refX( "ref-x", size ), refB( "ref-b", size);
   for( int i = 0; i < size; i ++ )
      refX[ i ] = 1.0; //( Real ) i * 1.0 / ( Real ) size;
   csrMatrix. vectorProduct( refX, refB );

   /****
    * CSR format benchmark
    */
   tnlSpmvBenchmarkCSRFormat< Real, int > csrFormatBenchmark;
   csrFormatBenchmark. setup( csrMatrix );
   csrFormatBenchmark. runBenchmark( refX, refB );
   csrFormatBenchmark. tearDown();

   /****
    * Open and write one line to the log file
    */
   fstream logFile;
   if( logFileName )
   {
      logFile. open( logFileName. getString(), ios :: out | ios :: app );
      if( ! logFile )
      {
         cerr << "Unable to open log file " << logFileName << " for appending logs." << endl;
         return false;
      }
      cout << "Writing to log file " << logFileName << "..." << endl;
      long int allElements = csrMatrix -> getSize() * csrMatrix -> getSize();
      logFile << "          <tr>" << endl;
      logFile << "             <td> " << input_file << "</td>" << endl;
      logFile << "             <td> " << csrMatrix -> getSize() << "</td>" << endl;
      logFile << "             <td> " << nonzeroElements << "</td>" << endl;
      logFile << "             <td> " << ( double ) nonzeroElements / allElements << " %" << "</td>" << endl;
      logFile << "             <td> " << csrFormatBenchmark. getGflops() << "</td>" << endl;
      //logFile << "             <td> " << spmv_hyb_gflops << "</td>" << endl;
      //logFile << "             <td> " << spmv_hyb_gflops / spmv_csr_gflops << "</td>" << endl;
      logFile << "          </tr>" << endl;
      logFile. close();
   }
}

#ifdef UNDEF

   if( verbose )
      cout << left << setw( 30 ) << "CSR " << flush;
   double time = stop_time;
   benchmarkSpMV< Real, tnlHost >( csrMatrix,
                                   refX,
                                   nonzero_elements,
                                   refB,
                                   time,
                                   benchmarkStatistics. spmv_csr_gflops,
                                   benchmarkStatistics. spmv_csr_iter );
   if( verbose )
      cout << right << setw( 12 ) << setprecision( 2 ) << time
           << right << setw( 15 ) << benchmarkStatistics. spmv_csr_iter
           << right << setw( 12 ) << setprecision( 2 ) << benchmarkStatistics. spmv_csr_gflops
           << left << setw( 12 ) << "  N/A" << endl;

#ifdef HAVE_CUSP
   /*
    * Benchmark of the Hybrid format implemented in the CUSP library
    */
   {
      if( verbose )
         cout << left << setw( 30 ) << "Hybrid (CUSP) " << flush;

      time = stop_time;
      tnlLongVector< Real, tnlHost > hyb_b( "hyb-b", size );

      cuspSpMVTest( input_mtx_file. getString(),
                    time,
                    nonzero_elements,
                    spmv_hyb_iter,
                    spmv_hyb_gflops,
                    hyb_b );

      Real max_err( 0.0 );
      for( int j = 0; j < size; j ++ )
      {
         //f << refB[ j ] << " - " << host_b[ j ] << " = "  << refB[ j ] - host_b[ j ] <<  endl;
         if( refB[ j ] != 0.0 )
            max_err = Max( max_err, ( Real ) fabs( refB[ j ] - hyb_b[ j ] ) /  ( Real ) fabs( refB[ j ] ) );
         else
            max_err = Max( max_err, ( Real ) fabs( hyb_b[ j ] ) );
      }
      //f. close();



      if( verbose )
         cout << right << setw( 12 ) << setprecision( 2 ) << time
              << right << setw( 15 ) << spmv_hyb_iter
              << right << setw( 12 ) << setprecision( 2 ) << spmv_hyb_gflops
              << left << setw( 12 ) << "  Max.err. is " << setprecision( 12 ) << max_err << endl;
   }
#endif
   /***
    * Benchmark of the Adaptive Row-grouped CSR format.
    */
   if( verbose )
      cout << left << setw( 25 ) << "AdaptiveRow-grouped CSR " << setw( 5 ) << flush;

   tnlAdaptiveRgCSRMatrix< Real, tnlHost > argcsrMatrix( "argcsr-matrix" );
   argcsrMatrix. setCUDABlockSize( 128 );
   if( argcsrMatrix. copyFrom( csrMatrix ) )
   {
      /*time = stop_time;
      benchmarkSpMV< Real, tnlCuda >( cuda_coacsrMatrix,
                                      cuda_x,
                                      nonzero_elements,
                                      cuda_b,
                                      time,
                                      spmv_cuda_coa_csr_gflops[ block_iter ],
                                      spmv_cuda_coa_csr_iter[ block_iter ] );

      if( verbose )
         cout << right << setw( 12 ) << setprecision( 2 ) << time
              << right << setw( 15 ) << spmv_cuda_coa_csr_iter[ block_iter ]
              << right << setw( 12 ) << setprecision( 2 ) << spmv_cuda_coa_csr_gflops[ block_iter ];

      if( refB != cuda_b )
      {
         if( verbose )
            cout << right << setw( 12 ) << "FAILED." << endl;
         //spmv_cuda_coa_csr_gflops[ block_iter ] = -1.0;
         //return false;
      }
      else*/
         if( verbose )
            cout << right << setw( 12 ) << "OK." << endl;

   }
   else
   {
      if( verbose )
         cout << "Format transfer failed!!!" << endl;
   }



#ifdef HAVE_CUDA
   /***
    * Benchmark of the Adaptive Row-grouped CSR format on the CUDA device.
    */
   if( verbose )
      cout << left << setw( 25 ) << "AdaptiveRow-grouped CSR " << setw( 5 ) << flush;

   tnlAdaptiveRgCSRMatrix< Real, tnlCuda > cuda_argcsrMatrix( "cuda-argcsr-matrix" );
   cuda_argcsrMatrix. setCUDABlockSize( 128 );

   if( cuda_argcsrMatrix. copyFrom( argcsrMatrix ) )
   {
      time = stop_time;
      benchmarkSpMV< Real, tnlCuda >( cuda_argcsrMatrix,
                                      cuda_x,
                                      nonzero_elements,
                                      cuda_b,
                                      time,
                                      benchmarkStatistics. spmv_cuda_arg_csr_gflops,
                                      benchmarkStatistics. spmv_cuda_arg_csr_iter );

      if( verbose )
         cout << right << setw( 12 ) << setprecision( 2 ) << time
              << right << setw( 15 ) << benchmarkStatistics. spmv_cuda_arg_csr_iter
              << right << setw( 12 ) << setprecision( 2 ) << benchmarkStatistics. spmv_cuda_arg_csr_gflops;

      if( refB != cuda_b )
      {
         if( verbose )
            cout << right << setw( 12 ) << "FAILED." << endl;
         //spmv_cuda_coa_csr_gflops[ block_iter ] = -1.0;
         //return false;
      }
      else
         if( verbose )
            cout << right << setw( 12 ) << "OK." << endl;

   }
   else
   {
      if( verbose )
         cout << "Format transfer failed!!!" << endl;
   }

#endif

   int block_iter = 0;
   for( int block_size = 16; block_size < 512; block_size *= 2 )
   {
      /***
       * Benchmark of the Row-grouped CSR format.
       */
      if( verbose )
         cout << left << setw( 25 ) << "Row-grouped CSR " << setw( 5 ) << block_size << flush;

      tnlRgCSRMatrix< Real, tnlHost, int > coacsrMatrix( "coacsr-matrix", block_size );

      if( coacsrMatrix. copyFrom( csrMatrix ) )
      {
         benchmarkStatistics. coa_csr_artificial_zeros[ block_iter ] = 100.0 * ( double ) coacsrMatrix. getArtificialZeroElements() / ( double ) coacsrMatrix. getNonzeroElements();

         time = stop_time;
         benchmarkSpMV< Real, tnlHost >( coacsrMatrix,
                                         host_x,
                                         nonzero_elements,
                                         host_b,
                                         time,
                                         benchmarkStatistics. spmv_coacsr_gflops[ block_iter ],
                                         benchmarkStatistics. spmv_coacsr_iter[ block_iter ] );
         if( verbose )
            cout << right << setw( 12 ) << setprecision( 2 ) << time
                 << right << setw( 15 ) << benchmarkStatistics. spmv_coacsr_iter[ block_iter ]
                 << right << setw( 12 ) << setprecision( 2 ) << benchmarkStatistics. spmv_coacsr_gflops[ block_iter ] << flush;


         if( refB != host_b )
         {
            if( verbose )
               cout << right << setw( 12 ) << "FAILED." << endl;
            benchmarkStatistics. spmv_coacsr_gflops[ block_iter ] = -1.0;
            //return false;
         }
         if( verbose )
            cout << left << setw( 12 ) << "  OK."
                 << right << setw( 14 ) << "Artif.zeros: " << fixed << benchmarkStatistics. coacsr_artificial_zeros[ block_iter ] << "%" << endl;

      }
      else
         if( verbose )
            cout << "Format transfer failed!!!" << endl;

#ifdef HAVE_CUDA
      /****
       * Benchmark of the Row-grouped CSR format on the CUDA device.
       */
      if( verbose )
         cout << left << setw( 25 ) << "Row-grouped CSR CUDA" << setw( 5 ) << block_size << flush;

      tnlRgCSRMatrix< Real, tnlCuda > cuda_coacsrMatrix( "cuda-coacsr-matrix" );

      if( cuda_coacsrMatrix. copyFrom( coacsrMatrix ) )
      {
         time = stop_time;
         benchmarkSpMV< Real, tnlCuda >( cuda_coacsrMatrix,
                                         cuda_x,
                                         nonzero_elements,
                                         cuda_b,
                                         time,
                                         benchmarkStatistics. spmv_cuda_coacsr_gflops[ block_iter ],
                                         benchmarkStatistics. spmv_cuda_coacsr_iter[ block_iter ] );

         if( verbose )
            cout << right << setw( 12 ) << setprecision( 2 ) << time
                 << right << setw( 15 ) << benchmarkStatistics. spmv_cuda_coacsr_iter[ block_iter ]
                 << right << setw( 12 ) << setprecision( 2 ) << benchmarkStatistics. spmv_cuda_coacsr_gflops[ block_iter ];

         if( refB != cuda_b )
			{
				if( verbose )
					cout << right << setw( 12 ) << "FAILED." << endl;
				//spmv_cuda_coa_csr_gflops[ block_iter ] = -1.0;
				//return false;
		  	}
			else
				if( verbose )
					cout << right << setw( 12 ) << "OK." << endl;

			fstream f;
			f. open( "spmv-test-2", ios ::out );
			host_b = cuda_b;
			Real max_err( 0.0 );
			for( int j = 0; j < size; j ++ )
			{
			   if( refB[ j ] != 0.0 )
			      max_err = Max( max_err, ( Real ) fabs( refB[ j ] - host_b[ j ] ) /  ( Real ) fabs( refB[ j ] ) );
			   else
			      max_err = Max( max_err, ( Real ) fabs( host_b[ j ] ) );
			   f << refB[ j ] << "  " << host_b[ j ] << endl;
			}
			f. close();
			if( verbose )
			   cout << left << setw( 12 ) << "  Max.err. is " << setprecision(12 )  << max_err << endl;
			//cerr << "Press ENTER." << endl;
			//getchar();

      }
      else
      {
         if( verbose )
            cout << "Format transfer failed!!!" << endl;
         benchmarkStatistics. spmv_cuda_coacsr_gflops[ block_iter ] = -1.0;
         benchmarkStatistics. spmv_cuda_coacsr_iter[ block_iter ] = 0;
      }
#endif
      block_iter ++;
   }
   if( verbose )
      cout << setfill( '-' ) << setw( 95 ) << "--" << endl
           << setfill( ' ');
   return true;


   /*
    * Benchmark of the Fast CSR format.
    */
   {
      if( verbose )
         cout << left << setw( 30 ) << "Fast CSR " << flush;

      tnlFastCSRMatrix< Real > fast_csrMatrix( "fast-csr-matrix" );

      if( fast_csrMatrix. copyFrom( csrMatrix ) )
      {
         benchmarkStatistics. fast_csr_compression = 100.0 * ( 1.0 -  ( double ) fast_csrMatrix. getColumnSequencesLength() / ( double ) fast_csrMatrix. getNonzeroElements() );

         time = stop_time;
         benchmarkSpMV< Real, tnlHost >( fast_csrMatrix,
                                         host_x,
                                         nonzero_elements,
                                         host_b,
                                         time,
                                         benchmarkStatistics. spmv_fast_csr_gflops,
                                         benchmarkStatistics. spmv_fast_csr_iter );
         if( verbose )
            cout << right << setw( 12 ) << setprecision( 2 ) << time
                 << right << setw( 15 ) << benchmarkStatistics. spmv_fast_csr_iter
                 << right << setw( 12 ) << setprecision( 2 ) << benchmarkStatistics. spmv_fast_csr_gflops << flush;

         if( refB != host_b )
         {
            if( verbose )
               cout << right << setw( 12 ) << "FAILED." << endl;
            Real max_err( 0.0 );
            for( int i = 0; i < size; i ++ )
               max_err = Max( max_err, ( Real ) fabs( host_b[ i ] - refB[ i ] ) );
            if( verbose )
               cout << left << setw( 12 ) <<  "  Max. err. is " << max_err << endl;
            benchmarkStatistics. spmv_fast_csr_gflops = -1.0;
            return false;
         }
         if( verbose )
            cout << left << setw( 12 ) << "  OK."
                 << right << setw( 14 ) << "Compression: " << benchmarkStatistics. fast_csr_compression << "%" << endl;
      }
      else
         if( verbose )
            cout << "Format transfer failed!!!" << endl;
      /*
       * Benchmark Coalesced Fast CSR format.
       */
      block_iter = 0;
      for( int block_size = 16; block_size < 64; block_size *= 2 )
      {
         if( verbose )
            cout << left << setw( 25 ) << "Colesced Fast CSR " << setw( 5 ) << block_size << flush;

         tnlFastRgCSRMatrix< Real > coa_fast_csrMatrix( "coa_fast-csr-matrix", block_size );

         if( coa_fast_csrMatrix. copyFrom( fast_csrMatrix ) )
         {
            //coa_fast_csr_compression = 100.0 * ( 1.0 -  ( double ) coa_fast_csrMatrix. getColumnSequencesLength() / ( double ) coa_fast_csrMatrix. getNonzeroElements() );
            benchmarkStatistics. coa_fast_csr_max_cs_dict_size[ block_iter ] = coa_fast_csrMatrix. getMaxColumnSequenceDictionarySize();
            time = stop_time;
            benchmarkSpMV< Real, tnlHost >( coa_fast_csrMatrix,
                                            host_x,
                                            nonzero_elements,
                                            host_b,
                                            time,
                                            benchmarkStatistics. spmv_coa_fast_csr_gflops[ block_iter ],
                                            benchmarkStatistics. spmv_coa_fast_csr_iter[ block_iter ] );
            if( verbose )
               cout << right << setw( 12 ) << setprecision( 2 ) << time
                    << right << setw( 15 ) << benchmarkStatistics. spmv_coa_fast_csr_iter[ block_iter ]
                    << right << setw( 12 ) << setprecision( 2 ) << benchmarkStatistics. spmv_coa_fast_csr_gflops[ block_iter ] << flush;

            if( refB != host_b )
            {
               if( verbose )
                  cout << right << setw( 12 ) << "FAILED." << endl;
               benchmarkStatistics. spmv_coa_fast_csr_gflops[ block_iter ] = -1.0;
               return false;
            }
            if( verbose )
               cout << left << setw( 12 ) << "  OK." << endl;
         }
         else
         {
            benchmarkStatistics. coa_fast_csr_max_cs_dict_size[ block_iter ] = coa_fast_csrMatrix. getMaxColumnSequenceDictionarySize();
            if( verbose )
               cout << "Format transfer failed!!!" << endl;
            continue;
         }

#ifdef HAVE_CUDA
         /*
          * Benchmark Coalesced Fast CSR format on the CUDA device.
          */

         if( verbose )
            cout << left << setw( 25 ) << "Coalesced Fast CSR CUDA" << setw( 5 ) << block_size << flush;

         tnlFastRgCSRMatrix< Real, tnlCuda > cuda_coa_fast_csrMatrix( "cuda-coa-fast-csr-matrix" );

         if( cuda_coa_fast_csrMatrix. copyFrom( coa_fast_csrMatrix ) )
         {
            time = stop_time;
            cuda_b. setValue( -1.0 );
            benchmarkSpMV< Real, tnlCuda >( cuda_coa_fast_csrMatrix,
                                            cuda_x,
                                            nonzero_elements,
                                            cuda_b,
                                            time,
                                            benchmarkStatistics. spmv_cuda_coa_fast_csr_gflops[ block_iter ],
                                            benchmarkStatistics. spmv_cuda_coa_fast_csr_iter[ block_iter ] );

            if( verbose )
               cout << right << setw( 12 ) << setprecision( 2 ) << time
                    << right << setw( 15 ) << benchmarkStatistics. spmv_cuda_coa_fast_csr_iter[ block_iter ]
                    << right << setw( 12 ) << setprecision( 2 ) << benchmarkStatistics. spmv_cuda_coa_fast_csr_gflops[ block_iter ] << endl;

            //if( refB != cuda_b )
				// {
				//	if( verbose )
				//	   cout << right << setw( 12 ) << "FAILED." << endl;
				//	//spmv_cuda_coa_fast_csr_gflops[ block_iter ] = -1.0;
				//	return false;
				// }
				// else
				// 	if( verbose )
				//		cout << right << setw( 12 ) << "OK." << endl;
         }
         else
            if( verbose )
               cout << "Format transfer failed!!!" << endl;
#endif
         block_iter ++;
      }
   }

#ifdef HAVE_CUDA
   /*
    * Benchmark of RgCSR format with fixed group size and variable CUDA block size
    * NOTE: RgCSR is the same as RgCSR just allows different groupSize and CUDA blockSize.
    */
   block_iter = 0;
   for( int groupSize = 16; groupSize < 64; groupSize *= 2 )
   {
      if( verbose )
         cout << left << setw( 25 ) << "Row grouped CSR " << setw( 5 ) << groupSize << flush;

         tnlRgCSRMatrix< Real > coacsrMatrix( "coacsr-matrix", groupSize );

         if( coacsrMatrix. copyFrom( csrMatrix ) )
         {
            double artifZeros = 100.0 * ( double ) coacsrMatrix. getArtificialZeroElements() / ( double ) coacsrMatrix. getNonzeroElements();
            if( verbose )
               cout << left << setw( 12 ) << "  OK."
                    << right << setw( 14 ) << "Artif.zeros: " << fixed << artifZeros << "%" << endl;

            tnlRgCSRMatrix< Real, tnlCuda > cuda_coacsrMatrix( "cuda-coacsr-matrix" );
            if( cuda_coacsrMatrix. copyFrom( coacsrMatrix ) )
            {
               for( int blockSize = 32; blockSize < 512; blockSize *= 2 )
               {
                  if( verbose )
                     cout << left << setw( 25 ) << " Row grouped CSR " << setw( 5 ) << blockSize << flush;

                  cuda_coacsrMatrix. setCUDABlockSize( blockSize );
                  time = stop_time;
                  benchmarkSpMV< Real, tnlCuda >( cuda_coacsrMatrix,
                                                  cuda_x,
                                                  nonzero_elements,
                                                  cuda_b,
                                                  time,
                                                  benchmarkStatistics. spmv_cuda_coacsr_gflops[ block_iter ],
                                                  benchmarkStatistics. spmv_cuda_coacsr_iter[ block_iter ] );

                  if( verbose )
                     cout << right << setw( 12 ) << setprecision( 2 ) << time
                          << right << setw( 15 ) << benchmarkStatistics. spmv_cuda_coacsr_iter[ block_iter ]
                          << right << setw( 12 ) << setprecision( 2 ) << benchmarkStatistics. spmv_cuda_coacsr_gflops[ block_iter ] << endl;
                  block_iter ++;
               }
            }
         }
         else
            if( verbose )
               cout << "Format transfer failed!!!" << endl;

   }
#endif

   /*
	 * Benchmarks of the ELLPACK format.
	 */
	/*{
      if( verbose )
         cout << "Benchmarking ELLPACK format ... " << flush;

      int max_row_length, min_row_length, average_row_length;
      csrMatrix. getRowStatistics( min_row_length,
                                    max_row_length,
                                    average_row_length );
      double alpha= 1.0;
      int ellpack_row_length = ( 1.0 - alpha ) * average_row_length +
                               alpha * max_row_length;
      tnlEllpackMatrix< Real, tnlHost > ellpack_matrix( "ellpack-matrix", ellpack_row_length );
      ellpack_artificial_zeros = 100.0 * ( double ) ellpack_matrix. getArtificialZeroElements() / ( double ) ellpack_matrix. getNonzeroElements();
      ellpack_matrix. copyFrom( csrMatrix );
      if( verbose )
           cout << "Min row length = " << min_row_length << endl
                << "Max row length = " << max_row_length << endl
                << "Average row length = " << average_row_length << endl
                << "Ellpack row length = " << ellpack_row_length << endl
                << "COO elements = " << ellpack_matrix. getCOONonzeroElements() << endl;
      time = stop_time;
      host_x. setValue( 1.0 );
      host_b. setValue( 0.0 );
      benchmarkSpMV< Real, tnlHost >( ellpack_matrix,
                                      host_x,
                                      host_b,
                                      time,
                                      spmv_ellpack_gflops,
                                      spmv_ellpack_iter );

      if( verbose )
         cout << time << " sec. " << spmv_ellpack_iter << " iterations " << spmv_ellpack_gflops << " GFLOPS." << endl;
      if( verbose )
         cout << "Comparing results ... ";
      if( refB != host_b )
      {
         if( verbose )
            cout << "FAILED." << endl;
         return false;
      }
      if( verbose )
         cout << "OK." << endl;

	}*/

   if( verbose )
      cout << setfill( '-' ) << setw( 95 ) << "--" << endl
           << setfill( ' ');
   return true;           

}
#endif

#endif /* SPARSEMATRIXBENCHMARK_H_ */
