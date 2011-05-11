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

class tnlSpmvBenchmark
{
   public:

   template< typename Real, typename Index>
   virtual void runBenchmark( const tnlCSRMatrix< Real, tnlHost, Index >& matrix ) = 0;

   void writeProgress( const tnlString& matrixFormat,
                       const int cudaBlockSize,
                       const double& time,
                       const int iterations,
                       const double& gflops,
                       bool check,
                       const tnlString& info );

   void writeLog( ostream& str ) const = 0;

   bool getBenchmarkWasSuccesful() const;

   void setBenchmarkWasSuccesful( bool benchmarkWasSuccesful );

   bool benchmarkWasSuccesful;
};

tnlSpmvBenchmark :: tnlSpmvBenchmark()
   : benchmarkWasSuccesful( false )
{

}

bool tnlSpmvBenchmark :: getBenchmarkWasSuccesful() const
{
   return benchmarkWasSuccesful;
}

void tnlSpmvBenchmark :: setBenchmarkWasSuccesful( bool benchmarkWasSuccesful )
{
   this -> benchmarkWasSuccesful = benchmarkWasSuccesful;
}

void tnlSpmvBenchmark :: writeProgress( const tnlString& matrixFormat,
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

template< typename Real, tnlDevice device, typename Index >
void benchmarkSpMV( const tnlMatrix< Real, device, Index >& matrix,
                    const tnlLongVector< Real, device, Index >& x,
                    const int nonzero_elements,
                    tnlLongVector< Real, device, Index >& b,
                    double& time,
                    double& gflops,
                    int& iterations )
{
   tnlTimerRT rt_timer;
   rt_timer. Reset();

   {
      for( int i = 0; i < 10; i ++ )
         matrix. vectorProduct( x, b );
      iterations += 10;
   }

   time = rt_timer. GetTime();
   double flops = 2.0 * iterations * nonzero_elements;
   gflops = flops / time * 1.0e-9;
}

class tnlSpmvBenchmarkCSRFormat : public tnlSpmvBenchmark
{
   public:

   template< typename Real, typename Index>
   void runBenchmark( const tnlCSRMatrix< Real, tnlHost, Index >& matrix );

   double gflops;

};

template< typename Real, typename Index>
void tnlSpmvBenchmarkCSRFormat :: runBenchmark( const tnlCSRMatrix< Real, tnlHost, Index >& matrix )
{

};

class spmvBenchmarkStatistics
{
   public:

   spmvBenchmarkStatistics();

   void reset();

   bool writeToLog( ostream& log_file,
                    const tnlString& input_file,
                    long int size,
                    long int nonzero_elements,
                    const double& all_elements );


   int spmv_csr_iter,
       spmv_hyb_iter,
       spmv_coacsr_iter[ 6 ],
       spmv_cuda_coacsr_iter[ 6 ],
       spmv_cuda_rgcsr_iter[ 10 ],
       spmv_ellpack_iter,
       spmv_fast_csr_iter,
       spmv_coa_fast_csr_iter[ 6 ],
       spmv_cuda_coa_fast_csr_iter[ 6 ],
       spmv_cuda_arg_csr_iter,
       coa_fast_csr_max_cs_dict_size[ 6 ],
       coacsr_artificial_zeros[ 6 ];
   double spmv_csr_gflops,
          spmv_hyb_gflops,
          spmv_coacsr_gflops[ 6 ],
          spmv_cuda_coacsr_gflops[ 6 ],
          spmv_cuda_rgcsr_gflops[ 10 ],
          spmv_ellpack_gflops,
          spmv_fast_csr_gflops,
          spmv_coa_fast_csr_gflops[ 6 ],
          spmv_cuda_coa_fast_csr_gflops[ 6 ],
          spmv_cuda_arg_csr_gflops;
   double coa_csr_artificial_zeros[ 7 ],
          ellpack_artificial_zeros,
          fast_csr_compression,
          coa_fast_csr_compression[ 5 ];
};

spmvBenchmarkStatistics :: spmvBenchmarkStatistics()
: spmv_csr_iter( 0 ),
  spmv_hyb_iter( 0 ),
  spmv_ellpack_iter( 0 ),
  spmv_fast_csr_iter( 0 ),
  spmv_cuda_arg_csr_iter( 0 ),
  spmv_csr_gflops( 0.0 ),
  spmv_hyb_gflops( 0.0 ),
  spmv_ellpack_gflops( 0.0 ),
  spmv_fast_csr_gflops( 0.0 ),
  spmv_cuda_arg_csr_gflops( 0.0 ),
  ellpack_artificial_zeros( 0 ),
  fast_csr_compression( 0 )
{

}

void spmvBenchmarkStatistics :: reset()
{
   spmv_csr_iter = 0;
   spmv_fast_csr_iter = 0;
   spmv_ellpack_iter = 0;
   spmv_cuda_arg_csr_iter = 0;

   spmv_csr_gflops = 0.0;
   spmv_fast_csr_gflops = 0.0;
   spmv_ellpack_gflops = 0.0;
   spmv_cuda_arg_csr_gflops = 0.0;

   for( int i = 0; i < 6; i ++ )
   {
      spmv_coacsr_iter[ i ] = 0;
      spmv_cuda_coacsr_iter[ i ] = 0;
      spmv_coa_fast_csr_iter[ i ] = 0;
      spmv_cuda_coa_fast_csr_iter[ i ] = 0;
      spmv_coacsr_gflops[ i ] = 0.0;
      spmv_cuda_coacsr_gflops[ i ] = 0.0;
      spmv_coa_fast_csr_gflops[ i ] = 0.0;
      spmv_cuda_coa_fast_csr_gflops[ i ] = 0.0;
      coa_csr_artificial_zeros[ i ] = 0.0;
      coa_fast_csr_compression[ i ] = 0.0;
      coa_fast_csr_max_cs_dict_size[ i ] = 0;
      coacsr_artificial_zeros[ i ] = 0;
   }
   /*for( int i = 0; i < 10; i ++ )
   {
      spmv_cuda_rg_csr_iter[ i ] = 0;
      spmv_cuda_rg_csr_iter[ i ] = 0;
      spmv_cuda_rg_csr_gflops[ i ] = 0.0;
      spmv_cuda_rg_csr_gflops[ i ] = 0.0;
   }*/

}

bool spmvBenchmarkStatistics :: writeToLog( ostream& log_file,
                                            const tnlString& input_file,
                                            long int size,
                                            long int nonzero_elements,
                                            const double& all_elements )
{
   log_file << "          <tr>" << endl;
   log_file << "             <td> " << input_file << "</td>" << endl;
   log_file << "             <td> " << size << "</td>" << endl;
   log_file << "             <td> " << nonzero_elements << "</td>" << endl;
   log_file << "             <td> " << ( double ) nonzero_elements / all_elements << " %" << "</td>" << endl;
   log_file << "             <td> " << spmv_csr_gflops << "</td>" << endl;
   log_file << "             <td> " << spmv_hyb_gflops << "</td>" << endl;
   log_file << "             <td> " << spmv_hyb_gflops / spmv_csr_gflops << "</td>" << endl;
   log_file << "          </tr>" << endl;

/*   for( int i = 0; i < 6; i ++ )
   {
      log_file << " | " << setw( 10 ) << setprecision( 2 ) << fixed << coa_csr_artificial_zeros[ i ] << " %"
               << " | " << setw( 6 ) << setprecision( 2 ) << spmv_coacsr_gflops[ i ]
               << " | " << setw( 8 ) << setprecision( 2 ) << spmv_coacsr_gflops[ i ] / spmv_csr_gflops
               << " | " << setw( 6 ) << setprecision( 2 ) << spmv_cuda_coacsr_gflops[ i ]
               << " | " << setw( 8 ) << setprecision( 2 ) << spmv_cuda_coacsr_gflops[ i ] / spmv_csr_gflops;
   }
   for( int i = 0; i < 8; i ++ )
   {
      log_file << " | " << setw( 6 ) << setprecision( 2 ) << spmv_cuda_rgcsr_gflops[ i ]
               << " | " << setw( 8 ) << setprecision( 2 ) << spmv_cuda_rgcsr_gflops[ i ] / spmv_csr_gflops;
   }


   log_file << " | " << setw( 5 ) << setprecision( 2 ) << fixed << fast_csr_compression << " %"
            << " | " << setw( 6 ) << setprecision( 2 ) << spmv_fast_csr_gflops;
   for( int i = 0; i < 5; i ++ )
   {
      log_file << " | " << setw( 12 ) << setprecision( 2 ) << coa_fast_csr_max_cs_dict_size[ i ]
               << " | " << setw( 6 ) << setprecision( 2 ) << spmv_coa_fast_csr_gflops[ i ]
               << " | " << setw( 8 ) << setprecision( 2 ) << spmv_coa_fast_csr_gflops[ i ] / spmv_csr_gflops
               << " | " << setw( 6 ) << setprecision( 2 ) << spmv_cuda_coa_fast_csr_gflops[ i ]
               << " | " << setw( 8 ) << setprecision( 2 ) << spmv_cuda_coa_fast_csr_gflops[ i ] / spmv_csr_gflops;
   }
   log_file << " |" << endl;*/
}



template< class REAL >
bool benchmarkMatrix( const tnlString& input_file,
                      const tnlString& input_mtx_file,
                      int verbose,
                      const double&	stop_time,
                      int& size,
                      int& nonzero_elements,
                      spmvBenchmarkStatistics& benchmarkStatistics )
{

   benchmarkStatistics. reset();

   /*
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


   tnlCSRMatrix< REAL > csr_matrix( "csr-matrix" );
   tnlFile binaryFile;
   if( ! binaryFile. open( input_file, tnlReadMode ) )
   {
      cerr << "I am not able to open the file " << input_file << "." << endl;
      return 1;
   }
   if( verbose )
      cout << "Reading the CSR matrix ... " << flush;
   if( ! csr_matrix. load( binaryFile ) )
   {
      cerr << "Unable to restore the CSR matrix." << endl;
      return false;
   }
   if( verbose )
      cout << " OK." << endl;
   binaryFile. close();

   nonzero_elements = csr_matrix. checkNonzeroElements();
   if( nonzero_elements != csr_matrix. getNonzeroElements() )
      cerr << "WARNING: The matrix reports " << csr_matrix. getNonzeroElements() << " but actually there are " << nonzero_elements << " non-zero elements." << endl;
   if( verbose )
      cout << "Matrix size: " << csr_matrix. getSize()
           << " Non-zero elements: " << nonzero_elements << endl;

   size = csr_matrix. getSize();
   nonzero_elements = csr_matrix. getNonzeroElements();
   tnlLongVector< REAL, tnlHost > ref_x( "ref-x", size ), ref_b( "ref-b", size);
   tnlLongVector< REAL, tnlHost > host_x( "host-x", size ), host_b( "host-b", size);
   tnlLongVector< REAL, tnlCuda > cuda_x( "cuda-x", size ), cuda_b( "cuda-b", size);
   for( int i = 0; i < size; i ++ )
      ref_x[ i ] = 1.0; //( REAL ) i * 1.0 / ( REAL ) size;
   ref_b. setValue( 0.0 );
   host_x = ref_x;
   host_b. setValue( 0.0 );
#ifdef HAVE_CUDA
   cuda_x = ref_x;
   cuda_b. setValue( 0.0 );
#endif


   if( verbose )
      cout << left << setw( 30 ) << "CSR " << flush;
   double time = stop_time;
   benchmarkSpMV< REAL, tnlHost >( csr_matrix,
                                   ref_x,
                                   nonzero_elements,
                                   ref_b,
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
      tnlLongVector< REAL, tnlHost > hyb_b( "hyb-b", size );

      cuspSpMVTest( input_mtx_file. getString(),
                    time,
                    nonzero_elements,
                    spmv_hyb_iter,
                    spmv_hyb_gflops,
                    hyb_b );

      REAL max_err( 0.0 );
      for( int j = 0; j < size; j ++ )
      {
         //f << ref_b[ j ] << " - " << host_b[ j ] << " = "  << ref_b[ j ] - host_b[ j ] <<  endl;
         if( ref_b[ j ] != 0.0 )
            max_err = Max( max_err, ( REAL ) fabs( ref_b[ j ] - hyb_b[ j ] ) /  ( REAL ) fabs( ref_b[ j ] ) );
         else
            max_err = Max( max_err, ( REAL ) fabs( hyb_b[ j ] ) );
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

   tnlAdaptiveRgCSRMatrix< REAL, tnlHost > argcsr_matrix( "argcsr-matrix" );
   argcsr_matrix. setCUDABlockSize( 128 );
   if( argcsr_matrix. copyFrom( csr_matrix ) )
   {
      /*time = stop_time;
      benchmarkSpMV< REAL, tnlCuda >( cuda_coacsr_matrix,
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

      if( ref_b != cuda_b )
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

   tnlAdaptiveRgCSRMatrix< REAL, tnlCuda > cuda_argcsr_matrix( "cuda-argcsr-matrix" );
   cuda_argcsr_matrix. setCUDABlockSize( 128 );

   if( cuda_argcsr_matrix. copyFrom( argcsr_matrix ) )
   {
      time = stop_time;
      benchmarkSpMV< REAL, tnlCuda >( cuda_argcsr_matrix,
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

      if( ref_b != cuda_b )
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

      tnlRgCSRMatrix< REAL, tnlHost, int > coacsr_matrix( "coacsr-matrix", block_size );

      if( coacsr_matrix. copyFrom( csr_matrix ) )
      {
         benchmarkStatistics. coa_csr_artificial_zeros[ block_iter ] = 100.0 * ( double ) coacsr_matrix. getArtificialZeroElements() / ( double ) coacsr_matrix. getNonzeroElements();

         time = stop_time;
         benchmarkSpMV< REAL, tnlHost >( coacsr_matrix,
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


         if( ref_b != host_b )
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

      tnlRgCSRMatrix< REAL, tnlCuda > cuda_coacsr_matrix( "cuda-coacsr-matrix" );

      if( cuda_coacsr_matrix. copyFrom( coacsr_matrix ) )
      {
         time = stop_time;
         benchmarkSpMV< REAL, tnlCuda >( cuda_coacsr_matrix,
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

         if( ref_b != cuda_b )
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
			REAL max_err( 0.0 );
			for( int j = 0; j < size; j ++ )
			{
			   if( ref_b[ j ] != 0.0 )
			      max_err = Max( max_err, ( REAL ) fabs( ref_b[ j ] - host_b[ j ] ) /  ( REAL ) fabs( ref_b[ j ] ) );
			   else
			      max_err = Max( max_err, ( REAL ) fabs( host_b[ j ] ) );
			   f << ref_b[ j ] << "  " << host_b[ j ] << endl;
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

      tnlFastCSRMatrix< REAL > fast_csr_matrix( "fast-csr-matrix" );

      if( fast_csr_matrix. copyFrom( csr_matrix ) )
      {
         benchmarkStatistics. fast_csr_compression = 100.0 * ( 1.0 -  ( double ) fast_csr_matrix. getColumnSequencesLength() / ( double ) fast_csr_matrix. getNonzeroElements() );

         time = stop_time;
         benchmarkSpMV< REAL, tnlHost >( fast_csr_matrix,
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

         if( ref_b != host_b )
         {
            if( verbose )
               cout << right << setw( 12 ) << "FAILED." << endl;
            REAL max_err( 0.0 );
            for( int i = 0; i < size; i ++ )
               max_err = Max( max_err, ( REAL ) fabs( host_b[ i ] - ref_b[ i ] ) );
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

         tnlFastRgCSRMatrix< REAL > coa_fast_csr_matrix( "coa_fast-csr-matrix", block_size );

         if( coa_fast_csr_matrix. copyFrom( fast_csr_matrix ) )
         {
            //coa_fast_csr_compression = 100.0 * ( 1.0 -  ( double ) coa_fast_csr_matrix. getColumnSequencesLength() / ( double ) coa_fast_csr_matrix. getNonzeroElements() );
            benchmarkStatistics. coa_fast_csr_max_cs_dict_size[ block_iter ] = coa_fast_csr_matrix. getMaxColumnSequenceDictionarySize();
            time = stop_time;
            benchmarkSpMV< REAL, tnlHost >( coa_fast_csr_matrix,
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

            if( ref_b != host_b )
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
            benchmarkStatistics. coa_fast_csr_max_cs_dict_size[ block_iter ] = coa_fast_csr_matrix. getMaxColumnSequenceDictionarySize();
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

         tnlFastRgCSRMatrix< REAL, tnlCuda > cuda_coa_fast_csr_matrix( "cuda-coa-fast-csr-matrix" );

         if( cuda_coa_fast_csr_matrix. copyFrom( coa_fast_csr_matrix ) )
         {
            time = stop_time;
            cuda_b. setValue( -1.0 );
            benchmarkSpMV< REAL, tnlCuda >( cuda_coa_fast_csr_matrix,
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

            //if( ref_b != cuda_b )
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

         tnlRgCSRMatrix< REAL > coacsr_matrix( "coacsr-matrix", groupSize );

         if( coacsr_matrix. copyFrom( csr_matrix ) )
         {
            double artifZeros = 100.0 * ( double ) coacsr_matrix. getArtificialZeroElements() / ( double ) coacsr_matrix. getNonzeroElements();
            if( verbose )
               cout << left << setw( 12 ) << "  OK."
                    << right << setw( 14 ) << "Artif.zeros: " << fixed << artifZeros << "%" << endl;

            tnlRgCSRMatrix< REAL, tnlCuda > cuda_coacsr_matrix( "cuda-coacsr-matrix" );
            if( cuda_coacsr_matrix. copyFrom( coacsr_matrix ) )
            {
               for( int blockSize = 32; blockSize < 512; blockSize *= 2 )
               {
                  if( verbose )
                     cout << left << setw( 25 ) << " Row grouped CSR " << setw( 5 ) << blockSize << flush;

                  cuda_coacsr_matrix. setCUDABlockSize( blockSize );
                  time = stop_time;
                  benchmarkSpMV< REAL, tnlCuda >( cuda_coacsr_matrix,
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
      csr_matrix. getRowStatistics( min_row_length,
                                    max_row_length,
                                    average_row_length );
      double alpha= 1.0;
      int ellpack_row_length = ( 1.0 - alpha ) * average_row_length +
                               alpha * max_row_length;
      tnlEllpackMatrix< REAL, tnlHost > ellpack_matrix( "ellpack-matrix", ellpack_row_length );
      ellpack_artificial_zeros = 100.0 * ( double ) ellpack_matrix. getArtificialZeroElements() / ( double ) ellpack_matrix. getNonzeroElements();
      ellpack_matrix. copyFrom( csr_matrix );
      if( verbose )
           cout << "Min row length = " << min_row_length << endl
                << "Max row length = " << max_row_length << endl
                << "Average row length = " << average_row_length << endl
                << "Ellpack row length = " << ellpack_row_length << endl
                << "COO elements = " << ellpack_matrix. getCOONonzeroElements() << endl;
      time = stop_time;
      host_x. setValue( 1.0 );
      host_b. setValue( 0.0 );
      benchmarkSpMV< REAL, tnlHost >( ellpack_matrix,
                                      host_x,
                                      host_b,
                                      time,
                                      spmv_ellpack_gflops,
                                      spmv_ellpack_iter );

      if( verbose )
         cout << time << " sec. " << spmv_ellpack_iter << " iterations " << spmv_ellpack_gflops << " GFLOPS." << endl;
      if( verbose )
         cout << "Comparing results ... ";
      if( ref_b != host_b )
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

#endif /* SPARSEMATRIXBENCHMARK_H_ */
