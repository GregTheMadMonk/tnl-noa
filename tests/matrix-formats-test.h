/***************************************************************************
                          matrix-formats-test.h  -  description
                             -------------------
    begin                : Jul 23, 2010
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

#ifndef MATRIXFORMATSTEST_H_
#define MATRIXFORMATSTEST_H_

#include <fstream>
#include <cstdlib>
#include <matrix/tnlFullMatrix.h>
#include <matrix/tnlCSRMatrix.h>
#include <matrix/tnlRgCSRMatrix.h>
#include <matrix/tnlAdaptiveRgCSRMatrix.h>
#include <matrix/tnlFastCSRMatrix.h>
#include <matrix/tnlFastRgCSRMatrix.h>
#include <matrix/tnlFastRgCSRMatrixCUDA.h>
#include <matrix/tnlEllpackMatrix.h>
#include <core/tnlFile.h>

using namespace std;

template< class T >
bool testMatrixFormats( const tnlString& input_file_name,
		        const tnlString& input_mtx_file_name,
		        int verbose,
		        bool& have_full_matrix,
		        bool& test_full_csr,
		        bool& test_coa_csr,
		        bool& test_cuda_coa_csr,
              bool& test_arg_csr,
              bool& test_cuda_arg_csr,
		        bool& test_fast_csr,
		        bool& test_coa_fast_csr,
		        bool& test_cuda_coa_fast_csr,
		        bool& test_ellpack )
{
	tnlFullMatrix< T >* full_matrix;
	tnlCSRMatrix< T >* csr_matrix;
	tnlRgCSRMatrix< T >* coacsr_matrix;
	tnlRgCSRMatrix< T, tnlCuda >* cuda_coacsr_matrix;
   tnlAdaptiveRgCSRMatrix< T >* adaptiveRgCsrMatrix;
   tnlAdaptiveRgCSRMatrix< T, tnlCuda >* cuda_adaptiveRgCsrMatrix;
	tnlFastCSRMatrix< T >* fast_csr_matrix;
	tnlFastRgCSRMatrix< T >* coa_fast_csr_matrix;
	tnlFastRgCSRMatrix< T, tnlCuda >* cuda_coa_fast_csr_matrix;
	tnlEllpackMatrix< T >* ellpack_matrix;

	const int spmv_test_iterations = 100;

	tnlFile csr_file;
	if( ! csr_file. open( input_file_name, tnlReadMode ) )
	{
		cerr << "Unable to open the file " << input_file_name << "." << endl;
		return false;
	}

	if( verbose )
		cout << "Reading the CSR matrix ... " << flush;
	csr_matrix = new tnlCSRMatrix< T >( "csr-matrix" );
	if( ! csr_matrix -> load( csr_file ) )
	{
		cerr << "Unable to restore the CSR matrix." << endl;
		return false;
	}
	if( verbose )
		cout << " OK." << endl;
	csr_file. close();

	int size = csr_matrix -> getSize();
	tnlLongVector< T, tnlHost > ref_x( "ref-x", size ), ref_b( "ref-b", size);
	tnlLongVector< T, tnlHost > host_x( "host-x", size ), host_b( "host-b", size);
	tnlLongVector< T, tnlCuda > cuda_x( "cuda-x", size ), cuda_b( "cuda-b", size);
	ref_x. setValue( 1.0 );
	ref_b. setValue( 0.0 );
	host_x. setValue( 1.0 );
	host_b. setValue( 0.0 );
#ifdef HAVE_CUDA
	cuda_x. setValue( 1.0 );
	cuda_b. setValue( 0.0 );
#endif


	cout << input_mtx_file_name << endl;
	if( input_mtx_file_name != "" )
	{
		if( verbose )
			cout << "Reading the FULL matrix ... " << endl;
		full_matrix = new tnlFullMatrix< T >( "full-matrix" );
		fstream mtx_file;
		mtx_file. open( input_mtx_file_name. getString(), ios :: in );
		if( ! full_matrix -> read( mtx_file, verbose ) )
			cerr << "Unable to get the FULL matrix." << endl;
		else
			have_full_matrix = true;
		mtx_file. close();
	}
	if( have_full_matrix )
	{
		if( verbose )
			cout << "Comparing the FULL and the CSR matrix ... " << flush;
		if( *full_matrix == *csr_matrix )
		{
			test_full_csr = true;
			if( verbose )
				cout << "OK." << endl;
		}
		else
		{
		   csr_matrix -> printOut( cout, 10 );
			if( verbose )
				cout << "FAILED." << endl;
		}
		delete full_matrix;
	}

	coacsr_matrix = new tnlRgCSRMatrix< T >( "coacsr-matrix" );
	coacsr_matrix -> tuneFormat( 16 );
	coacsr_matrix -> copyFrom( *csr_matrix );

	if( verbose )
		cout << "Comparing the CSR and the Coalesced CSR matrix ... " << flush;
	if( *coacsr_matrix == *csr_matrix )
	{
		test_coa_csr = true;
		if( verbose )
			cout << "OK." << endl;
	}
	else
		if( verbose )
			cout << "FAILED." << endl;

#ifdef HAVE_CUDA
	if( verbose )
		cout << "Comparing the CSR and the Coalesced CSR CUDA matrix by SpMV ... ";
	cuda_coacsr_matrix = new tnlRgCSRMatrix< T, tnlCuda >( "cuda-coacsr-matrix" );
	cuda_coacsr_matrix -> copyFrom( *coacsr_matrix );

	int k( 0 );
	test_cuda_coa_csr = true;
	while( k < size && test_cuda_coa_csr == true )
	{
		if( verbose )
		   cout << "\rComparing the CSR and the Coalesced CSR CUDA matrix by SpMV ... " << k << " / " << size << "        " << flush;
		if( k > 0 )
			host_x[ k - 1 ] = 0.0;
		host_x[ k ] = k;
		cuda_x = host_x;
		csr_matrix -> vectorProduct( host_x,
				             ref_b );

		cuda_coacsr_matrix -> vectorProduct( cuda_x,
				                     cuda_b );
		if( ref_b != cuda_b )
			test_cuda_coa_csr = false;
		k ++;
	}
	if( verbose )
		if( test_cuda_coa_csr == true )
			cout << "\rComparing the CSR and the Coalesced CSR CUDA matrix by SpMV ... " << size << " / " << size << " OK.       " << endl;
		else
		{
			cout << "FAILED at " << k << "th test. " << endl;
			fstream f;
			f. open( "spmv-test", ios ::out );
			host_b = cuda_b;
			for( int j = 0; j < size; j ++ )
				f << ref_b[ j ] << " - " << host_b[ j ] << " = "  << ref_b[ j ] - host_b[ j ] <<  endl;
			f. close();
		}
	delete cuda_coacsr_matrix;
#endif
	delete coacsr_matrix;

   adaptiveRgCsrMatrix = new tnlAdaptiveRgCSRMatrix< T >( "adaptiveRgCsrMatrix" );
   adaptiveRgCsrMatrix -> copyFrom( *csr_matrix );

   if( verbose )
      cout << "Comparing the CSR and the Adaptive Row-grouped CSR matrix ... " << flush;
   if( *adaptiveRgCsrMatrix == *csr_matrix )
   {
      test_arg_csr = true;
      if( verbose )
         cout << "OK." << endl;
   }
   else
      if( verbose )
         cout << "FAILED." << endl;
   //adaptiveRgCsrMatrix -> printOut( cout, 10 );
   //csr_matrix -> printOut( cout, 10 );

   return true;
#ifdef HAVE_CUDA
   if( verbose )
      cout << "Comparing the CSR and the Coalesced CSR CUDA matrix by SpMV ... ";
   cuda_adaptiveRgCsrMatrix = new tnlRgCSRMatrix< T, tnlCuda >( "cuda_adaptiveRgCsrMatrix" );
   cuda_adaptiveRgCsrMatrix -> copyFrom( *adaptiveRgCsrMatrix );

   int k( 0 );
   test_cuda_arg_csr = true;
   while( k < size && test_cuda_coa_csr == true )
   {
      if( verbose )
         cout << "\rComparing the CSR and the Coalesced CSR CUDA matrix by SpMV ... " << k << " / " << size << "        " << flush;
      if( k > 0 )
         host_x[ k - 1 ] = 0.0;
      host_x[ k ] = k;
      cuda_x = host_x;
      csr_matrix -> vectorProduct( host_x,
                         ref_b );

      cuda_coacsr_matrix -> vectorProduct( cuda_x,
                                 cuda_b );
      if( ref_b != cuda_b )
         test_cuda_coa_csr = false;
      k ++;
   }
   if( verbose )
      if( test_cuda_coa_csr == true )
         cout << "\rComparing the CSR and the Coalesced CSR CUDA matrix by SpMV ... " << size << " / " << size << " OK.       " << endl;
      else
      {
         cout << "FAILED at " << k << "th test. " << endl;
         fstream f;
         f. open( "spmv-test", ios ::out );
         host_b = cuda_b;
         for( int j = 0; j < size; j ++ )
            f << ref_b[ j ] << " - " << host_b[ j ] << " = "  << ref_b[ j ] - host_b[ j ] <<  endl;
         f. close();
      }
   delete cuda_coacsr_matrix;
#endif
   delete coacsr_matrix;





	fast_csr_matrix = new tnlFastCSRMatrix< T >( "fast-csr-matrix" );
	fast_csr_matrix -> copyFrom( *csr_matrix );
	if( verbose )
		cout << "Comparing the CSR and the Fast CSR matrix ... " << flush;
	if( *fast_csr_matrix == *csr_matrix )
	{
		test_fast_csr = true;
		if( verbose )
			cout << "OK." << endl;
	}
	else
		if( verbose )
			cout << "FAILED." << endl;

	coa_fast_csr_matrix = new tnlFastRgCSRMatrix< T >( "coa-fast-csr-matrix", 32 );
	//fast_csr_matrix -> printOut( cout );
	bool coa_fast_csr_matrix_transfered = coa_fast_csr_matrix -> copyFrom( *fast_csr_matrix );

	delete fast_csr_matrix;

	if( verbose )
		cout << "Comparing the CSR and the Coalesced Fast CSR matrix ... " << flush;
	if( coa_fast_csr_matrix_transfered )
	{
	   if( *coa_fast_csr_matrix == *csr_matrix )
	   {
	      test_coa_fast_csr = true;
	      if( verbose )
	         cout << "OK." << endl;
	   }
	   else
	      if( verbose )
	         cout << "FAILED." << endl;
	}
	else
	{
	   if( verbose )
	      cout << " N/A." << endl;
	}

#ifdef HAVE_CUDA
	if( verbose )
		cout << "Comparing the CSR and the Coalesced Fast CSR CUDA matrix by SpMV ... ";
	cuda_coa_fast_csr_matrix = new tnlFastRgCSRMatrix< T, tnlCuda >( "cuda-coa-fast-csr-matrix" );
	if( coa_fast_csr_matrix_transfered &&
	    cuda_coa_fast_csr_matrix -> copyFrom( *coa_fast_csr_matrix ) )
	{

		int k( 0 );
		test_cuda_coa_fast_csr = true;
		while( k < size && test_cuda_coa_fast_csr == true )
		{
			if( verbose )
			   cout << "\rComparing the CSR and the Coalesced Fast CSR CUDA matrix by SpMV ... " << k << " / " << size << "        " << flush;
			if( k > 0 )
				host_x[ k - 1 ] = 0.0;
			host_x[ k ] = k;
			cuda_x = host_x;
			csr_matrix -> vectorProduct( host_x, ref_b );

			cuda_coa_fast_csr_matrix -> vectorProduct( cuda_x, cuda_b );
			if( ref_b != cuda_b )
				test_cuda_coa_fast_csr = false;

			k ++;
		}
		if( verbose )
			if( test_cuda_coa_fast_csr == true )
				cout << "\rComparing the CSR and the Coalesced Fast CSR CUDA matrix by SpMV ... " << size << " / " << size << " OK.   " << endl;
			else
			{
				cout << "FAILED at " << k << "th test. " << endl;
				fstream f;
				f. open( "spmv-test", ios ::out );
				host_b = cuda_b;
				for( int j = 0; j < size; j ++ )
					f << setprecision( 20 ) << ref_b[ j ] << " - " << host_b[ j ] << " = "  << ref_b[ j ] - host_b[ j ] <<  endl;
				f << endl;
				coa_fast_csr_matrix -> printOut( f );
				f << endl;
				cuda_coa_fast_csr_matrix -> printOut( f );
				f. close();
			}
	}
	delete cuda_coa_fast_csr_matrix;
#endif
	//coa_fast_csr_matrix -> printOut( cout );
	delete coa_fast_csr_matrix;


	/*int max_row_length, min_row_length, average_row_length;
	csr_matrix -> getRowStatistics( min_row_length,
			                          max_row_length,
			                          average_row_length );
   double alpha= 0.75;
   int ellpack_row_length = ( 1.0 - alpha ) * average_row_length +
                            alpha * max_row_length;

   if( verbose )
      cout << "Comparing the CSR and the ELLPACK matrix ... " << endl
           << "Min row length = " << min_row_length << endl
	        << "Max row length = " << max_row_length << endl
	        << "Average row length = " << average_row_length << endl
	        << "Ellpack row length = " << ellpack_row_length << endl;

	ellpack_matrix = new tnlEllpackMatrix< T >( "ellpack-matrix", ellpack_row_length );
   ellpack_matrix -> copyFrom( *csr_matrix );

	if( verbose )
	   cout << "COO elements = " << ellpack_matrix -> getCOONonzeroElements() << endl;


   if( *ellpack_matrix == *csr_matrix )
   {
      test_ellpack = true;
      if( verbose )
         cout << endl << "Result .... OK." << endl;
   }
   else
      if( verbose )
         cout << endl << "Result .... FAILED." << endl;
   delete ellpack_matrix;*/

	return true;
}


#endif /* MATRIXFORMATSTEST_H_ */
