/***************************************************************************
                          cusp-test.h  -  description
                             -------------------
    begin                : Oct 3, 2010
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

#ifndef CUSPTEST_H_
#define CUSPTEST_H_


#include <config.h>
#include <core/tnlTimerRT.h>
#ifdef HAVE_CUSP
   #include <hyb_matrix.h>
   #include <io/matrix_market.h>
   #include <multiply.h>
   #include <print.h>
#endif
#include <core/tnlLongVector.h>

template< class REAL >
bool cuspSpMVTest( const char* mtx_file_name,
                   double& time,
                   int nonzero_elements,
                   int& spmv_hyb_iter,
                   double& spmv_hyb_gflops,
                   tnlLongVector< REAL >& hyb_result )
{
#ifdef HAVE_CUSP
   // create an empty sparse matrix structure (HYB format)
   cusp::hyb_matrix< int, REAL, cusp::device_memory > A;

   // load a matrix stored in MatrixMarket format
   cusp::io::read_matrix_market_file( A, mtx_file_name );

   // allocate storage for solution (x) and right hand side (b)
   cusp::array1d< REAL, cusp::device_memory > x( A.num_rows, 1 );
   cusp::array1d< REAL, cusp::device_memory > b( A.num_rows, 0 );

   tnlTimerRT rt_timer;
   rt_timer. Reset();

   int iterations( 0 );
   //while( rt_timer. GetTime() < time )
   {
      for( int i = 0; i < 10; i ++ )
         cusp::multiply(A, x, b);
      iterations += 10;
   }

   cusp::array1d< REAL, cusp::host_memory > host_b( b );
   host_b = b;
   hyb_result. setSize( b. size() );
   for( int i = 0; i < b. size(); i ++ )
      hyb_result. setElement( i,  host_b[ i ] );
   //cout << endl << hyb_result << endl;


   time = rt_timer. GetTime();
   double flops = 2.0 * iterations * nonzero_elements;
   spmv_hyb_gflops = flops / time * 1.0e-9;
   spmv_hyb_iter = iterations;

#endif
   return true;
}

#endif /* CUSPTEST_H_ */
