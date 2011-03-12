/***************************************************************************
                          matrix-solvers-benchmark.h  -  description
                             -------------------
    begin                : Jan 8, 2011
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

#ifndef MATRIXSOLVERSBENCHMARK_H_
#define MATRIXSOLVERSBENCHMARK_H_

#include <core/tnlFile.h>
#include <matrix/tnlCSRMatrix.h>
#include <solver/tnlGMRESSolver.h>

template< typename Real, typename Index >
bool benchmarkMatrix( const tnlString& fileName )
{
   tnlCSRMatrix< Real, tnlHost, Index > csrMatrix( "matrix-solvers-benchmark:csrMatrix" );
   if( ! csrMatrix. load( fileName ) )
      return false;

   const Index size = csrMatrix. getSize();
   tnlLongVector< Real, tnlHost, Index > x1( "matrix-solvers-benchmark:x1" );
   tnlLongVector< Real, tnlHost, Index > x( "matrix-solvers-benchmark:x" );
   tnlLongVector< Real, tnlHost, Index > b( "matrix-solvers-benchmark:b" );
   if( ! x1. setSize( size ) ||
       ! x. setSize( size ) ||
       ! b. setSize( size ) )
   {
      cerr << "Sorry, I do not have enough memory for the benchmark." << endl;
      return false;
   }
   x1. setValue( ( Real ) 1.0 );
   x. setValue( ( Real ) 0.0 );

   tnlGMRESSolver< Real, tnlHost, Index > gmresSolver( "matrix-solvers-benchmark:gmresSolver" );
   gmresSolver. setRestarting( 500 );
   gmresSolver. setVerbosity( 5 );

   cout << "Matrix size is " << size << endl;
   csrMatrix. vectorProduct( x1, b );
   if( !gmresSolver. solve( csrMatrix,
                            b,
                            x,
                            1.0e-6,
                            100000 ) )
      return false;
   cout << endl << "L1 diff. norm = " << tnlDifferenceLpNorm( x, x1, ( Real ) 1.0 )
        << " L2 diff. norm = " << tnlDifferenceLpNorm( x, x1, ( Real ) 2.0 )
        << " Max. diff. norm = " << tnlDifferenceMax( x, x1 ) << endl;
#ifdef HAVE_CUDA
   tnlLongVector< Real, tnlCuda, Index > cudaX( "matrix-solvers-benchmark:cudaX" );
   tnlLongVector< Real, tnlCuda, Index > cudaB( "matrix-solvers-benchmark:cudaB" );
   cudaX. setLike( x );
   cudaX = x;
   cudaB. setLike( b );
   cudaB = b;
   tnlRgCSRMatrix< Real, tnlCuda, Index > rgCSRMatrix( "matrix-solvers-benchmark:rgCSRMatrix" );
   rgCSRMatrix = csrMatrix;
   tnlGMRESSolver< Real, tnlCuda, Index > cudaGMRESSolver( "matrix-solvers-benchmark:cudaGMRESSolver" );
   cudaGMRESSolver. setRestarting( 500 );
   cudaGMRESSolver. setVerbosity( 5 );

   if( !cudaGMRESSolver. solve( rgCSRMatrix,
                                cudaB,
                                cudaX,
                                1.0e-6,
                                100000 ) )
      return false;
   cout << endl << "L1 diff. norm = " << tnlDifferenceLpNorm( cudaX, x1, ( Real ) 1.0 )
        << " L2 diff. norm = " << tnlDifferenceLpNorm( cudaX, x1, ( Real ) 2.0 )
        << " Max. diff. norm = " << tnlDifferenceMax( cudaX, x1 ) << endl;

#endif
   return true;
}

#endif /* MATRIXSOLVERSBENCHMARK_H_ */
