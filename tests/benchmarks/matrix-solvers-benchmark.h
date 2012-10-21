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

#include <fstream>
#include <core/tnlFile.h>
#include <core/tnlObject.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <matrix/tnlCSRMatrix.h>
#include <solvers/tnlSimpleIterativeSolverMonitor.h>
#include <solvers/linear/krylov/tnlGMRESSolver.h>

#include "tnlConfig.h"
const char configFile[] = TNL_CONFIG_DIRECTORY "tnl-matrix-solvers-benchmark.cfg.desc";


template< typename Real, typename Index >
bool benchmarkMatrix( const tnlString& fileName )
{
   typedef tnlCSRMatrix< Real, tnlHost, Index > csrMatrixType;
   csrMatrixType csrMatrix( "matrix-solvers-benchmark:csrMatrix" );
   if( ! csrMatrix. load( fileName ) )
   {
      cerr << "Unable to load file " << fileName << endl;
      return false;
   }

   const Index size = csrMatrix. getSize();
   tnlVector< Real, tnlHost, Index > x1( "matrix-solvers-benchmark:x1" );
   tnlVector< Real, tnlHost, Index > x( "matrix-solvers-benchmark:x" );
   tnlVector< Real, tnlHost, Index > b( "matrix-solvers-benchmark:b" );
   if( ! x1. setSize( size ) ||
       ! x. setSize( size ) ||
       ! b. setSize( size ) )
   {
      cerr << "Sorry, I do not have enough memory for the benchmark." << endl;
      return false;
   }
   x1. setValue( ( Real ) 1.0 );
   x. setValue( ( Real ) 0.0 );

   tnlGMRESSolver< csrMatrixType, tnlHost > gmresSolver;
   //gmresSolver. setName( "matrix-solvers-benchmark:gmresSolver" );
   gmresSolver. setRestarting( 500 );
   //gmresSolver. setVerbosity( 5 );

   cout << "Matrix size is " << size << endl;
   csrMatrix. vectorProduct( x1, b );
   gmresSolver. setRestarting( 500 );
   gmresSolver. setMatrix( csrMatrix );
   gmresSolver. setMaxIterations( 100000 );
   gmresSolver. setMaxResidue( 1.0e-6 );
   tnlSimpleIterativeSolverMonitor< Real, Index > solverMonitor;
   solverMonitor. setSolver( gmresSolver );
   gmresSolver. setSolverMonitor( solverMonitor );
   gmresSolver. setRefreshRate( 10 );
   if( ! gmresSolver. solve( b, x ) )
      return false;
   cout << endl << "L1 diff. norm = " << x. differenceLpNorm( x1, ( Real ) 1.0 )
        << " L2 diff. norm = " << x. differenceLpNorm( x1, ( Real ) 2.0 )
        << " Max. diff. norm = " << x. differenceMax( x1 ) << endl;
#ifdef HAVE_CUDA
   typedef tnlRgCSRMatrix< Real, tnlCuda, Index > rgCSRMatrixType;
   tnlVector< Real, tnlCuda, Index > cudaX( "matrix-solvers-benchmark:cudaX" );
   tnlVector< Real, tnlCuda, Index > cudaB( "matrix-solvers-benchmark:cudaB" );
   cudaX. setLike( x );
   cudaX = x;
   cudaB. setLike( b );
   cudaB = b;
   rgCSRMatrixType rgCSRMatrix( "matrix-solvers-benchmark:rgCSRMatrix" );
   rgCSRMatrix = csrMatrix;
   tnlGMRESSolver< rgCSRMatrixType, tnlCuda > cudaGMRESSolver( "matrix-solvers-benchmark:cudaGMRESSolver" );
   cudaGMRESSolver. setRestarting( 500 );
   cudaGMRESSolver. setMatrix( rgCSRMatrix );
   cudaGMRESSolver. setMaxIterations( 10000 );
   cudaGMRESSolver. setMaxResidue( 1.0e-6 );
   solverMonitor. setSolver( cudaGMRESSolver );
   cudaGMRESSolver. setSolverMonitor( solverMonitor );
   cudaGMRESSolver. setRefreshRate( 10 );

   if( !cudaGMRESSolver. solve( cudaB, cudaX ) )
      return false;
   cout << endl << "L1 diff. norm = " << tnlDifferenceLpNorm( cudaX, x1, ( Real ) 1.0 )
        << " L2 diff. norm = " << tnlDifferenceLpNorm( cudaX, x1, ( Real ) 2.0 )
        << " Max. diff. norm = " << tnlDifferenceMax( cudaX, x1 ) << endl;

#endif
   return true;
}

int main( int argc, char* argv[] )
{
   /****
    * Parsing command line arguments ...
    */
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;

   if( conf_desc. ParseConfigDescription( configFile ) != 0 )
      return 1;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc. PrintUsage( argv[ 0 ] );
      return 1;
   }
   tnlString inputFile = parameters. GetParameter< tnlString >( "input-file" );
   tnlString str_input_mtx_file = parameters. GetParameter< tnlString >( "input-mtx-file" );
   tnlString log_file_name = parameters. GetParameter< tnlString >( "log-file" );
   double stop_time = parameters. GetParameter< double >( "stop-time" );
   int verbose = parameters. GetParameter< int >( "verbose");

   /****
    * Checking a type of the input data
    */
   tnlString objectType;
   if( ! getObjectType( inputFile, objectType ) )
   {
      cerr << "Unable to detect object type in " << inputFile << endl;
      return EXIT_FAILURE;
   }
   tnlList< tnlString > parsedObjectType;
   parseObjectType( objectType,
                    parsedObjectType );
   tnlString objectClass = parsedObjectType[ 0 ];
   if( objectClass != "tnlCSRMatrix" )
   {
      cerr << "I am sorry, I am expecting tnlCSRMatrix in the input file but I found " << objectClass << "." << endl;
      return EXIT_FAILURE;
   }

   tnlString precision = parsedObjectType[ 1 ];
   //tnlString indexing = parsedObjectType[ 3 ];
   if( precision == "float" )
      if( ! benchmarkMatrix< float, int >( inputFile ) )
         return EXIT_FAILURE;
   if( precision == "double" )
      if( ! benchmarkMatrix< double, int >( inputFile ) )
         return EXIT_FAILURE;


   fstream log_file;
   if( log_file_name )
   {
      log_file. open( log_file_name. getString(), ios :: out | ios :: app );
      if( ! log_file )
      {
         cerr << "Unable to open log file " << log_file_name << " for appending logs." << endl;
         return EXIT_FAILURE;
      }
      cout << "Writing to log file " << log_file_name << "..." << endl;
   }
   return EXIT_SUCCESS;

}


#endif /* MATRIXSOLVERSBENCHMARK_H_ */
