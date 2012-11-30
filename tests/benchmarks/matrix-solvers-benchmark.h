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
#include <solvers/linear/stationary/tnlSORSolver.h>
#include <solvers/linear/krylov/tnlGMRESSolver.h>

#include "tnlConfig.h"
const char configFile[] = TNL_CONFIG_DIRECTORY "tnl-matrix-solvers-benchmark.cfg.desc";

template< typename Solver, typename Matrix, typename Vector >
bool benchmarkSolver( const tnlParameterContainer&  parameters,
                      Solver& solver,
                      const Matrix& matrix,
                      const Vector& b,
                      Vector& x )
{
   typedef typename Matrix :: RealType RealType;
   typedef typename Matrix :: Device DeviceType;
   typedef typename Matrix :: IndexType IndexType;

   const RealType& maxResidue = parameters. GetParameter< double >( "max-residue" );
   const IndexType& size = matrix. getSize();
   const IndexType nonZeros = matrix. getNonzeroElements();
   //const IndexType maxIterations = size * ( ( double ) size * size / ( double ) nonZeros );
   const IndexType maxIterations = 2 * size;
   cout << "Setting max. number of iterations to " << maxIterations << endl;

   solver. setMatrix( matrix );
   solver. setMaxIterations( maxIterations );
   solver. setMaxResidue( maxResidue );
   tnlSimpleIterativeSolverMonitor< RealType, IndexType > solverMonitor;
   solverMonitor. setSolver( solver );
   solver. setSolverMonitor( solverMonitor );
   solver. setRefreshRate( 10 );
   solverMonitor. resetTimers();
   bool testSuccesfull = solver. solve( b, x );

   tnlString logFileName = parameters. GetParameter< tnlString >( "log-file" );
   fstream logFile;
   if( logFileName != "" )
   {
      logFile. open( logFileName. getString(), ios :: out | ios :: app );
      if( ! logFile )
         cerr << "Unable to open the log file " << logFileName << endl;
      else
      {
         if( testSuccesfull )
         {
            double cpuTime = solverMonitor. getCPUTime();
            double realTime = solverMonitor. getRealTime();
            logFile << "             <td> " << solver. getResidue() << " </td> " << endl
                    << "             <td> " << solver. getIterations() << " </td> " << endl
                    << "             <td> " << cpuTime << " </td> " << endl
                    << "             <td> " << realTime << " </td> " << endl;
         }
         else
         {
            logFile << "Solver diverged." << endl;
         }
         logFile. close();
      }
   }
   return testSuccesfull;
}

template< typename Matrix, typename Vector >
bool benchmarkMatrixOnDevice( const tnlParameterContainer&  parameters,
                              const Matrix& matrix,
                              const Vector& b,
                              Vector& x )
{
   typedef typename Matrix :: RealType RealType;
   typedef typename Matrix :: Device DeviceType;
   typedef typename Matrix :: IndexType IndexType;

   const tnlString& solverName = parameters. GetParameter< tnlString >( "solver-name" );
   IndexType iterations( 0 );
   RealType residue( 0.0 );
   bool converged( false );
   if( solverName == "gmres" )
   {
      tnlGMRESSolver< Matrix, DeviceType > gmresSolver;
      const IndexType& gmresRestarting = parameters. GetParameter< int >( "gmres-restarting" );
      gmresSolver. setRestarting( gmresRestarting );
      if( ! benchmarkSolver( parameters, gmresSolver, matrix, b, x ) )
         return false;
   }
   if( solverName == "sor" )
   {
      tnlSORSolver< Matrix, DeviceType > sorSolver;
      const RealType& sorOmega = parameters. GetParameter< double >( "sor-omega" );
      sorSolver. setOmega( sorOmega );
      if( ! benchmarkSolver( parameters, sorSolver, matrix, b, x ) )
         return false;
   }
}


template< typename Real, typename Index >
bool benchmarkMatrix( const tnlParameterContainer&  parameters )
{
   /****
    * Loading the matrix from the input file
    */
   typedef tnlCSRMatrix< Real, tnlHost, Index > csrMatrixType;
   tnlString inputFile = parameters. GetParameter< tnlString >( "input-file" );
   csrMatrixType csrMatrix( "matrix-solvers-benchmark:csrMatrix" );
   if( ! csrMatrix. load( inputFile ) )
   {
      cerr << "Unable to load file " << inputFile << endl;
      return false;
   }

   /****
    * Setting up the linear problem
    */
   const Index size = csrMatrix. getSize();
   cout << "Matrix size is " << size << endl;
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
   csrMatrix. vectorProduct( x1, b );

   const tnlString device = parameters. GetParameter< tnlString >( "device" );
   if( device == "host" )
      if( ! benchmarkMatrixOnDevice( parameters, csrMatrix, b, x ) )
         return false;

   if( device == "cuda" )
   {
#ifdef HAVE_CUDA
      typedef tnlRgCSRMatrix< Real, tnlCuda, Index > rgCSRMatrixType;
      rgCSRMatrixType rgCSRMatrix( "matrix-solvers-benchmark:rgCSRMatrix" );
      rgCSRMatrix = csrMatrix;
      tnlVector< Real, tnlCuda, Index > cudaX( "matrix-solvers-benchmark:cudaX" );
      tnlVector< Real, tnlCuda, Index > cudaB( "matrix-solvers-benchmark:cudaB" );
      cudaX. setLike( x );
      cudaX = x;
      cudaB. setLike( b );
      cudaB = b;
      if( ! benchmarkMatrixOnDevice( parameters, rgCSRMatrix, cudaB, cudaX ) )
         return false;
      x = cudaX;
#else
      cerr << "CUDA support is missing on this system." << endl;
      return false;
#endif
   }

   cout << endl << "L1 diff. norm = " << x. differenceLpNorm( x1, ( Real ) 1.0 )
        << " L2 diff. norm = " << x. differenceLpNorm( x1, ( Real ) 2.0 )
        << " Max. diff. norm = " << x. differenceMax( x1 ) << endl;
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
      if( ! benchmarkMatrix< float, int >( parameters ) )
         return EXIT_FAILURE;
   if( precision == "double" )
      if( ! benchmarkMatrix< double, int >( parameters ) )
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
