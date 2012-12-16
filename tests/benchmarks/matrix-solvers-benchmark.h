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
#include <solvers/linear/krylov/tnlCGSolver.h>
#include <solvers/linear/krylov/tnlBICGStabSolver.h>
#include <solvers/linear/krylov/tnlGMRESSolver.h>
#include <solvers/linear/krylov/tnlTFQMRSolver.h>
#ifdef HAVE_PETSC
   #include <petsc.h>
#endif

#include "tnlConfig.h"
const char configFile[] = TNL_CONFIG_DIRECTORY "tnl-matrix-solvers-benchmark.cfg.desc";

void writeTestFailToLog( const tnlParameterContainer& parameters )
{
   const tnlString& logFileName = parameters. GetParameter< tnlString >( "log-file" );
   fstream logFile;
   if( logFileName != "" )
   {
      logFile. open( logFileName. getString(), ios :: out | ios :: app );
      if( ! logFile )
         cerr << "Unable to open the log file " << logFileName << endl;
      else
      {
         tnlString bgColor( "#FF0000" );
         logFile << "             <td bgcolor=" << bgColor << "> N/A </td> " << endl
                 << "             <td bgcolor=" << bgColor << "> N/A </td> " << endl
                 << "             <td bgcolor=" << bgColor << "> N/A </td> " << endl;
         logFile. close();
      }
   }
}

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
   const IndexType maxIterations = size;
   cout << "Setting max. number of iterations to " << maxIterations << endl;

   solver. setMatrix( matrix );
   solver. setMaxIterations( maxIterations );
   solver. setMaxResidue( maxResidue );
   solver. setMinResidue( 1.0e9 );
   tnlSimpleIterativeSolverMonitor< RealType, IndexType > solverMonitor;
   solverMonitor. setSolver( solver );
   solver. setSolverMonitor( solverMonitor );
   solver. setRefreshRate( 10 );
   solverMonitor. resetTimers();
   solver. solve( b, x );

   bool solverConverged( solver. getResidue() < maxResidue );
   const tnlString& logFileName = parameters. GetParameter< tnlString >( "log-file" );
   fstream logFile;
   if( logFileName != "" )
   {
      logFile. open( logFileName. getString(), ios :: out | ios :: app );
      if( ! logFile )
         cerr << "Unable to open the log file " << logFileName << endl;
      else
      {
         tnlString bgColor( "#FF0000" );
         if( solver. getResidue() < 1 )
            bgColor="#FF8888";
         if( solver. getResidue() < maxResidue )
         {
            bgColor="#88FF88";
         }
         double cpuTime = solverMonitor. getCPUTime();
         double realTime = solverMonitor. getRealTime();
         logFile << "             <td bgcolor=" << bgColor << "> " << solver. getResidue() << " </td> " << endl
                 << "             <td bgcolor=" << bgColor << "> " << solver. getIterations() << " </td> " << endl
                 << "             <td bgcolor=" << bgColor << "> " << cpuTime << " </td> " << endl;
         logFile. close();
      }
   }
   return solverConverged;

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

   const tnlString& solverClass = parameters. GetParameter< tnlString >( "solver-class" );
   if( solverClass == "tnl" )
   {
      const tnlString& solverName = parameters. GetParameter< tnlString >( "solver-name" );
      IndexType iterations( 0 );
      RealType residue( 0.0 );
      bool converged( false );
      if( solverName == "sor" )
      {
         tnlSORSolver< Matrix, DeviceType > solver;
         const RealType& sorOmega = parameters. GetParameter< double >( "sor-omega" );
         solver. setOmega( sorOmega );
         return benchmarkSolver( parameters, solver, matrix, b, x );
      }
      if( solverName == "cg" )
      {
         tnlCGSolver< Matrix, DeviceType > solver;
         return benchmarkSolver( parameters, solver, matrix, b, x );
      }
      if( solverName == "bicgstab" )
      {
         tnlBICGStabSolver< Matrix, DeviceType > solver;
         return benchmarkSolver( parameters, solver, matrix, b, x );
      }
      if( solverName == "gmres" )
      {
         tnlGMRESSolver< Matrix, DeviceType > solver;
         const IndexType& gmresRestarting = parameters. GetParameter< int >( "gmres-restarting" );
         solver. setRestarting( gmresRestarting );
         return benchmarkSolver( parameters, solver, matrix, b, x );
      }
      if( solverName == "tfqmr" )
      {
         tnlTFQMRSolver< Matrix, DeviceType > solver;
         return benchmarkSolver( parameters, solver, matrix, b, x );
      }
      cerr << "Unknown solver " << solverName << endl;
      return false;
   }
   if( solverClass == "petsc" )
   {
#ifndef HAVE_PETSC
      cerr << "PETSC is not installed on this system." << endl;
      writeTestFailToLog( parameters );
      return false;
#else
      if( DeviceType :: getDeviceType() != "tnlHost" )
      {
         cerr << "PETSC tests can run only on host. The current device is " << DeviceType :: getDeviceType() << endl;
         writeTestFailToLog( parameters );
         return false;
      }
      /****
       * Set-up the PETSC matrix
       */
      const IndexType n = matrix. getSize();
      Mat A;
      MatCreate( PETSC_COMM_WORLD, &A );
      MatSetType( A, MATAIJ );
      MatSetSizes( A, PETSC_DECIDE, PETSC_DECIDE, n, n );
      MatSetUp( A );

      /****
       * Inserting data
       */
      tnlArray< PetscScalar > petscVals;
      tnlArray< PetscInt > petscCols;
      petscVals. setSize( n );
      petscCols. setSize( n );
      for( IndexType i = 0; i < n; i ++ )
      {
         const IndexType rowLength = matrix. getRowLength( i );
         for( IndexType j = 0; j < rowLength; j ++ )
         {
            petscVals. setElement( j, matrix. getRowValues( i )[ j ] );
            petscCols. setElement( j, matrix. getRowColumnIndexes( i )[ j ] );
         }
         MatSetValues( A,
                       1,  // setting one row
                       &i, // index of thew row
                       rowLength,
                       petscCols. getData(),
                       petscVals. getData(),
                       INSERT_VALUES );
      }
      MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY );
      MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY );

      /****
       * Check matrix conversion
       */
      /*for( IndexType i = 0; i < n; i++ )
         for( IndexType j = 0; j < n; j ++ )
         {
            PetscScalar value;
            MatGetValues( A, 1, &i, 1, &j, &value );
            if( matrix. getElement( i, j ) != value )
            {
               cerr << "Conversion to PETSC matrix was not correct at position " << i << " " << j << "." << endl;
               cerr << "Values are " << value << " and " << matrix. getElement( i, j ) << endl;
               return false;
            }
         }
      cerr << "PETSC CONVERSION WAS OK!!!" << endl;
      return true;*/

      Vec petscB, petscX;
      KSP ksp;
      KSPCreate( PETSC_COMM_WORLD, &ksp );


#endif
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
    * Writing matrix statistics
    */
   tnlString matrixStatsFileName = parameters. GetParameter< tnlString >( "matrix-stats-file" );
   if( matrixStatsFileName )
   {
      fstream matrixStatsFile;
      matrixStatsFile. open( matrixStatsFileName. getString(), ios :: out );
      if( ! matrixStatsFile )
      {
         cerr << "Unable to open matrix statistics file " << matrixStatsFileName << endl;
         return false;
      }
      matrixStatsFile << "             <td> " << csrMatrix. getSize() << " </td> " << endl
                      << "             <td> " << csrMatrix. getNonzeroElements() << " </td> " << endl;
      matrixStatsFile. close();
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
#ifdef HAVE_PETSC
   PetscInitialize( &argc, &argv, ( char* ) 0, ( char* ) 0 );
#endif
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
      {
#ifdef HAVE_PETSC
         PetscFinalize();
#endif
         return EXIT_FAILURE;
      }

   if( precision == "double" )
      if( ! benchmarkMatrix< double, int >( parameters ) )
      {
#ifdef HAVE_PETSC
         PetscFinalize();
#endif
         return EXIT_FAILURE;
      }

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
#ifdef HAVE_PETSC
   PetscFinalize();
#endif
   return EXIT_SUCCESS;

}


#endif /* MATRIXSOLVERSBENCHMARK_H_ */
