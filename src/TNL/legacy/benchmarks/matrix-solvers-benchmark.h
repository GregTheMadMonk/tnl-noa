/***************************************************************************
                          matrix-solvers-benchmark.h  -  description
                             -------------------
    begin                : Jan 8, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef MATRIXSOLVERSBENCHMARK_H_
#define MATRIXSOLVERSBENCHMARK_H_

#include <fstream>
#include <TNL/File.h>
#include <TNL/Object.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Matrices/CSRMatrix.h>
#include <TNL/legacy/matrices/tnlRgCSRMatrix.h>
#include <TNL/Solvers/IterativeSolverMonitor.h>
#include <TNL/Solvers/Linear/stationary/SOR.h>
#include <TNL/Solvers/Linear/CG.h>
#include <TNL/Solvers/Linear/BICGStab.h>
#include <TNL/Solvers/Linear/GMRES.h>
#include <TNL/Solvers/Linear/TFQMR.h>
#ifdef HAVE_PETSC
   #include <petsc.h>
#endif

#include "tnlConfig.h"
const char configFile[] = TNL_CONFIG_DIRECTORY "tnl-matrix-solvers-benchmark.cfg.desc";

void writeTestFailToLog( const Config::ParameterContainer& parameters )
{
   const String& logFileName = parameters. getParameter< String >( "log-file" );
   std::fstream logFile;
   if( logFileName != "" )
   {
      logFile. open( logFileName. getString(), std::ios::out | std::ios::app );
      if( ! logFile )
         std::cerr << "Unable to open the log file " << logFileName << std::endl;
      else
      {
         String bgColor( "#FF0000" );
         logFile << "             <td bgcolor=" << bgColor << "> N/A </td> " << std::endl
                 << "             <td bgcolor=" << bgColor << "> N/A </td> " << std::endl
                 << "             <td bgcolor=" << bgColor << "> N/A </td> " << std::endl;
         logFile. close();
      }
   }
}

template< typename Solver, typename Matrix, typename Vector >
bool benchmarkSolver( const Config::ParameterContainer&  parameters,
                      Solver& solver,
                      const Matrix& matrix,
                      const Vector& b,
                      Vector& x )
{
   typedef typename Matrix :: RealType RealType;
   typedef typename Matrix :: DeviceType DeviceType;
   typedef typename Matrix :: IndexType IndexType;

   const RealType& maxResidue = parameters. getParameter< double >( "max-residue" );
   const IndexType& size = matrix. getRows();
   const IndexType nonZeros = matrix. getNumberOfMatrixElements();
   //const IndexType maxIterations = size * ( ( double ) size * size / ( double ) nonZeros );
   const IndexType maxIterations = size;
  std::cout << "Setting max. number of iterations to " << maxIterations << std::endl;

   solver. setMatrix( matrix );
   solver. setMaxIterations( maxIterations );
   solver. setMaxResidue( maxResidue );
   solver. setMinResidue( 1.0e9 );
   IterativeSolverMonitor< RealType, IndexType > solverMonitor;
   solver. setSolverMonitor( solverMonitor );
   solver. setRefreshRate( 10 );
   solverMonitor. resetTimers();
#ifdef HAVE_NOT_CXX11
   solver. template solve< Vector, LinearResidueGetter< Matrix, Vector > >( b, x );
#else
   solver. solve( b, x );
#endif

   bool solverConverged( solver. getResidue() < maxResidue );
   const String& logFileName = parameters. getParameter< String >( "log-file" );
   std::fstream logFile;
   if( logFileName != "" )
   {
      logFile. open( logFileName. getString(), std::ios::out | std::ios::app );
      if( ! logFile )
         std::cerr << "Unable to open the log file " << logFileName << std::endl;
      else
      {
         String bgColor( "#FF0000" );
         if( solver. getResidue() < 1 )
            bgColor="#FF8888";
         if( solver. getResidue() < maxResidue )
         {
            bgColor="#88FF88";
         }
         double cpuTime = solverMonitor. getCPUTime();
         double realTime = solverMonitor. getRealTime();
         logFile << "             <td bgcolor=" << bgColor << "> " << solver. getResidue() << " </td> " << std::endl
                 << "             <td bgcolor=" << bgColor << "> " << solver. getIterations() << " </td> " << std::endl
                 << "             <td bgcolor=" << bgColor << "> " << cpuTime << " </td> " << std::endl;
         logFile. close();
      }
   }
   return solverConverged;

}

template< typename Matrix, typename Vector >
bool benchmarkMatrixOnDevice( const Config::ParameterContainer&  parameters,
                              const Matrix& matrix,
                              const Vector& b,
                              Vector& x )
{
   typedef typename Matrix :: RealType RealType;
   typedef typename Matrix :: DeviceType DeviceType;
   typedef typename Matrix :: IndexType IndexType;

   const String& solverClass = parameters. getParameter< String >( "solver-class" );
   if( solverClass == "tnl" )
   {
      const String& solverName = parameters. getParameter< String >( "solver-name" );
      IndexType iterations( 0 );
      RealType residue( 0.0 );
      bool converged( false );
      if( solverName == "sor" )
      {
         SOR< Matrix > solver;
         const RealType& sorOmega = parameters. getParameter< double >( "sor-omega" );
         solver. setOmega( sorOmega );
         return benchmarkSolver( parameters, solver, matrix, b, x );
      }
      if( solverName == "cg" )
      {
         CG< Matrix > solver;
         return benchmarkSolver( parameters, solver, matrix, b, x );
      }
      if( solverName == "bicgstab" )
      {
         BICGStab< Matrix > solver;
         return benchmarkSolver( parameters, solver, matrix, b, x );
      }
      if( solverName == "gmres" )
      {
         GMRES< Matrix > solver;
         const IndexType& gmresRestarting = parameters. getParameter< int >( "gmres-restarting" );
         solver. setRestarting( gmresRestarting );
         return benchmarkSolver( parameters, solver, matrix, b, x );
      }
      if( solverName == "tfqmr" )
      {
         TFQMR< Matrix > solver;
         return benchmarkSolver( parameters, solver, matrix, b, x );
      }
      std::cerr << "Unknown solver " << solverName << std::endl;
      return false;
   }
   if( solverClass == "petsc" )
   {
#ifndef HAVE_PETSC
      std::cerr << "PETSC is not installed on this system." << std::endl;
      writeTestFailToLog( parameters );
      return false;
#else
      if( DeviceType :: getDeviceType() != "Devices::Host" )
      {
         std::cerr << "PETSC tests can run only on host. The current device is " << DeviceType :: getDeviceType() << std::endl;
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
      Array< PetscScalar > petscVals;
      Array< PetscInt > petscCols;
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
               std::cerr << "Conversion to PETSC matrix was not correct at position " << i << " " << j << "." << std::endl;
               std::cerr << "Values are " << value << " and " << matrix. getElement( i, j ) << std::endl;
               return false;
            }
         }
      std::cerr << "PETSC CONVERSION WAS OK!!!" << std::endl;
      return true;*/

      Vec petscB, petscX;
      KSP ksp;
      KSPCreate( PETSC_COMM_WORLD, &ksp );


#endif
   }

}


template< typename Real, typename Index >
bool benchmarkMatrix( const Config::ParameterContainer&  parameters )
{
   /****
    * Loading the matrix from the input file
    */
   typedef CSRMatrix< Real, Devices::Host, Index > csrMatrixType;
   String inputFile = parameters. getParameter< String >( "input-file" );
   csrMatrixType csrMatrix;
   if( ! csrMatrix. load( inputFile ) )
   {
      std::cerr << "Unable to load file " << inputFile << std::endl;
      return false;
   }

   /****
    * Writing matrix statistics
    */
   String matrixStatsFileName = parameters. getParameter< String >( "matrix-stats-file" );
   if( matrixStatsFileName )
   {
      std::fstream matrixStatsFile;
      matrixStatsFile. open( matrixStatsFileName. getString(), std::ios::out );
      if( ! matrixStatsFile )
      {
         std::cerr << "Unable to open matrix statistics file " << matrixStatsFileName << std::endl;
         return false;
      }
      matrixStatsFile << "             <td> " << csrMatrix. getRows() << " </td> " << std::endl
                      << "             <td> " << csrMatrix. getNumberOfMatrixElements() << " </td> " << std::endl;
      matrixStatsFile. close();
   }

   /****
    * Setting up the linear problem
    */
   const Index size = csrMatrix. getRows();
  std::cout << "Matrix size is " << size << std::endl;
   Vector< Real, Devices::Host, Index > x1( "matrix-solvers-benchmark:x1" );
   Vector< Real, Devices::Host, Index > x( "matrix-solvers-benchmark:x" );
   Vector< Real, Devices::Host, Index > b( "matrix-solvers-benchmark:b" );
   if( ! x1. setSize( size ) ||
       ! x. setSize( size ) ||
       ! b. setSize( size ) )
   {
      std::cerr << "Sorry, I do not have enough memory for the benchmark." << std::endl;
      return false;
   }
   x1. setValue( ( Real ) 1.0 );
   x. setValue( ( Real ) 0.0 );
   csrMatrix. vectorProduct( x1, b );

   const String device = parameters. getParameter< String >( "device" );
   if( device == "host" )
      if( ! benchmarkMatrixOnDevice( parameters, csrMatrix, b, x ) )
         return false;

   if( device == "cuda" )
   {
#ifdef HAVE_CUDA
      tnlRgCSRMatrix< Real, Devices::Cuda, Index > rgCSRMatrix( "matrix-solvers-benchmark:rgCSRMatrix" );
      // FIX THIS
      //rgCSRMatrix = csrMatrix;
      /*Vector< Real, Devices::Cuda, Index > cudaX( "matrix-solvers-benchmark:cudaX" );
      Vector< Real, Devices::Cuda, Index > cudaB( "matrix-solvers-benchmark:cudaB" );
      cudaX. setLike( x );
      cudaX = x;
      cudaB. setLike( b );
      cudaB = b;
      if( ! benchmarkMatrixOnDevice( parameters, rgCSRMatrix, cudaB, cudaX ) )
         return false;
      x = cudaX;*/
#else
      CudaSupportMissingMessage;;
      return false;
#endif
   }

  std::cout << std::endl << "L1 diff. norm = " << x. differenceLpNorm( x1, ( Real ) 1.0 )
        << " L2 diff. norm = " << x. differenceLpNorm( x1, ( Real ) 2.0 )
        << " Max. diff. norm = " << x. differenceMax( x1 ) << std::endl;
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
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   if( conf_desc.parseConfigDescription( configFile ) != 0 )
      return 1;
   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc.printUsage( argv[ 0 ] );
      return 1;
   }
   String inputFile = parameters. getParameter< String >( "input-file" );
   String str_input_mtx_file = parameters. getParameter< String >( "input-mtx-file" );
   String log_file_name = parameters. getParameter< String >( "log-file" );
   double stop_time = parameters. getParameter< double >( "stop-time" );
   int verbose = parameters. getParameter< int >( "verbose");

   /****
    * Checking a type of the input data
    */
   String objectType;
   if( ! getObjectType( inputFile, objectType ) )
   {
      std::cerr << "Unable to detect object type in " << inputFile << std::endl;
      return EXIT_FAILURE;
   }
   List< String > parsedObjectType;
   parseObjectType( objectType,
                    parsedObjectType );
   String objectClass = parsedObjectType[ 0 ];
   if( objectClass != "CSRMatrix" )
   {
      std::cerr << "I am sorry, I am expecting CSRMatrix in the input file but I found " << objectClass << "." << std::endl;
      return EXIT_FAILURE;
   }

   String precision = parsedObjectType[ 1 ];
   //String indexing = parsedObjectType[ 3 ];
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

   std::fstream log_file;
   if( log_file_name )
   {
      log_file. open( log_file_name. getString(), std::ios::out | std::ios::app );
      if( ! log_file )
      {
         std::cerr << "Unable to open log file " << log_file_name << " for appending logs." << std::endl;
         return EXIT_FAILURE;
      }
     std::cout << "Writing to log file " << log_file_name << "..." << std::endl;
   }
#ifdef HAVE_PETSC
   PetscFinalize();
#endif
   return EXIT_SUCCESS;

}


#endif /* MATRIXSOLVERSBENCHMARK_H_ */
