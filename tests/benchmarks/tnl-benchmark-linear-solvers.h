/***************************************************************************
                          tnl-benchmark-linear-solvers.h  -  description
                             -------------------
    begin                : Jun 22, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNL_BENCHMARK_LINEAR_SOLVERS_H_
#define TNL_BENCHMARK_LINEAR_SOLVERS_H_

#include <fstream>
#include <iomanip>
#include <unistd.h>

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/TimerRT.h>
#include <TNL/matrices/tnlDenseMatrix.h>
#include <TNL/matrices/tnlTridiagonalMatrix.h>
#include <TNL/matrices/tnlMultidiagonalMatrix.h>
#include <TNL/matrices/tnlCSRMatrix.h>
#include <TNL/matrices/tnlEllpackMatrix.h>
#include <TNL/matrices/tnlSlicedEllpackMatrix.h>
#include <TNL/matrices/tnlChunkedEllpackMatrix.h>
#include <TNL/matrices/tnlMatrixReader.h>
#include <TNL/solvers/linear/krylov/tnlGMRESSolver.h>
#include <TNL/solvers/linear/krylov/tnlCGSolver.h>
#include <TNL/solvers/linear/krylov/tnlBICGStabSolver.h>
#include <TNL/solvers/linear/krylov/tnlTFQMRSolver.h>
#include <TNL/solvers/linear/tnlLinearResidueGetter.h>
#include <TNL/solvers/tnlIterativeSolverMonitor.h>

using namespace std;
using namespace TNL;

void configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter                            ( "General settings:" );
   config.addRequiredEntry< String >( "test" , "Test to be performed." );
      config.addEntryEnum< String >( "mtx" );
      config.addEntryEnum< String >( "tnl" );
   config.addRequiredEntry< String >( "input-file" , "Input binary file name." );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-linear-solvers.log");
   config.addEntry< String >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntry< String >( "matrix-format", "Matrix format.", "csr" );
      config.addEntryEnum< String >( "dense" );
      config.addEntryEnum< String >( "tridiagonal" );
      config.addEntryEnum< String >( "multidiagonal" );
      config.addEntryEnum< String >( "ellpack" );
      config.addEntryEnum< String >( "sliced-ellpack" );
      config.addEntryEnum< String >( "chunked-ellpack" );
      config.addEntryEnum< String >( "csr" );
   config.addEntry< String >( "solver", "Linear solver.", "gmres" );
      config.addEntryEnum< String >( "sor" );
      config.addEntryEnum< String >( "cg" );
      config.addEntryEnum< String >( "gmres" );
   config.addEntry< String >( "device", "Device.", "host" );
      config.addEntryEnum< String >( "host" );
      config.addEntryEnum< String >( "cuda" );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
}

template< typename Solver >
bool benchmarkSolver( const Config::ParameterContainer& parameters,
                      const typename Solver::MatrixType& matrix)
{
   typedef typename Solver::MatrixType MatrixType;
   typedef typename MatrixType::RealType RealType;
   typedef typename MatrixType::DeviceType DeviceType;
   typedef typename MatrixType::IndexType IndexType;
   typedef Vectors::Vector< RealType, DeviceType, IndexType > VectorType;

   VectorType x, y, b;
   x.setSize( matrix.getColumns() );
   x.setValue( 1.0 / ( RealType ) matrix.getColumns() );
   y.setSize( matrix.getColumns() );
   b.setSize( matrix.getRows() );
   matrix.vectorProduct( x, b );

   Solver solver;
   tnlIterativeSolverMonitor< RealType, IndexType > monitor;
   monitor.setVerbose( 1 );
   solver.setSolverMonitor( monitor );
   solver.setMatrix( matrix );
   solver.setConvergenceResidue( 1.0e-6 );
   solver.template solve< VectorType, tnlLinearResidueGetter< MatrixType, VectorType > >( b, y );
   std::cout << std::endl;
   return true;
}

template< typename Matrix >
bool readMatrix( const Config::ParameterContainer& parameters,
                 Matrix& matrix )
{
   const String fileName = parameters.getParameter< String >( "input-file" );

   Matrix* hostMatrix;
   if( Matrix::DeviceType::DeviceType == ( int ) tnlCudaDevice )
   {

   }
   if( Matrix::DeviceType::DeviceType == ( int ) tnlHostDevice )
   {
      hostMatrix = &matrix;
      try
      {
         if( ! tnlMatrixReader< Matrix >::readMtxFile( fileName, *hostMatrix ) )
         {
            std::cerr << "I am not able to read the matrix file " << fileName << "." << std::endl;
            /*logFile << std::endl;
            logFile << inputFileName << std::endl;
            logFile << "Benchmark failed: Unable to read the matrix." << std::endl;*/
            return false;
         }
      }
      catch( std::bad_alloc )
      {
         std::cerr << "Not enough memory to read the matrix." << std::endl;
         /*logFile << std::endl;
         logFile << inputFileName << std::endl;
         logFile << "Benchmark failed: Not enough memory." << std::endl;*/
         return false;
      }
   }
   return true;
}

template< typename Matrix >
bool resolveLinearSolver( const Config::ParameterContainer& parameters )
{
   const String& solver = parameters.getParameter< String >( "solver" );

   Matrix matrix;
   if( ! readMatrix( parameters, matrix ) )
      return false;

   if( solver == "gmres" )
      return benchmarkSolver< tnlGMRESSolver< Matrix > >( parameters, matrix );

   if( solver == "cg" )
      return benchmarkSolver< tnlCGSolver< Matrix > >( parameters, matrix );

   if( solver == "bicgstab" )
      return benchmarkSolver< tnlBICGStabSolver< Matrix > >( parameters, matrix );

   if( solver == "tfqmr" )
      return benchmarkSolver< tnlTFQMRSolver< Matrix > >( parameters, matrix );

   std::cerr << "Unknown solver " << solver << "." << std::endl;
   return false;
}

template< typename Real,
          typename Device >
bool resolveMatrixFormat( const Config::ParameterContainer& parameters )
{
   const String& matrixFormat = parameters.getParameter< String >( "matrix-format" );

   if( matrixFormat == "dense" )
      return resolveLinearSolver< tnlDenseMatrix< Real, Device, int > >( parameters );

   if( matrixFormat == "tridiagonal" )
      return resolveLinearSolver< tnlTridiagonalMatrix< Real, Device, int > >( parameters );

   if( matrixFormat == "multidiagonal" )
      return resolveLinearSolver< tnlMultidiagonalMatrix< Real, Device, int > >( parameters );

   if( matrixFormat == "ellpack" )
      return resolveLinearSolver< tnlEllpackMatrix< Real, Device, int > >( parameters );

   if( matrixFormat == "sliced-ellpack" )
      return resolveLinearSolver< tnlSlicedEllpackMatrix< Real, Device, int > >( parameters );

   if( matrixFormat == "chunked-ellpack" )
      return resolveLinearSolver< tnlChunkedEllpackMatrix< Real, Device, int > >( parameters );

   if( matrixFormat == "csr" )
      return resolveLinearSolver< tnlCSRMatrix< Real, Device, int > >( parameters );

   std::cerr << "Unknown matrix format " << matrixFormat << "." << std::endl;
   return false;
}

template< typename Real >
bool resolveDevice( const Config::ParameterContainer& parameters )
{
   const String& device = parameters.getParameter< String >( "device" );

   if( device == "host" )
      return resolveMatrixFormat< Real, tnlHost >( parameters );

   if( device == "cuda" )
      return resolveMatrixFormat< Real, tnlCuda >( parameters );

   std::cerr << "Uknown device " << device << "." << std::endl;
   return false;
}

int main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   configSetup( conf_desc );
 
   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc.printUsage( argv[ 0 ] );
      return 1;
   }
   const String& precision = parameters.getParameter< String >( "precision" );
   if( precision == "float" )
      if( ! resolveDevice< float >( parameters ) )
         return EXIT_FAILURE;
   if( precision == "double" )
      if( ! resolveDevice< double >( parameters ) )
         return EXIT_FAILURE;
   return EXIT_SUCCESS;
}


#endif /* TNL_BENCHMARK_LINEAR_SOLVERS_H_ */
