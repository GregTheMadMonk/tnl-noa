/***************************************************************************
                          tnl-benchmark-linear-solvers.h  -  description
                             -------------------
    begin                : Jun 22, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNL_BENCHMARK_LINEAR_SOLVERS_H_
#define TNL_BENCHMARK_LINEAR_SOLVERS_H_

#include <fstream>
#include <iomanip>
#include <unistd.h>

#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <core/tnlTimerRT.h>
#include <matrices/tnlDenseMatrix.h>
#include <matrices/tnlTridiagonalMatrix.h>
#include <matrices/tnlMultidiagonalMatrix.h>
#include <matrices/tnlCSRMatrix.h>
#include <matrices/tnlEllpackMatrix.h>
#include <matrices/tnlSlicedEllpackMatrix.h>
#include <matrices/tnlChunkedEllpackMatrix.h>
#include <matrices/tnlMatrixReader.h>
#include <solvers/linear/krylov/tnlGMRESSolver.h>
#include <solvers/linear/krylov/tnlCGSolver.h>
#include <solvers/linear/krylov/tnlBICGStabSolver.h>
#include <solvers/linear/krylov/tnlTFQMRSolver.h>
#include <solvers/linear/tnlLinearResidueGetter.h>
#include <solvers/tnlIterativeSolverMonitor.h>

using namespace std;

#include "tnlConfig.h"
const char configFile[] = TNL_CONFIG_DIRECTORY "tnl-benchmark-linear-solvers.cfg.desc";

void configSetup( tnlConfigDescription& config )
{
   config.addDelimiter                            ( "General settings:" );
   config.addRequiredEntry< tnlString >( "test" , "Test to be performed." );
      config.addEntryEnum< tnlString >( "tridiagonal" );
      config.addEntryEnum< tnlString >( "multidiagonal" );
      config.addEntryEnum< tnlString >( "multidiagonal-with-long-rows" );
      config.addEntryEnum< tnlString >( "mtx" );
      config.addEntryEnum< tnlString >( "tnl" );
   config.addRequiredEntry< tnlString >( "input-file" , "Input binary file name." );
   config.addEntry< tnlString >( "log-file", "Log file name.", "tnl-benchmark-linear-solvers.log");
   config.addEntry< tnlString >( "precison", "Precision of the arithmetics.", "double" );
   config.addEntry< tnlString >( "matrix-format", "Matrix format.", "csr" );
      config.addEntryEnum< tnlString >( "dense" );
      config.addEntryEnum< tnlString >( "tridiagonal" );
      config.addEntryEnum< tnlString >( "multidiagonal" );
      config.addEntryEnum< tnlString >( "ellpack" );
      config.addEntryEnum< tnlString >( "sliced-ellpack" );
      config.addEntryEnum< tnlString >( "chunked-ellpack" );
      config.addEntryEnum< tnlString >( "csr" );
   config.addEntry< tnlString >( "solver", "Linear solver.", "gmres" );
      config.addEntryEnum< tnlString >( "sor" );
      config.addEntryEnum< tnlString >( "cg" );
      config.addEntryEnum< tnlString >( "gmres" );
   config.addEntry< tnlString >( "device", "Device.", "host" );
      config.addEntryEnum< tnlString >( "host" );
      config.addEntryEnum< tnlString >( "cuda" );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
}

template< typename Solver >
bool benchmarkSolver( const tnlParameterContainer& parameters,
                      const typename Solver::MatrixType& matrix)
{
   typedef typename Solver::MatrixType MatrixType;
   typedef typename MatrixType::RealType RealType;
   typedef typename MatrixType::DeviceType DeviceType;
   typedef typename MatrixType::IndexType IndexType;
   typedef tnlVector< RealType, DeviceType, IndexType > VectorType;

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
   cout << endl;
   return true;
}

template< typename Matrix >
bool readMatrix( const tnlParameterContainer& parameters,
                 Matrix& matrix )
{
   const tnlString fileName = parameters.GetParameter< tnlString >( "input-file" );

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
            cerr << "I am not able to read the matrix file " << fileName << "." << endl;
            /*logFile << endl;
            logFile << inputFileName << endl;
            logFile << "Benchmark failed: Unable to read the matrix." << endl;*/
            return false;
         }
      }
      catch( std::bad_alloc )
      {
         cerr << "Not enough memory to read the matrix." << endl;
         /*logFile << endl;
         logFile << inputFileName << endl;
         logFile << "Benchmark failed: Not enough memory." << endl;*/
         return false;
      }
   }
   return true;
}

template< typename Matrix >
bool resolveLinearSolver( const tnlParameterContainer& parameters )
{
   const tnlString& solver = parameters.GetParameter< tnlString >( "solver" );

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

   cerr << "Unknown solver " << solver << "." << endl;
   return false;
}

template< typename Real,
          typename Device >
bool resolveMatrixFormat( const tnlParameterContainer& parameters )
{
   const tnlString& matrixFormat = parameters.GetParameter< tnlString >( "matrix-format" );

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

   cerr << "Unknown matrix format " << matrixFormat << "." << endl;
   return false;
}

template< typename Real >
bool resolveDevice( const tnlParameterContainer& parameters )
{
   const tnlString& device = parameters.GetParameter< tnlString >( "device" );

   if( device == "host" )
      return resolveMatrixFormat< Real, tnlHost >( parameters );

   if( device == "cuda" )
      return resolveMatrixFormat< Real, tnlCuda >( parameters );

   cerr << "Uknown device " << device << "." << endl;
   return false;
}

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;

   configSetup( conf_desc );
   
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc.printUsage( argv[ 0 ] );
      return 1;
   }
   const tnlString& precision = parameters.GetParameter< tnlString >( "precision" );
   if( precision == "float" )
      if( ! resolveDevice< float >( parameters ) )
         return EXIT_FAILURE;
   if( precision == "double" )
      if( ! resolveDevice< double >( parameters ) )
         return EXIT_FAILURE;
   return EXIT_SUCCESS;
}


#endif /* TNL_BENCHMARK_LINEAR_SOLVERS_H_ */
