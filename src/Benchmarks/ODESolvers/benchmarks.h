/***************************************************************************
                          benchmarks.h  -  description
                             -------------------
    begin                : Jul 13, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber

#pragma once

#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Config/ParameterContainer.h>

#include "../Benchmarks.h"


#include <stdexcept>  // std::runtime_error

using namespace TNL;
using namespace TNL::Pointers;
using namespace TNL::Benchmarks;


template< typename Device >
const char*
getPerformer()
{
   if( std::is_same< Device, Devices::Cuda >::value )
      return "GPU";
   return "CPU";
}

/*template< typename Matrix >
void barrier( const Matrix& matrix )
{
}

template< typename Matrix, typename Communicator >
void barrier( const Matrices::DistributedMatrix< Matrix, Communicator >& matrix )
{
   Communicator::Barrier( matrix.getCommunicationGroup() );
}*/

template< typename Device >
bool checkDevice( const Config::ParameterContainer& parameters )
{
   const String device = parameters.getParameter< String >( "device" );
   if( device == "all" )
      return true;
   if( std::is_same< Device, Devices::Host >::value && device == "host" )
      return true;
   if( std::is_same< Device, Devices::Cuda >::value && device == "cuda" )
      return true;
   return false;
}


template< typename Solver, typename Problem, typename VectorPointer >
void
benchmarkSolver( Benchmark& benchmark,
                 const Config::ParameterContainer& parameters,
                 Problem& problem,
                 VectorPointer& u )
{
   using DeviceType = typename Problem::DeviceType;
   // skip benchmarks on devices which the user did not select
   if( ! checkDevice< DeviceType >( parameters ) )
      return;

   const char* performer = getPerformer< DeviceType >();

   // setup
   Solver solver;
   solver.setup( parameters );
   solver.setProblem( problem );
   
   // FIXME: getMonitor returns solver monitor specialized for double and int
   //solver.setSolverMonitor( benchmark.getMonitor() );

   // reset function
   auto reset = [&]() {
      *u = 0.0;
   };

   // benchmark function
   auto compute = [&]() {
      bool converged = solver.solve( u );
      //barrier( matrix );
      if( ! converged )
         throw std::runtime_error("solver did not converge");
   };

   // subclass BenchmarkResult to add extra columns to the benchmark
   // (iterations, preconditioned residue, true residue)
   /*struct MyBenchmarkResult : public BenchmarkResult
   {
      using HeaderElements = BenchmarkResult::HeaderElements;
      using RowElements = BenchmarkResult::RowElements;

      Solver< Matrix >& solver;
      const SharedPointer< Matrix >& matrix;
      const Vector& x;
      const Vector& b;

      MyBenchmarkResult( Solver< Matrix >& solver,
                         const SharedPointer< Matrix >& matrix,
                         const Vector& x,
                         const Vector& b )
      : solver(solver), matrix(matrix), x(x), b(b)
      {}

      virtual HeaderElements getTableHeader() const override
      {
         return HeaderElements({"time", "speedup", "converged", "iterations", "residue_precond", "residue_true"});
      }

      virtual RowElements getRowElements() const override
      {
         const bool converged = ! std::isnan(solver.getResidue()) && solver.getResidue() < solver.getConvergenceResidue();
         const long iterations = solver.getIterations();
         const double residue_precond = solver.getResidue();

         Vector r;
         r.setLike( x );
         matrix->vectorProduct( x, r );
         r.addVector( b, 1.0, -1.0 );
         const double residue_true = lpNorm( r.getView(), 2.0 ) / lpNorm( b.getView(), 2.0 );

         return RowElements({ time, speedup, (double) converged, (double) iterations,
                              residue_precond, residue_true });
      }
   };
   MyBenchmarkResult benchmarkResult( solver, matrix, x, b );*/

   benchmark.time< DeviceType >( reset, performer, compute ); //, benchmarkResult );
}

