/***************************************************************************
                          tnl-benchmark-ode-solvers.h  -  description
                             -------------------
    begin                : Jul 13, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber

#pragma once

#include <set>
#include <sstream>
#include <string>

#ifndef NDEBUG
#include <TNL/Debugging/FPE.h>
#endif

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/MPI/ScopedInitializer.h>
#include <TNL/MPI/Config.h>
#include <TNL/Solvers/ODE/Euler.h>
#include <TNL/Solvers/ODE/Merson.h>

#include "../Benchmarks.h"
#include "benchmarks.h"
#include "SimpleProblem.h"
#include "Euler.h"
#include "Merson.h"

using namespace TNL;
using namespace TNL::Benchmarks;
using namespace TNL::Pointers;


template< typename Real, typename Index >
void
benchmarkODESolvers( Benchmark& benchmark,
                     const Config::ParameterContainer& parameters,
                     size_t dofs )
{
   using HostVectorType = Containers::Vector< Real, Devices::Host, Index >;
   using CudaVectorType = Containers::Vector< Real, Devices::Cuda, Index >;
   using HostVectorPointer = Pointers::SharedPointer< HostVectorType >;
   using CudaVectorPointer = Pointers::SharedPointer< CudaVectorType >;
   using HostProblem = SimpleProblem< Real, Devices::Host, Index >;
   using CudaProblem = SimpleProblem< Real, Devices::Cuda, Index >;
   using SolverMonitorType = typename Benchmark::SolverMonitorType;

   const auto& solvers = parameters.getList< String >( "solvers" );
   for( auto&& solver : solvers )
   {
      HostVectorPointer host_u( dofs );
      *host_u = 0.0;
#ifdef HAVE_CUDA
      CudaVectorPointer cuda_u( dofs );
      *cuda_u = 0.0;
#endif
      if( solver == "euler" || solver == "all" ) {
         using HostSolver = Solvers::ODE::Euler< HostProblem, SolverMonitorType >;
         benchmark.setOperation("Euler");
         benchmarkSolver< HostSolver >( benchmark, parameters, host_u );
         using HostSolverNonET = Benchmarks::Euler< HostProblem, SolverMonitorType >;
         benchmark.setOperation("Euler non-ET");
         benchmarkSolver< HostSolverNonET >( benchmark, parameters, host_u );
#ifdef HAVE_CUDA
         using CudaSolver = Solvers::ODE::Euler< CudaProblem, SolverMonitorType >;
         benchmark.setOperation( "Euler" );
         benchmarkSolver< CudaSolver >( benchmark, parameters, cuda_u );
         using CudaSolverNonET = Benchmarks::Euler< CudaProblem, SolverMonitorType >;
         benchmark.setOperation("Euler non-ET");
         benchmarkSolver< CudaSolverNonET >( benchmark, parameters, cuda_u );
#endif
      }

      if( solver == "merson" || solver == "all" ) {
         using HostSolver = Solvers::ODE::Merson< HostProblem, SolverMonitorType >;
         benchmark.setOperation("Merson");
         benchmarkSolver< HostSolver >( benchmark, parameters, host_u );
         using HostSolverNonET = Benchmarks::Merson< HostProblem, SolverMonitorType >;
         benchmark.setOperation("Merson non-ET");
         benchmarkSolver< HostSolverNonET >( benchmark, parameters, host_u );
#ifdef HAVE_CUDA
         using CudaSolver = Solvers::ODE::Merson< CudaProblem, SolverMonitorType >;
         benchmark.setOperation("Merson");
         benchmarkSolver< CudaSolver >( benchmark, parameters, cuda_u );
         using CudaSolverNonET = Benchmarks::Merson< CudaProblem, SolverMonitorType >;
         benchmark.setOperation("Merson non-ET");
         benchmarkSolver< CudaSolverNonET >( benchmark, parameters, cuda_u );
#endif
      }
   }
}

template< typename Real, typename Index >
struct ODESolversBenchmark
{
   using RealType = Real;
   using IndexType = Index;
   using VectorType = Containers::Vector< RealType, Devices::Host, IndexType >;
   using VectorPointer = Pointers::SharedPointer< VectorType >;

   static bool
   run( Benchmark& benchmark,
        Benchmark::MetadataMap metadata,
        const Config::ParameterContainer& parameters )
   {
      const String name = String( (TNL::MPI::GetSize() > 1) ? "Distributed ODE solvers" : "ODE solvers" );
                          //+ " (" + parameters.getParameter< String >( "name" ) + "): ";
      benchmark.newBenchmark( name, metadata );
      for( size_t dofs = 25; dofs <= 10000000; dofs *= 2 ) {
         benchmark.setMetadataColumns( Benchmark::MetadataColumns({
            // TODO: strip the device
            { "DOFs", convertToString( dofs ) },
         } ));

         if( TNL::MPI::GetSize() > 1 )
            runDistributed( benchmark, metadata, parameters, dofs );
         else
            runNonDistributed( benchmark, metadata, parameters, dofs );
      }
      return true;
   }

   static void
   runDistributed( Benchmark& benchmark,
                   Benchmark::MetadataMap metadata,
                   const Config::ParameterContainer& parameters,
                   size_t dofs )
   {
      //const auto group = TNL::MPI::AllGroup();

      std::cout << "Iterative solvers:" << std::endl;
      benchmarkODESolvers< Real, Index >( benchmark, parameters, dofs );
   }

   static void
   runNonDistributed( Benchmark& benchmark,
                      Benchmark::MetadataMap metadata,
                      const Config::ParameterContainer& parameters,
                      size_t dofs )
   {
      std::cout << "Iterative solvers:" << std::endl;
      benchmarkODESolvers< Real, Index >( benchmark, parameters, dofs );
   }
};

template< typename Real >
bool resolveIndexType( Benchmark& benchmark,
   Benchmark::MetadataMap& metadata,
   Config::ParameterContainer& parameters )
{
   const String& index = parameters.getParameter< String >( "index-type" );
   if( index == "int" ) return ODESolversBenchmark< Real, int >::run( benchmark, metadata, parameters );
   return ODESolversBenchmark< Real, long int >::run( benchmark, metadata, parameters );
}

bool resolveRealTypes( Benchmark& benchmark,
   Benchmark::MetadataMap& metadata,
   Config::ParameterContainer& parameters )
{
   const String& realType = parameters.getParameter< String >( "real-type" );
   if( ( realType == "float" || realType == "all" ) &&
       ! resolveIndexType< float >( benchmark, metadata, parameters ) )
      return false;
   if( ( realType == "double" || realType == "all" ) &&
       ! resolveIndexType< double >( benchmark, metadata, parameters ) )
      return false;
   return true;
}

void
configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-linear-solvers.log");
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< int >( "loops", "Number of repetitions of the benchmark.", 10 );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
   config.addList< String >( "solvers", "List of solvers to run benchmarks for.", {"all"} );
   config.addEntryEnum< String >( "euler" );
   config.addEntryEnum< String >( "merson" );
   config.addEntryEnum< String >( "all" );
   config.addEntry< String >( "real-type", "Run benchmarks with given precision.", "all" );
   config.addEntryEnum< String >( "float" );
   config.addEntryEnum< String >( "double" );
   config.addEntryEnum< String >( "all" );
   config.addEntry< String >( "index-type", "Run benchmarks with given index type.", "int" );
   config.addEntryEnum< String >( "int" );
   config.addEntryEnum< String >( "long int" );
   config.addEntry< double >( "final-time", "Final time of the benchmark test.", 1.0 );
   config.addEntry< double >( "time-step", "Time step of the benchmark test.", 1.0e-2 );

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
   TNL::MPI::configSetup( config );

   config.addDelimiter( "ODE solver settings:" );
   Solvers::IterativeSolver< double, int >::configSetup( config );
   using Problem = SimpleProblem< double, Devices::Host, int >;
   Solvers::ODE::Euler< Problem >::configSetup( config );
   Solvers::ODE::Merson< Problem >::configSetup( config );
}

int
main( int argc, char* argv[] )
{
#ifndef NDEBUG
   Debugging::trackFloatingPointExceptions();
#endif

   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   configSetup( conf_desc );

   TNL::MPI::ScopedInitializer mpi(argc, argv);
   const int rank = TNL::MPI::GetRank();

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;
   if( ! Devices::Host::setup( parameters ) ||
       ! Devices::Cuda::setup( parameters ) ||
       ! TNL::MPI::setup( parameters ) )
      return EXIT_FAILURE;

   const String & logFileName = parameters.getParameter< String >( "log-file" );
   const String & outputMode = parameters.getParameter< String >( "output-mode" );
   const int loops = parameters.getParameter< int >( "loops" );
   const int verbose = (rank == 0) ? parameters.getParameter< int >( "verbose" ) : 0;

   // open log file
   auto mode = std::ios::out;
   if( outputMode == "append" )
       mode |= std::ios::app;
   std::ofstream logFile;
   if( rank == 0 )
      logFile.open( logFileName.getString(), mode );

   // init benchmark and common metadata
   Benchmark benchmark( loops, verbose );

   // prepare global metadata
   Benchmark::MetadataMap metadata = getHardwareMetadata();

   const bool status = resolveRealTypes( benchmark, metadata, parameters );

   if( rank == 0 )
      if( ! benchmark.save( logFile ) ) {
         std::cerr << "Failed to write the benchmark results to file '" << parameters.getParameter< String >( "log-file" ) << "'." << std::endl;
         return EXIT_FAILURE;
      }

   return ! status;
}
