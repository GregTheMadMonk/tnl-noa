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

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Communicators/NoDistrCommunicator.h>
#include <TNL/Communicators/ScopedInitializer.h>
#include <TNL/Solvers/ODE/Euler.h>
#include <TNL/Solvers/ODE/Merson.h>

#include "../Benchmarks.h"
#include "benchmarks.h"
#include "SimpleProblem.h"


#include <TNL/Matrices/SlicedEllpack.h>

using namespace TNL;
using namespace TNL::Benchmarks;
using namespace TNL::Pointers;

#ifdef HAVE_MPI
using CommunicatorType = Communicators::MpiCommunicator;
#else
using CommunicatorType = Communicators::NoDistrCommunicator;
#endif


static const std::set< std::string > valid_solvers = {
   "euler",
   "merson",
};

std::set< std::string >
parse_comma_list( const Config::ParameterContainer& parameters,
                  const char* parameter,
                  const std::set< std::string >& options )
{
   const String solvers = parameters.getParameter< String >( parameter );

   if( solvers == "all" )
      return options;

   std::stringstream ss( solvers.getString() );
   std::string s;
   std::set< std::string > set;

   while( std::getline( ss, s, ',' ) ) {
      if( ! options.count( s ) )
         throw std::logic_error( std::string("Invalid value in the comma-separated list for the parameter '")
                                 + parameter + "': '" + s + "'. The list contains: '" + solvers.getString() + "'." );

      set.insert( s );

      if( ss.peek() == ',' )
         ss.ignore();
   }

   return set;
}

template< typename Problem, typename VectorPointer >
void
benchmarkODESolvers( Benchmark& benchmark,
                     const Config::ParameterContainer& parameters,
                     VectorPointer& u )
{
   Problem problem;
   const std::set< std::string > solvers = parse_comma_list( parameters, "solvers", valid_solvers );

   if( solvers.count( "euler" ) ) {
      using Solver = Solvers::ODE::Euler< Problem >;
      benchmark.setOperation("Euler");
      benchmarkSolver< Solver >( benchmark, parameters, problem, u );
      #ifdef HAVE_CUDA
      benchmarkSolver< Solver >( benchmark, parameters, problem, u );
      #endif
   }

   if( solvers.count( "merson" ) ) {
      using Solver = Solvers::ODE::Merson< Problem >;
      benchmark.setOperation("Merson");
      benchmarkSolver< Solver >( benchmark, parameters, problem, u );
      #ifdef HAVE_CUDA
      benchmarkSolver< Solver >( benchmark, parameters, problem, u );
      #endif
   }

}

template< typename Real, typename Device, typename Index >
struct ODESolversBenchmark
{
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;
   using VectorPointer = Pointers::SharedPointer< VectorType >;

   static bool
   run( Benchmark& benchmark,
        Benchmark::MetadataMap metadata,
        const Config::ParameterContainer& parameters )
   {
      const String name = String( (CommunicatorType::isDistributed()) ? "Distributed ODE solvers" : "ODE solvers" )
                          + " (" + parameters.getParameter< String >( "name" ) + "): ";
      benchmark.newBenchmark( name, metadata );
      for( int dofs = 25; dofs <= 100000; dofs *= 2 ) {
         benchmark.setMetadataColumns( Benchmark::MetadataColumns({
            // TODO: strip the device
            { "DOFs", convertToString( dofs ) },
         } ));

         Pointers::SharedPointer< VectorType > u( dofs );
         *u = 0.0;
         if( CommunicatorType::isDistributed() )
            runDistributed( benchmark, metadata, parameters, u );
         else
            runNonDistributed( benchmark, metadata, parameters, u );
      }
      return true;
   }

   static void
   runDistributed( Benchmark& benchmark,
                   Benchmark::MetadataMap metadata,
                   const Config::ParameterContainer& parameters,
                   VectorPointer& u )
   {
      using Problem = SimpleProblem< Real, Device, Index >;
      const auto group = CommunicatorType::AllGroup;

      std::cout << "Iterative solvers:" << std::endl;
      benchmarkODESolvers< Problem, VectorPointer >( benchmark, parameters, u );
   }

   static void
   runNonDistributed( Benchmark& benchmark,
                      Benchmark::MetadataMap metadata,
                      const Config::ParameterContainer& parameters,
                      VectorPointer& u )
   {
      using Problem = SimpleProblem< Real, Device, Index >;
      std::cout << "Iterative solvers:" << std::endl;
      benchmarkODESolvers< Problem, VectorPointer >( benchmark, parameters, u );
   }
};

template< typename Real, typename Device >
bool resolveIndexType( Benchmark& benchmark,
   Benchmark::MetadataMap& metadata,
   Config::ParameterContainer& parameters )
{
   const String& index = parameters.getParameter< String >( "index" );
   if( index == "int" ) return ODESolversBenchmark< Real, Device, int >::run( benchmark, metadata, parameters );
   if( index == "long int" ) return ODESolversBenchmark< Real, Device, long int >::run( benchmark, metadata, parameters );
}

template< typename Real >
bool resolveDeviceType( Benchmark& benchmark,
   Benchmark::MetadataMap& metadata,
   Config::ParameterContainer& parameters )
{
   const String& device = parameters.getParameter< String >( "device" );
   if( device == "host" ) return resolveIndexType< Real, Devices::Host >( benchmark, metadata, parameters );
#ifdef HAVE_CUDA
   if( device == "cuda" ) return resolveIndexType< Real, Devices::Cuda >( benchmark, metadata, parameters );
#endif
}

bool resolveRealType( Benchmark& benchmark,
   Benchmark::MetadataMap& metadata,
   Config::ParameterContainer& parameters )
{
   const String& real = parameters.getParameter< String >( "real" );
   if( real == "float" ) return resolveDeviceType< float >( benchmark, metadata, parameters );
   if( real == "double" ) return resolveDeviceType< double >( benchmark, metadata, parameters );
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
   config.addEntry< String >( "solvers", "Comma-separated list of solvers to run benchmarks for. Options: gmres, cwygmres, tfqmr, bicgstab, bicgstab-ell.", "all" );
   config.addEntry< String >( "devices", "Run benchmarks on these devices.", "all" );
   config.addEntryEnum( "all" );
   config.addEntryEnum( "host" );
   #ifdef HAVE_CUDA
   config.addEntryEnum( "cuda" );
   #endif

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
   CommunicatorType::configSetup( config );

   config.addDelimiter( "ODE solver settings:" );
   Solvers::IterativeSolver< double, int >::configSetup( config );
   Solvers::ODE::Euler<>::configSetup( config );
   Solvers::ODE::Merson<>::configSetup( config );
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

   Communicators::ScopedInitializer< CommunicatorType > scopedInit(argc, argv);
   const int rank = CommunicatorType::GetRank( CommunicatorType::AllGroup );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) ) {
      conf_desc.printUsage( argv[ 0 ] );
      return EXIT_FAILURE;
   }
   if( ! Devices::Host::setup( parameters ) ||
       ! Devices::Cuda::setup( parameters ) ||
       ! CommunicatorType::setup( parameters ) )
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

   const bool status = resolveRealType( benchmark, metadata, parameters );

   if( rank == 0 )
      if( ! benchmark.save( logFile ) ) {
         std::cerr << "Failed to write the benchmark results to file '" << parameters.getParameter< String >( "log-file" ) << "'." << std::endl;
         return EXIT_FAILURE;
      }

   return ! status;
}
