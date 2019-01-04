/***************************************************************************
                          tnl-benchmark-traversers.h  -  description
                             -------------------
    begin                : Dec 17, 2018
    copyright            : (C) 2018 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber

#pragma once

#include "../Benchmarks.h"
//#include "grid-traversing.h"
#include "GridTraversersBenchmark.h"

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/ParallelFor.h>
#include <TNL/Containers/List.h>

using namespace TNL;
using namespace TNL::Benchmarks;
using namespace TNL::Benchmarks::Traversers;


template< int Dimension,
          typename Real = float,
          typename Index = int >
bool runBenchmark( const Config::ParameterContainer& parameters,
                   Benchmark& benchmark,
                   Benchmark::MetadataMap& metadata )
{
   const Containers::List< String >& tests = parameters.getParameter< Containers::List< String > >( "tests" );
   // FIXME: getParameter< std::size_t >() does not work with parameters added with addEntry< int >(),
   // which have a default value. The workaround below works for int values, but it is not possible
   // to pass 64-bit integer values
   // const std::size_t minSize = parameters.getParameter< std::size_t >( "min-size" );
   // const std::size_t maxSize = parameters.getParameter< std::size_t >( "max-size" );
   const std::size_t minSize = parameters.getParameter< int >( "min-size" );
   const std::size_t maxSize = parameters.getParameter< int >( "max-size" );
#ifdef HAVE_CUDA
   const bool withCuda = parameters.getParameter< bool >( "with-cuda" );
#else
   const bool withCuda = false;
#endif

   /****
    * Full grid traversing with no boundary conditions
    */
   benchmark.newBenchmark( String("Traversing without boundary conditions" + convertToString( Dimension ) + "D" ), metadata );
   for( std::size_t size = minSize; size <= maxSize; size *= 2 )
   {
      GridTraversersBenchmark< Dimension, Devices::Host, Real, Index > hostTraverserBenchmark( size );
#ifdef HAVE_CUDA
      GridTraversersBenchmark< Dimension, Devices::Cuda, Real, Index > cudaTraverserBenchmark( size );
#endif

      auto hostReset = [&]()
      {
         hostTraverserBenchmark.reset();
      };

#ifdef HAVE_CUDA
      auto cudaReset = [&]()
      {
         cudaTraverserBenchmark.reset();
      };
#endif

      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns( 
            {  {"size", convertToString( size ) }, } ) );

      /****
       * Write one using C for
       */
      if( tests.containsValue( "all" ) || tests.containsValue( "no-bc-pure-c"  ) )
      {
         benchmark.setOperation( "Pure C", pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );

         auto hostWriteOneUsingPureC = [&] ()
         {
            hostTraverserBenchmark.writeOneUsingPureC();
         };
         benchmark.time< Devices::Host >( hostReset, "CPU", hostWriteOneUsingPureC );

#ifdef HAVE_CUDA
         auto cudaWriteOneUsingPureC = [&] ()
         {
            cudaTraverserBenchmark.writeOneUsingPureC();
         };
         if( withCuda )
            benchmark.time< Devices::Cuda >( "GPU", cudaWriteOneUsingPureC );
#endif
      }

      /****
       * Write one using parallel for
       */
      if( tests.containsValue( "all" ) || tests.containsValue( "no-bc-parallel-for" ) )
      {
         benchmark.setOperation( "parallel for", pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );

         auto hostWriteOneUsingParallelFor = [&] ()
         {
            hostTraverserBenchmark.writeOneUsingParallelFor();
         };
         benchmark.time< Devices::Host >( hostReset, "CPU", hostWriteOneUsingParallelFor );

#ifdef HAVE_CUDA
         auto cudaWriteOneUsingParallelFor = [&] ()
         {
            cudaTraverserBenchmark.writeOneUsingParallelFor();
         };
         if( withCuda )
            benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaWriteOneUsingParallelFor );
#endif
      }

      /****
       * Write one using parallel for with grid entity
       */
      if( tests.containsValue( "all" ) || tests.containsValue( "no-bc-parallel-for-and-grid-entity" ) )
      {
         auto hostWriteOneUsingParallelForAndGridEntity = [&] ()
         {
            hostTraverserBenchmark.writeOneUsingParallelForAndGridEntity();
         };
         benchmark.setOperation( "par.for+grid ent.", pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         benchmark.time< Devices::Host >( hostReset, "CPU", hostWriteOneUsingParallelForAndGridEntity );

#ifdef HAVE_CUDA
         auto cudaWriteOneUsingParallelForAndGridEntity = [&] ()
         {
            cudaTraverserBenchmark.writeOneUsingParallelForAndGridEntity();
         };
         if( withCuda )
            benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaWriteOneUsingParallelForAndGridEntity );
#endif
      }

      /****
       * Write one using parallel for with mesh function
       */
      if( tests.containsValue( "all" ) || tests.containsValue( "no-bc-parallel-for-and-mesh-function" ) )
      {
         auto hostWriteOneUsingParallelForAndMeshFunction = [&] ()
         {
            hostTraverserBenchmark.writeOneUsingParallelForAndMeshFunction();
         };
         benchmark.setOperation( "par.for+mesh fc.", pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         benchmark.time< Devices::Host >( hostReset, "CPU", hostWriteOneUsingParallelForAndMeshFunction );

#ifdef HAVE_CUDA
         auto cudaWriteOneUsingParallelForAndMeshFunction = [&] ()
         {
            cudaTraverserBenchmark.writeOneUsingParallelForAndMeshFunction();
         };
         if( withCuda )
            benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaWriteOneUsingParallelForAndMeshFunction );
#endif

      }

      /****
       * Write one using traverser
       */
      if( tests.containsValue( "all" ) || tests.containsValue( "no-bc-traverser" ) )
      {
         benchmark.setOperation( "traverser", pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         auto hostWriteOneUsingTraverser = [&] ()
         {
            hostTraverserBenchmark.writeOneUsingTraverser();
         };
         benchmark.time< Devices::Host >( hostReset, "CPU", hostWriteOneUsingTraverser );

#ifdef HAVE_CUDA
         auto cudaWriteOneUsingTraverser = [&] ()
         {
            cudaTraverserBenchmark.writeOneUsingTraverser();
         };
         if( withCuda )
            benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaWriteOneUsingTraverser );
#endif
      }
      std::cout << "--------------------------------------------------------------------------------------------------------" << std::endl;
   }

   /****
    * Full grid traversing including boundary conditions
    */
   benchmark.newBenchmark( String("Traversing with boundary conditions" + convertToString( Dimension ) + "D" ), metadata );
   for( std::size_t size = minSize; size <= maxSize; size *= 2 )
   {
      GridTraversersBenchmark< Dimension, Devices::Host, Real, Index > hostTraverserBenchmark( size );
      GridTraversersBenchmark< Dimension, Devices::Cuda, Real, Index > cudaTraverserBenchmark( size );

      auto hostReset = [&]()
      {
         hostTraverserBenchmark.reset();
      };

#ifdef HAVE_CUDA
      auto cudaReset = [&]()
      {
         cudaTraverserBenchmark.reset();
      };
#endif

      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns(
            {  {"size", convertToString( size ) }, } ) );

      /****
       * Write one and two (as BC) using C for
       */
      auto hostTraverseUsingPureC = [&] ()
      {
         hostTraverserBenchmark.traverseUsingPureC();
      };

#ifdef HAVE_CUDA
      auto cudaTraverseUsingPureC = [&] ()
      {
         cudaTraverserBenchmark.traverseUsingPureC();
      };
#endif

      if( tests.containsValue( "all" ) || tests.containsValue( "bc-pure-c" ) )
      {
         benchmark.setOperation( "Pure C", pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         benchmark.time< Devices::Host >( "CPU", hostTraverseUsingPureC );
#ifdef HAVE_CUDA
         if( withCuda )
            benchmark.time< Devices::Cuda >( "GPU", cudaTraverseUsingPureC );
#endif

         benchmark.setOperation( "Pure C RST", pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         benchmark.time< Devices::Host >( hostReset, "CPU", hostTraverseUsingPureC );
#ifdef HAVE_CUDA
         if( withCuda )
            benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaTraverseUsingPureC );
#endif
      }

      /****
       * Write one and two (as BC) using parallel for
       */
      auto hostTraverseUsingParallelFor = [&] ()
      {
         hostTraverserBenchmark.writeOneUsingParallelFor();
      };

#ifdef HAVE_CUDA
      auto cudaTraverseUsingParallelFor = [&] ()
      {
         cudaTraverserBenchmark.writeOneUsingParallelFor();
      };
#endif

      if( tests.containsValue( "all" ) || tests.containsValue( "bc-parallel-for" ) )
      {
         benchmark.setOperation( "parallel for", pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         benchmark.time< Devices::Host >( "CPU", hostTraverseUsingParallelFor );
#ifdef HAVE_CUDA
         if( withCuda )
            benchmark.time< Devices::Cuda >( "GPU", cudaTraverseUsingParallelFor );
#endif

         benchmark.setOperation( "parallel for RST", pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         benchmark.time< Devices::Host >( hostReset, "CPU", hostTraverseUsingParallelFor );
#ifdef HAVE_CUDA
         if( withCuda )
            benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaTraverseUsingParallelFor );
#endif
      }

      /****
       * Write one and two (as BC) using traverser
       */
      auto hostTraverseUsingTraverser = [&] ()
      {
         hostTraverserBenchmark.writeOneUsingTraverser();
      };

#ifdef HAVE_CUDA
      auto cudaTraverseUsingTraverser = [&] ()
      {
         cudaTraverserBenchmark.writeOneUsingTraverser();
      };
#endif

      if( tests.containsValue( "all" ) || tests.containsValue( "bc-traverser" ) )
      {
         benchmark.setOperation( "traverser", pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         benchmark.time< Devices::Host >( hostReset, "CPU", hostTraverseUsingTraverser );
#ifdef HAVE_CUDA
         benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaTraverseUsingTraverser );
#endif

         benchmark.setOperation( "traverser RST", pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         benchmark.time< Devices::Host >( "CPU", hostTraverseUsingTraverser );
#ifdef HAVE_CUDA
         benchmark.time< Devices::Cuda >( "GPU", cudaTraverseUsingTraverser );
#endif
      }
   }
   return true;
}

void setupConfig( Config::ConfigDescription& config )
{
   config.addList< String >( "tests", "Tests to be performed.", "all" );
   config.addEntryEnum( "all" );
   config.addEntryEnum( "no-bc-pure-c" );
   config.addEntryEnum( "no-bc-parallel-for" );
   config.addEntryEnum( "no-bc-parallel-for-and-grid-entity" );
   config.addEntryEnum( "no-bc-traverser" );
   config.addEntryEnum( "bc-pure-c" );
   config.addEntryEnum( "bc-parallel-for" );
   config.addEntryEnum( "bc-traverser" );
#ifdef HAVE_CUDA
   config.addEntry< bool >( "with-cuda", "Perform even the CUDA benchmarks.", true );
#else
   config.addEntry< bool >( "with-cuda", "Perform even the CUDA benchmarks.", false );
#endif
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-traversers.log");
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< String >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
   config.addEntry< int >( "dimension", "Set the problem dimension. 0 means all dimensions 1,2 and 3.", 0 );
   config.addEntry< int >( "min-size", "Minimum size of arrays/vectors used in the benchmark.", 10 );
   config.addEntry< int >( "max-size", "Minimum size of arrays/vectors used in the benchmark.", 1000 );
   config.addEntry< int >( "size-step-factor", "Factor determining the size of arrays/vectors used in the benchmark. First size is min-size and each following size is stepFactor*previousSize, up to max-size.", 2 );

   Benchmark::configSetup( config );

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
}

template< int Dimension >
bool setupBenchmark( const Config::ParameterContainer& parameters )
{
   const String & logFileName = parameters.getParameter< String >( "log-file" );
   const String & outputMode = parameters.getParameter< String >( "output-mode" );
   const String & precision = parameters.getParameter< String >( "precision" );
   const unsigned sizeStepFactor = parameters.getParameter< unsigned >( "size-step-factor" );

   Benchmark benchmark; //( loops, verbose );
   benchmark.setup( parameters );
   Benchmark::MetadataMap metadata = getHardwareMetadata();
   runBenchmark< Dimension >( parameters, benchmark, metadata );

   auto mode = std::ios::out;
   if( outputMode == "append" )
       mode |= std::ios::app;
   std::ofstream logFile( logFileName.getString(), mode );

   if( ! benchmark.save( logFile ) )
   {
      std::cerr << "Failed to write the benchmark results to file '" << parameters.getParameter< String >( "log-file" ) << "'." << std::endl;
      return false;
   }
   return true;
}

int main( int argc, char* argv[] )
{
   Config::ConfigDescription config;
   Config::ParameterContainer parameters;
   
   setupConfig( config );
   if( ! parseCommandLine( argc, argv, config, parameters ) ) {
      config.printUsage( argv[ 0 ] );
      return EXIT_FAILURE;
   }

   if( ! Devices::Host::setup( parameters ) ||
       ! Devices::Cuda::setup( parameters ) )
      return EXIT_FAILURE;
   
   const int dimension = parameters.getParameter< int >( "dimension" );
   bool status( false );
   if( ! dimension )
   {
      status = setupBenchmark< 1 >( parameters );
      status |= setupBenchmark< 2 >( parameters );
      status |= setupBenchmark< 3 >( parameters );
   }
   else
   {
      switch( dimension )
      {
         case 1:
            status = setupBenchmark< 1 >( parameters );
            break;
         case 2:
            status = setupBenchmark< 2 >( parameters );
            break;
         case 3:
            status = setupBenchmark< 3 >( parameters );
            break;
      }
   }
   if( status == false )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}
