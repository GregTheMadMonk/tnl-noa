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
   // FIXME: the --tests is just a string because list does not work with enums
//   const Containers::List< String >& tests = parameters.getParameter< Containers::List< String > >( "tests" );
   Containers::List< String > tests;
   tests.Append( parameters.getParameter< String >( "tests" ) );
   // FIXME: getParameter< std::size_t >() does not work with parameters added with addEntry< int >(),
   // which have a default value. The workaround below works for int values, but it is not possible
   // to pass 64-bit integer values
   // const std::size_t minSize = parameters.getParameter< std::size_t >( "min-size" );
   // const std::size_t maxSize = parameters.getParameter< std::size_t >( "max-size" );
   const std::size_t minSize = parameters.getParameter< int >( "min-size" );
   const std::size_t maxSize = parameters.getParameter< int >( "max-size" );
   const bool withHost = parameters.getParameter< bool >( "with-host" );
#ifdef HAVE_CUDA
   const bool withCuda = parameters.getParameter< bool >( "with-cuda" );
#else
   const bool withCuda = false;
#endif
   const bool check = parameters.getParameter< bool >( "check" );

   /****
    * Full grid traversing with no boundary conditions
    */
   benchmark.newBenchmark( String("Traversing without boundary conditions" + convertToString( Dimension ) + "D" ), metadata );
   for( std::size_t size = minSize; size <= maxSize; size *= 2 )
   {
      GridTraversersBenchmark< Dimension, Devices::Host, Real, Index > hostTraverserBenchmark( size );
      GridTraversersBenchmark< Dimension, Devices::Cuda, Real, Index > cudaTraverserBenchmark( size );

      auto hostReset = [&]()
      {
         hostTraverserBenchmark.reset();
      };

      auto cudaReset = [&]()
      {
         cudaTraverserBenchmark.reset();
      };

      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns( 
            {  {"size", convertToString( size ) }, } ) );

      /****
       * Add one using pure C code
       */
      if( tests.containsValue( "all" ) || tests.containsValue( "add-one-pure-c"  ) )
      {
         benchmark.setOperation( "Pure C", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );

         auto hostWriteOneUsingPureC = [&] ()
         {
            hostTraverserBenchmark.addOneUsingPureC();
         };
         if( withHost )
         {
            benchmark.time< Devices::Host >( hostReset, "CPU", hostWriteOneUsingPureC );
            if( check && ! hostTraverserBenchmark.checkAddOne(
                  benchmark.getPerformedLoops(),
                  benchmark.isResetingOn() ) )
               benchmark.addErrorMessage( "Test results are not correct." );
         }

         auto cudaWriteOneUsingPureC = [&] ()
         {
            cudaTraverserBenchmark.addOneUsingPureC();
         };
         if( withCuda )
         {
            benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaWriteOneUsingPureC );
            if( check && ! cudaTraverserBenchmark.checkAddOne(
                  benchmark.getPerformedLoops(),
                  benchmark.isResetingOn() ) )
               benchmark.addErrorMessage( "Test results are not correct." );
         }
      }

      /****
       * Add one using parallel for
       */
      if( tests.containsValue( "all" ) || tests.containsValue( "add-one-parallel-for" ) )
      {
         benchmark.setOperation( "parallel for", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );

         auto hostWriteOneUsingParallelFor = [&] ()
         {
            hostTraverserBenchmark.addOneUsingParallelFor();
         };
         if( withHost )
         {
            benchmark.time< Devices::Host >( hostReset, "CPU", hostWriteOneUsingParallelFor );
            if( check && ! hostTraverserBenchmark.checkAddOne( 
                  benchmark.getPerformedLoops(),
                  benchmark.isResetingOn() ) )
               benchmark.addErrorMessage( "Test results are not correct." );
         }

         auto cudaWriteOneUsingParallelFor = [&] ()
         {
            cudaTraverserBenchmark.addOneUsingParallelFor();
         };
         if( withCuda )
         {
            benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaWriteOneUsingParallelFor );
            if( check && ! cudaTraverserBenchmark.checkAddOne( 
                  benchmark.getPerformedLoops(),
                  benchmark.isResetingOn() ) )
               benchmark.addErrorMessage( "Test results are not correct." );
         }
      }

      /****
       * Add one using parallel for with grid entity
       */
      if( tests.containsValue( "all" ) || tests.containsValue( "add-one-simple-cell" ) )
      {
         auto hostAddOneUsingSimpleCell = [&] ()
         {
            hostTraverserBenchmark.addOneUsingSimpleCell();
         };
         benchmark.setOperation( "simple cell", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         if( withHost )
         {
            benchmark.time< Devices::Host >( hostReset, "CPU", hostAddOneUsingSimpleCell );
            if( check && ! hostTraverserBenchmark.checkAddOne( 
                  benchmark.getPerformedLoops(),
                  benchmark.isResetingOn() ) )
               benchmark.addErrorMessage( "Test results are not correct." );
         }

         auto cudaAddOneUsingSimpleCell = [&] ()
         {
            cudaTraverserBenchmark.addOneUsingSimpleCell();
         };
         if( withCuda )
         {
            benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaAddOneUsingSimpleCell );
            if( check && ! cudaTraverserBenchmark.checkAddOne( 
                  benchmark.getPerformedLoops(),
                  benchmark.isResetingOn() ) )
               benchmark.addErrorMessage( "Test results are not correct." );
         }
      }

      /****
       * Add one using parallel for with mesh function
       */
      if( tests.containsValue( "all" ) || tests.containsValue( "add-one-parallel-for-and-mesh-function" ) )
      {
         auto hostAddOneUsingParallelForAndMeshFunction = [&] ()
         {
            hostTraverserBenchmark.addOneUsingParallelForAndMeshFunction();
         };
         benchmark.setOperation( "par.for+mesh fc.", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         if( withHost )
         {
            benchmark.time< Devices::Host >( hostReset, "CPU", hostAddOneUsingParallelForAndMeshFunction );
            if( check && ! hostTraverserBenchmark.checkAddOne( 
                  benchmark.getPerformedLoops(),
                  benchmark.isResetingOn() ) )
               benchmark.addErrorMessage( "Test results are not correct." );
         }

         auto cudaAddOneUsingParallelForAndMeshFunction = [&] ()
         {
            cudaTraverserBenchmark.addOneUsingParallelForAndMeshFunction();
         };
         if( withCuda )
         {
            benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaAddOneUsingParallelForAndMeshFunction );
            if( check && ! cudaTraverserBenchmark.checkAddOne( 
                  benchmark.getPerformedLoops(),
                  benchmark.isResetingOn() ) )
               benchmark.addErrorMessage( "Test results are not correct." );
         }
      }

      /****
       * Add one using traverser
       */
      if( tests.containsValue( "all" ) || tests.containsValue( "add-one-traverser" ) )
      {
         benchmark.setOperation( "traverser", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         auto hostWriteOneUsingTraverser = [&] ()
         {
            hostTraverserBenchmark.addOneUsingTraverser();
         };
         if( withHost )
         {
            benchmark.time< Devices::Host >( hostReset, "CPU", hostWriteOneUsingTraverser );
            if( check && ! hostTraverserBenchmark.checkAddOne( 
                  benchmark.getPerformedLoops(),
                  benchmark.isResetingOn() ) )
               benchmark.addErrorMessage( "Test results are not correct." );
         }

         auto cudaWriteOneUsingTraverser = [&] ()
         {
            cudaTraverserBenchmark.addOneUsingTraverser();
         };
         if( withCuda )
         {
            benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaWriteOneUsingTraverser );
            if( check && ! cudaTraverserBenchmark.checkAddOne( 
                  benchmark.getPerformedLoops(),
                  benchmark.isResetingOn() ) )
               benchmark.addErrorMessage( "Test results are not correct." );
         }
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

      auto cudaReset = [&]()
      {
         cudaTraverserBenchmark.reset();
      };

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

      auto cudaTraverseUsingPureC = [&] ()
      {
         cudaTraverserBenchmark.traverseUsingPureC();
      };

      if( tests.containsValue( "all" ) || tests.containsValue( "bc-pure-c" ) )
      {
         benchmark.setOperation( "Pure C", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         if( withHost )
            benchmark.time< Devices::Host >( "CPU", hostTraverseUsingPureC );
         if( withCuda )
            benchmark.time< Devices::Cuda >( "GPU", cudaTraverseUsingPureC );

         benchmark.setOperation( "Pure C RST", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         if( withHost )
            benchmark.time< Devices::Host >( hostReset, "CPU", hostTraverseUsingPureC );
         if( withCuda )
            benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaTraverseUsingPureC );
      }

      /****
       * Write one and two (as BC) using parallel for
       */
      auto hostTraverseUsingParallelFor = [&] ()
      {
         hostTraverserBenchmark.addOneUsingParallelFor();
      };

      auto cudaTraverseUsingParallelFor = [&] ()
      {
         cudaTraverserBenchmark.addOneUsingParallelFor();
      };

      if( tests.containsValue( "all" ) || tests.containsValue( "bc-parallel-for" ) )
      {
         benchmark.setOperation( "parallel for", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         if( withHost )
            benchmark.time< Devices::Host >( "CPU", hostTraverseUsingParallelFor );
         if( withCuda )
            benchmark.time< Devices::Cuda >( "GPU", cudaTraverseUsingParallelFor );

         benchmark.setOperation( "parallel for RST", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         if( withHost )
            benchmark.time< Devices::Host >( hostReset, "CPU", hostTraverseUsingParallelFor );
         if( withCuda )
            benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaTraverseUsingParallelFor );
      }

      /****
       * Write one and two (as BC) using traverser
       */
      auto hostTraverseUsingTraverser = [&] ()
      {
         hostTraverserBenchmark.addOneUsingTraverser();
      };

      auto cudaTraverseUsingTraverser = [&] ()
      {
         cudaTraverserBenchmark.addOneUsingTraverser();
      };

      if( tests.containsValue( "all" ) || tests.containsValue( "bc-traverser" ) )
      {
         benchmark.setOperation( "traverser", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         if( withHost )
            benchmark.time< Devices::Host >( "CPU", hostTraverseUsingTraverser );
         if( withCuda )
            benchmark.time< Devices::Cuda >( "GPU", cudaTraverseUsingTraverser );

         benchmark.setOperation( "traverser RST", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         if( withHost )
            benchmark.time< Devices::Host >( hostReset, "CPU", hostTraverseUsingTraverser );
         if( withCuda )
            benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaTraverseUsingTraverser );
      }
   }
   return true;
}

void setupConfig( Config::ConfigDescription& config )
{
   // FIXME: addList does not work with addEntryEnum - ConfigDescription::addEntryEnum throws std::bad_cast
//   config.addList< String >( "tests", "Tests to be performed.", "all" );
   config.addEntry< String >( "tests", "Tests to be performed.", "all" );
   config.addEntryEnum( "all" );
   config.addEntryEnum( "add-one-pure-c" );
   config.addEntryEnum( "add-one-parallel-for" );
   config.addEntryEnum( "add-one-parallel-for-and-grid-entity" );
   config.addEntryEnum( "add-one-traverser" );
   config.addEntryEnum( "bc-pure-c" );
   config.addEntryEnum( "bc-parallel-for" );
   config.addEntryEnum( "bc-traverser" );
   config.addEntry< bool >( "with-host", "Perform CPU benchmarks.", true );
#ifdef HAVE_CUDA
   config.addEntry< bool >( "with-cuda", "Perform CUDA benchmarks.", true );
#else
   config.addEntry< bool >( "with-cuda", "Perform CUDA benchmarks.", false );
#endif
   config.addEntry< bool >( "check", "Checking correct results of benchmark tests.", false );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-traversers.log");
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
//   config.addEntry< String >( "precision", "Precision of the arithmetics.", "double" );
//   config.addEntryEnum( "float" );
//   config.addEntryEnum( "double" );
//   config.addEntryEnum( "all" );
   config.addEntry< int >( "dimension", "Set the problem dimension. 0 means all dimensions 1,2 and 3.", 0 );
   config.addEntry< int >( "min-size", "Minimum size of arrays/vectors used in the benchmark.", 10 );
   config.addEntry< int >( "max-size", "Minimum size of arrays/vectors used in the benchmark.", 1000 );
//   config.addEntry< int >( "size-step-factor", "Factor determining the size of arrays/vectors used in the benchmark. First size is min-size and each following size is stepFactor*previousSize, up to max-size.", 2 );

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
//   const String & precision = parameters.getParameter< String >( "precision" );
//   const int sizeStepFactor = parameters.getParameter< int >( "size-step-factor" );

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
