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

using namespace TNL;
using namespace TNL::Benchmarks;


template< int Dimension,
          typename Real = float,
          typename Index = int >
bool runBenchmark( const Config::ParameterContainer& parameters,
                   Benchmark& benchmark,
                   Benchmark::MetadataMap& metadata )
{
   // FIXME: getParameter< std::size_t >() does not work with parameters added with addEntry< int >(),
   // which have a default value. The workaround below works for int values, but it is not possible
   // to pass 64-bit integer values
   // const std::size_t minSize = parameters.getParameter< std::size_t >( "min-size" );
   // const std::size_t maxSize = parameters.getParameter< std::size_t >( "max-size" );
   const int minSize = parameters.getParameter< int >( "min-size" );
   const int maxSize = parameters.getParameter< int >( "max-size" );
   
   // Full grid traversing
   benchmark.newBenchmark( String("Full grid traversing " + convertToString( Dimension ) + "D" ), metadata );
   for( std::size_t size = minSize; size <= maxSize; size *= 2 )
   {

      GridTraversersBenchmark< Dimension, Devices::Host, Real, Index > hostTraverserBenchmark( size );
      GridTraversersBenchmark< Dimension, Devices::Cuda, Real, Index > cudaTraverserBenchmark( size );         

      auto reset = [&]() {};
      
      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns( 
            {  {"size", convertToString( size ) }, } ) );

      /****
       * Write one using parallel for
       */
      auto hostWriteOneUsingParallelFor = [&] ()
      {
         hostTraverserBenchmark.writeOneUsingParallelFor();
      }; 

      auto cudaWriteOneUsingParallelFor = [&] ()
      {
         cudaTraverserBenchmark.writeOneUsingParallelFor();
      }; 

      benchmark.setOperation( "write 1 using parallel for", pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
      benchmark.time( reset, "CPU", hostWriteOneUsingParallelFor );
#ifdef HAVE_CUDA
      benchmark.time( reset, "GPU", cudaWriteOneUsingParallelFor );
#endif

      /****
       * Write one using traverser
       */
      auto hostWriteOneUsingTraverser = [&] ()
      {
         hostTraverserBenchmark.writeOneUsingTraverser();
      }; 

      auto cudaWriteOneUsingTraverser = [&] ()
      {
         cudaTraverserBenchmark.writeOneUsingTraverser();
      }; 
      
      benchmark.setOperation( "write 1 using traverser", pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
      benchmark.time( reset, "CPU", hostWriteOneUsingTraverser );
#ifdef HAVE_CUDA
      benchmark.time( reset, "GPU", cudaWriteOneUsingTraverser );
#endif
      
      
   }   
   return true;
}

void setupConfig( Config::ConfigDescription& config )
{
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
