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

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/ParallelFor.h>

using namespace TNL;
using namespace TNL::Benchmarks;

void setupConfig( Config::ConfigDescription& config )
{
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-blas.log");
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< String >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
   config.addEntry< int >( "dimension", "Set the problem dimension. 0 means all dimensions 1,2 and 3.", 0 );   
   config.addEntry< std::size_t >( "min-size", "Minimum size of arrays/vectors used in the benchmark.", 10 );
   config.addEntry< std::size_t >( "max-size", "Minimum size of arrays/vectors used in the benchmark.", 1000 );
   config.addEntry< int >( "size-step-factor", "Factor determining the size of arrays/vectors used in the benchmark. First size is min-size and each following size is stepFactor*previousSize, up to max-size.", 2 );
   config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );   
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );   
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
   
   const String & logFileName = parameters.getParameter< String >( "log-file" );
   const String & outputMode = parameters.getParameter< String >( "output-mode" );
   const String & precision = parameters.getParameter< String >( "precision" );
   // FIXME: getParameter< std::size_t >() does not work with parameters added with addEntry< int >(),
   // which have a default value. The workaround below works for int values, but it is not possible
   // to pass 64-bit integer values
   // const std::size_t minSize = parameters.getParameter< std::size_t >( "min-size" );
   // const std::size_t maxSize = parameters.getParameter< std::size_t >( "max-size" );
   const int dimension = parameters.getParameter< int >( "dimension" );
   const std::size_t minSize = parameters.getParameter< std::size_t >( "min-size" );
   const std::size_t maxSize = parameters.getParameter< std::size_t >( "max-size" );
   const unsigned sizeStepFactor = parameters.getParameter< unsigned >( "size-step-factor" );
   const unsigned loops = parameters.getParameter< unsigned >( "loops" );
   const unsigned verbose = parameters.getParameter< unsigned >( "verbose" );
   
   bool status( false );
   if( ! dimension )
   {
      status = performBenchmark< 1 >( parameters );
      status |= performBenchmark< 2 >( parameters );
      status |= performBenchmark< 3 >( parameters );
   }
   else
   {
      switch( dimension )
      {
         case 1:
            status = performBenchmark< 1 >( parameters );
            break;
         case 2:
            status = performBenchmark< 2 >( parameters );
            break;
         case 3:
            status = performBenchmark< 3 >( parameters );
            break;
      }
   }
   if( status == false )
      return EXIT_FAILURE;
   return EXIT_SUCCES;
}