/***************************************************************************
                          tnl-benchmark-blas.h  -  description
                             -------------------
    begin                : Jan 27, 2010
    copyright            : (C) 2010 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Allocators/CudaHost.h>
#include <TNL/Allocators/CudaManaged.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Config/parseCommandLine.h>

#include "array-operations.h"
#include "vector-operations.h"
#include "triad.h"
#include "gemv.h"


using namespace TNL;
using namespace TNL::Benchmarks;


template< typename Real >
void
runBlasBenchmarks( Benchmark<> & benchmark,
                   Benchmark<>::MetadataMap metadata,
                   const std::size_t & minSize,
                   const std::size_t & maxSize,
                   const double & sizeStepFactor )
{
   const String precision = getType< Real >();
   metadata["precision"] = precision;

   // Array operations
   benchmark.newBenchmark( String("Array operations (") + precision + ", host allocator = Host)",
                           metadata );
   for( std::size_t size = minSize; size <= maxSize; size *= 2 ) {
      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns({
         { "size", convertToString( size ) },
      } ));
      benchmarkArrayOperations< Real >( benchmark, size );
   }
#ifdef HAVE_CUDA
   benchmark.newBenchmark( String("Array operations (") + precision + ", host allocator = CudaHost)",
                           metadata );
   for( std::size_t size = minSize; size <= maxSize; size *= 2 ) {
      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns({
         { "size", convertToString( size ) },
      } ));
      benchmarkArrayOperations< Real, int, Allocators::CudaHost >( benchmark, size );
   }
   benchmark.newBenchmark( String("Array operations (") + precision + ", host allocator = CudaManaged)",
                           metadata );
   for( std::size_t size = minSize; size <= maxSize; size *= 2 ) {
      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns({
         { "size", convertToString( size ) },
      } ));
      benchmarkArrayOperations< Real, int, Allocators::CudaManaged >( benchmark, size );
   }
#endif

   // Vector operations
   benchmark.newBenchmark( String("Vector operations (") + precision + ")",
                           metadata );
   for( std::size_t size = minSize; size <= maxSize; size *= sizeStepFactor ) {
      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns({
         { "size", convertToString( size ) },
      } ));
      benchmarkVectorOperations< Real >( benchmark, size );
   }

   // Triad benchmark: copy from host, compute, copy to host
#ifdef HAVE_CUDA
   benchmark.newBenchmark( String("Triad benchmark (") + precision + ")",
                           metadata );
   for( std::size_t size = minSize; size <= maxSize; size *= 2 ) {
      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns({
         { "size", convertToString( size ) },
      } ));
      benchmarkTriad< Real >( benchmark, size );
   }
#endif

   // Dense matrix-vector multiplication
   benchmark.newBenchmark( String("Dense matrix-vector multiplication (") + precision + ")",
                           metadata );
   for( std::size_t rows = 10; rows <= 20000 * 20000; rows *= 2 ) {
      for( std::size_t columns = 10; columns <= 20000 * 20000; columns *= 2 ) {
         if( rows * columns > 20000 * 20000 )
            break;
         benchmark.setMetadataColumns( Benchmark<>::MetadataColumns({
            { "rows", convertToString( rows ) },
            { "columns", convertToString( columns ) }
         } ));
         benchmarkGemv< Real >( benchmark, rows, columns );
      }
   }
}

void
setupConfig( Config::ConfigDescription & config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-blas.log");
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< String >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
   config.addEntry< int >( "min-size", "Minimum size of arrays/vectors used in the benchmark.", 100000 );
   config.addEntry< int >( "max-size", "Minimum size of arrays/vectors used in the benchmark.", 10000000 );
   config.addEntry< int >( "size-step-factor", "Factor determining the size of arrays/vectors used in the benchmark. First size is min-size and each following size is stepFactor*previousSize, up to max-size.", 2 );
   config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
}

int
main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   setupConfig( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   if( ! Devices::Host::setup( parameters ) ||
       ! Devices::Cuda::setup( parameters ) )
      return EXIT_FAILURE;

   const String & logFileName = parameters.getParameter< String >( "log-file" );
   const String & outputMode = parameters.getParameter< String >( "output-mode" );
   const String & precision = parameters.getParameter< String >( "precision" );
   // FIXME: getParameter< std::size_t >() does not work with parameters added with addEntry< int >(),
   // which have a default value. The workaround below works for int values, but it is not possible
   // to pass 64-bit integer values
//   const std::size_t minSize = parameters.getParameter< std::size_t >( "min-size" );
//   const std::size_t maxSize = parameters.getParameter< std::size_t >( "max-size" );
   const std::size_t minSize = parameters.getParameter< int >( "min-size" );
   const std::size_t maxSize = parameters.getParameter< int >( "max-size" );
   const int sizeStepFactor = parameters.getParameter< int >( "size-step-factor" );
   const int loops = parameters.getParameter< int >( "loops" );
   const int verbose = parameters.getParameter< int >( "verbose" );

   if( sizeStepFactor <= 1 ) {
       std::cerr << "The value of --size-step-factor must be greater than 1." << std::endl;
       return EXIT_FAILURE;
   }

   // open log file
   auto mode = std::ios::out;
   if( outputMode == "append" )
       mode |= std::ios::app;
   std::ofstream logFile( logFileName, mode );

   // init benchmark and common metadata
   Benchmark<> benchmark( logFile, loops, verbose );

   // prepare global metadata
   Logging::MetadataMap metadata = getHardwareMetadata();

   if( precision == "all" || precision == "float" )
      runBlasBenchmarks< float >( benchmark, metadata, minSize, maxSize, sizeStepFactor );
   if( precision == "all" || precision == "double" )
      runBlasBenchmarks< double >( benchmark, metadata, minSize, maxSize, sizeStepFactor );

   return EXIT_SUCCESS;
}
