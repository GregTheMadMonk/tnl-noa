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

#include <TNL/Devices/Host.h>
#include <TNL/Devices/CudaDeviceInfo.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>

#include "array-operations.h"
#include "vector-operations.h"
#include "spmv.h"

using namespace TNL;
using namespace TNL::benchmarks;


// TODO: should benchmarks check the result of the computation?


template< typename Real >
void
runBlasBenchmarks( Benchmark & benchmark,
                   Benchmark::MetadataMap metadata,
                   const std::size_t & minSize,
                   const std::size_t & maxSize,
                   const double & sizeStepFactor,
                   const unsigned & loops,
                   const unsigned & elementsPerRow )
{
    const String precision = getType< Real >();
    metadata["precision"] = precision;

    // Array operations
    benchmark.newBenchmark( String("Array operations (") + precision + ")",
                            metadata );
    for( std::size_t size = minSize; size <= maxSize; size *= 2 ) {
        benchmark.setMetadataColumns( Benchmark::MetadataColumns({
           {"size", size},
        } ));
        benchmarkArrayOperations< Real >( benchmark, loops, size );
    }

    // Vector operations
    benchmark.newBenchmark( String("Vector operations (") + precision + ")",
                            metadata );
    for( std::size_t size = minSize; size <= maxSize; size *= sizeStepFactor ) {
        benchmark.setMetadataColumns( Benchmark::MetadataColumns({
           {"size", size},
        } ));
        benchmarkVectorOperations< Real >( benchmark, loops, size );
    }

    // Sparse matrix-vector multiplication
    benchmark.newBenchmark( String("Sparse matrix-vector multiplication (") + precision + ")",
                            metadata );
    for( std::size_t size = minSize; size <= maxSize; size *= 2 ) {
        benchmark.setMetadataColumns( Benchmark::MetadataColumns({
            {"rows", size},
            {"columns", size},
            {"elements per row", elementsPerRow},
        } ));
        benchmarkSpmvSynthetic< Real >( benchmark, loops, size, elementsPerRow );
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
    config.addEntry< int >( "elements-per-row", "Number of elements per row of the sparse matrix used in the matrix-vector multiplication benchmark.", 5 );
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

    if( ! parseCommandLine( argc, argv, conf_desc, parameters ) ) {
        conf_desc.printUsage( argv[ 0 ] );
        return 1;
    }

    Devices::Host::setup( parameters );
    Devices::Cuda::setup( parameters );

    const String & logFileName = parameters.getParameter< String >( "log-file" );
    const String & outputMode = parameters.getParameter< String >( "output-mode" );
    const String & precision = parameters.getParameter< String >( "precision" );
    const std::size_t minSize = parameters.getParameter< std::size_t >( "min-size" );
    const std::size_t maxSize = parameters.getParameter< std::size_t >( "max-size" );
    const unsigned sizeStepFactor = parameters.getParameter< unsigned >( "size-step-factor" );
    const unsigned loops = parameters.getParameter< unsigned >( "loops" );
    const unsigned elementsPerRow = parameters.getParameter< unsigned >( "elements-per-row" );
    const unsigned verbose = parameters.getParameter< unsigned >( "verbose" );

    if( sizeStepFactor <= 1 ) {
        std::cerr << "The value of --size-step-factor must be greater than 1." << std::endl;
        return EXIT_FAILURE;
    }

    // open log file
    auto mode = std::ios::out;
    if( outputMode == "append" )
        mode |= std::ios::app;
    std::ofstream logFile( logFileName.getString(), mode );

    // init benchmark and common metadata
    Benchmark benchmark( loops, verbose );

    // prepare global metadata
    const int cpu_id = 0;
    Devices::CacheSizes cacheSizes = Devices::Host::getCPUCacheSizes( cpu_id );
    String cacheInfo = String( cacheSizes.L1data ) + ", "
                        + String( cacheSizes.L1instruction ) + ", "
                        + String( cacheSizes.L2 ) + ", "
                        + String( cacheSizes.L3 );
#ifdef HAVE_CUDA
    const int activeGPU = Devices::CudaDeviceInfo::getActiveDevice();
    const String deviceArch = String( Devices::CudaDeviceInfo::getArchitectureMajor( activeGPU ) ) + "." +
                              String( Devices::CudaDeviceInfo::getArchitectureMinor( activeGPU ) );
#endif
    Benchmark::MetadataMap metadata {
        { "host name", Devices::Host::getHostname() },
        { "architecture", Devices::Host::getArchitecture() },
        { "system", Devices::Host::getSystemName() },
        { "system release", Devices::Host::getSystemRelease() },
        { "start time", Devices::Host::getCurrentTime() },
        { "CPU model name", Devices::Host::getCPUModelName( cpu_id ) },
        { "CPU cores", Devices::Host::getNumberOfCores( cpu_id ) },
        { "CPU threads per core", Devices::Host::getNumberOfThreads( cpu_id ) / Devices::Host::getNumberOfCores( cpu_id ) },
        { "CPU max frequency (MHz)", Devices::Host::getCPUMaxFrequency( cpu_id ) / 1e3 },
        { "CPU cache sizes (L1d, L1i, L2, L3) (kiB)", cacheInfo },
#ifdef HAVE_CUDA
        { "GPU name", Devices::CudaDeviceInfo::getDeviceName( activeGPU ) },
        { "GPU architecture", deviceArch },
        { "GPU CUDA cores", Devices::CudaDeviceInfo::getCudaCores( activeGPU ) },
        { "GPU clock rate (MHz)", (double) Devices::CudaDeviceInfo::getClockRate( activeGPU ) / 1e3 },
        { "GPU global memory (GB)", (double) Devices::CudaDeviceInfo::getGlobalMemory( activeGPU ) / 1e9 },
        { "GPU memory clock rate (MHz)", (double) Devices::CudaDeviceInfo::getMemoryClockRate( activeGPU ) / 1e3 },
        { "GPU memory ECC enabled", Devices::CudaDeviceInfo::getECCEnabled( activeGPU ) },
#endif
    };

    if( precision == "all" || precision == "float" )
        runBlasBenchmarks< float >( benchmark, metadata, minSize, maxSize, sizeStepFactor, loops, elementsPerRow );
    if( precision == "all" || precision == "double" )
        runBlasBenchmarks< double >( benchmark, metadata, minSize, maxSize, sizeStepFactor, loops, elementsPerRow );

    if( ! benchmark.save( logFile ) ) {
        std::cerr << "Failed to write the benchmark results to file '" << parameters.getParameter< String >( "log-file" ) << "'." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
