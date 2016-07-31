/***************************************************************************
                          tnl-benchmarks.h  -  description
                             -------------------
    begin                : Jan 27, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef tnlCudaBENCHMARKS_H_
#define TNLCUDBENCHMARKS_H_

#include <TNL/SystemInfo.h>
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
runCudaBenchmarks( Benchmark & benchmark,
                   Benchmark::MetadataMap metadata,
                   const unsigned & minSize,
                   const unsigned & maxSize,
                   const double & sizeStepFactor,
                   const unsigned & loops,
                   const unsigned & elementsPerRow )
{
    const String precision = getType< Real >();
    metadata["precision"] = precision;

    // Array operations
    benchmark.newBenchmark( String("Array operations (") + precision + ")",
                            metadata );
    for( unsigned size = minSize; size <= maxSize; size *= 2 ) {
        benchmark.setMetadataColumns( Benchmark::MetadataColumns({
           {"size", size},
        } ));
        benchmarkArrayOperations< Real >( benchmark, loops, size );
    }

    // Vector operations
    benchmark.newBenchmark( String("Vector operations (") + precision + ")",
                            metadata );
    for( unsigned size = minSize; size <= maxSize; size *= sizeStepFactor ) {
        benchmark.setMetadataColumns( Benchmark::MetadataColumns({
           {"size", size},
        } ));
        benchmarkVectorOperations< Real >( benchmark, loops, size );
    }

    // Sparse matrix-vector multiplication
    benchmark.newBenchmark( String("Sparse matrix-vector multiplication (") + precision + ")",
                            metadata );
    for( unsigned size = minSize; size <= maxSize; size *= 2 ) {
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
    config.addEntry< String >( "log-file", "Log file name.", "tnl-cuda-benchmarks.log");
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
}

int
main( int argc, char* argv[] )
{
#ifdef HAVE_CUDA
    Config::ParameterContainer parameters;
    Config::ConfigDescription conf_desc;

    setupConfig( conf_desc );

    if( ! parseCommandLine( argc, argv, conf_desc, parameters ) ) {
        conf_desc.printUsage( argv[ 0 ] );
        return 1;
    }

    const String & logFileName = parameters.getParameter< String >( "log-file" );
    const String & outputMode = parameters.getParameter< String >( "output-mode" );
    const String & precision = parameters.getParameter< String >( "precision" );
    const unsigned minSize = parameters.getParameter< unsigned >( "min-size" );
    const unsigned maxSize = parameters.getParameter< unsigned >( "max-size" );
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
    SystemInfo systemInfo;
    const int cpu_id = 0;
    tnlCacheSizes cacheSizes = systemInfo.getCPUCacheSizes( cpu_id );
    String cacheInfo = String( cacheSizes.L1data ) + ", "
                        + String( cacheSizes.L1instruction ) + ", "
                        + String( cacheSizes.L2 ) + ", "
                        + String( cacheSizes.L3 );
    const int activeGPU = Devices::CudaDeviceInfo::getActiveDevice();
    const String deviceArch = String( Devices::CudaDeviceInfo::getArchitectureMajor( activeGPU ) ) + "." +
                                 String( Devices::CudaDeviceInfo::getArchitectureMinor( activeGPU ) );
    Benchmark::MetadataMap metadata {
        { "host name", systemInfo.getHostname() },
        { "architecture", systemInfo.getArchitecture() },
        { "system", systemInfo.getSystemName() },
        { "system release", systemInfo.getSystemRelease() },
        { "start time", systemInfo.getCurrentTime() },
        { "CPU model name", systemInfo.getCPUModelName( cpu_id ) },
        { "CPU cores", systemInfo.getNumberOfCores( cpu_id ) },
        { "CPU threads per core", systemInfo.getNumberOfThreads( cpu_id ) / systemInfo.getNumberOfCores( cpu_id ) },
        { "CPU max frequency (MHz)", systemInfo.getCPUMaxFrequency( cpu_id ) / 1e3 },
        { "CPU cache sizes (L1d, L1i, L2, L3) (kiB)", cacheInfo },
        { "GPU name", Devices::CudaDeviceInfo::getDeviceName( activeGPU ) },
        { "GPU architecture", deviceArch },
        { "GPU CUDA cores", Devices::CudaDeviceInfo::getCudaCores( activeGPU ) },
        { "GPU clock rate (MHz)", (double) Devices::CudaDeviceInfo::getClockRate( activeGPU ) / 1e3 },
        { "GPU global memory (GB)", (double) Devices::CudaDeviceInfo::getGlobalMemory( activeGPU ) / 1e9 },
        { "GPU memory clock rate (MHz)", (double) Devices::CudaDeviceInfo::getMemoryClockRate( activeGPU ) / 1e3 },
        { "GPU memory ECC enabled", Devices::CudaDeviceInfo::getECCEnabled( activeGPU ) },
    };

    if( precision == "all" || precision == "float" )
        runCudaBenchmarks< float >( benchmark, metadata, minSize, maxSize, sizeStepFactor, loops, elementsPerRow );
    if( precision == "all" || precision == "double" )
        runCudaBenchmarks< double >( benchmark, metadata, minSize, maxSize, sizeStepFactor, loops, elementsPerRow );

    if( ! benchmark.save( logFile ) ) {
        std::cerr << "Failed to write the benchmark results to file '" << parameters.getParameter< String >( "log-file" ) << "'." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
#else
    CudaSupportMissingMessage;
    return EXIT_FAILURE;
#endif
}

#endif /* Devices::CudaBENCHMARKS_H_ */
