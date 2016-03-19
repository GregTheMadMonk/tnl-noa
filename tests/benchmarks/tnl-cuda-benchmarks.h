/***************************************************************************
                          tnl-benchmarks.h  -  description
                             -------------------
    begin                : Jan 27, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLCUDABENCHMARKS_H_
#define TNLCUDBENCHMARKS_H_

#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>

#include "array-operations.h"
#include "vector-operations.h"
#include "spmv.h"

using namespace tnl::benchmarks;


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
    const tnlString precision = getType< Real >();
    metadata["precision"] = precision;

    // Array operations
    benchmark.newBenchmark( tnlString("Array operations (") + precision + ")",
                            metadata );
    for( unsigned size = minSize; size <= maxSize; size *= 2 ) {
        benchmark.setMetadataColumns( Benchmark::MetadataColumns({
           {"size", size},
        } ));
        benchmarkArrayOperations< Real >( benchmark, loops, size );
    }

    // Vector operations
    benchmark.newBenchmark( tnlString("Vector operations (") + precision + ")",
                            metadata );
    for( unsigned size = minSize; size <= maxSize; size *= sizeStepFactor ) {
        benchmark.setMetadataColumns( Benchmark::MetadataColumns({
           {"size", size},
        } ));
        benchmarkVectorOperations< Real >( benchmark, loops, size );
    }

    // Sparse matrix-vector multiplication
    benchmark.newBenchmark( tnlString("Sparse matrix-vector multiplication (") + precision + ")",
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
setupConfig( tnlConfigDescription & config )
{
    config.addDelimiter( "Benchmark settings:" );
    config.addEntry< tnlString >( "log-file", "Log file name.", "tnl-cuda-benchmarks.log");
    config.addEntry< tnlString >( "output-mode", "Mode for opening the log file.", "overwrite" );
    config.addEntryEnum( "append" );
    config.addEntryEnum( "overwrite" );
    config.addEntry< tnlString >( "precision", "Precision of the arithmetics.", "double" );
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
    tnlParameterContainer parameters;
    tnlConfigDescription conf_desc;

    setupConfig( conf_desc );

    if( ! parseCommandLine( argc, argv, conf_desc, parameters ) ) {
        conf_desc.printUsage( argv[ 0 ] );
        return 1;
    }

    const tnlString & logFileName = parameters.getParameter< tnlString >( "log-file" );
    const tnlString & outputMode = parameters.getParameter< tnlString >( "output-mode" );
    const tnlString & precision = parameters.getParameter< tnlString >( "precision" );
    const unsigned minSize = parameters.getParameter< unsigned >( "min-size" );
    const unsigned maxSize = parameters.getParameter< unsigned >( "max-size" );
    const unsigned sizeStepFactor = parameters.getParameter< unsigned >( "size-step-factor" );
    const unsigned loops = parameters.getParameter< unsigned >( "loops" );
    const unsigned elementsPerRow = parameters.getParameter< unsigned >( "elements-per-row" );
    const unsigned verbose = parameters.getParameter< unsigned >( "verbose" );

    if( sizeStepFactor <= 1 ) {
        cerr << "The value of --size-step-factor must be greater than 1." << endl;
        return EXIT_FAILURE;
    }

    // open log file
    auto mode = ios::out;
    if( outputMode == "append" )
        mode |= ios::app;
    ofstream logFile( logFileName.getString(), mode );

    // init benchmark and common metadata
    Benchmark benchmark( loops, verbose );
    // TODO: add hostname, CPU info, GPU info, date, ...
    Benchmark::MetadataMap metadata {
//        {"key", value},
    };

    if( precision == "all" || precision == "float" )
        runCudaBenchmarks< float >( benchmark, metadata, minSize, maxSize, sizeStepFactor, loops, elementsPerRow );
    if( precision == "all" || precision == "double" )
        runCudaBenchmarks< double >( benchmark, metadata, minSize, maxSize, sizeStepFactor, loops, elementsPerRow );

    if( ! benchmark.save( logFile ) ) {
        cerr << "Failed to write the benchmark results to file '" << parameters.getParameter< tnlString >( "log-file" ) << "'." << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
#else
    tnlCudaSupportMissingMessage;
    return EXIT_FAILURE;
#endif
}

#endif /* TNLCUDABENCHMARKS_H_ */
