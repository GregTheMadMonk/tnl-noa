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
#include <core/tnlList.h>
#include <matrices/tnlCSRMatrix.h>
#include <matrices/tnlEllpackMatrix.h>
#include <matrices/tnlSlicedEllpackMatrix.h>
#include <matrices/tnlChunkedEllpackMatrix.h>

#include "array-operations.h"
#include "vector-operations.h"

using namespace tnl::benchmarks;


// TODO: should benchmarks check the result of the computation?


// silly alias to match the number of template parameters with other formats
template< typename Real, typename Device, typename Index >
using SlicedEllpackMatrix = tnlSlicedEllpackMatrix< Real, Device, Index >;

template< typename Matrix >
int setHostTestMatrix( Matrix& matrix,
                       const int elementsPerRow )
{
   const int size = matrix.getRows();
   int elements( 0 );
   for( int row = 0; row < size; row++ )
   {
      int col = row - elementsPerRow / 2;
      for( int element = 0; element < elementsPerRow; element++ )
      {
         if( col + element >= 0 &&
             col + element < size )
         {
            matrix.setElement( row, col + element, element + 1 );
            elements++;
         }
      }
   }
   return elements;
}

template< typename Matrix >
__global__ void setCudaTestMatrixKernel( Matrix* matrix,
                                         const int elementsPerRow,
                                         const int gridIdx )
{
   const int rowIdx = ( gridIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( rowIdx >= matrix->getRows() )
      return;
   int col = rowIdx - elementsPerRow / 2;
   for( int element = 0; element < elementsPerRow; element++ )
   {
      if( col + element >= 0 &&
          col + element < matrix->getColumns() )
         matrix->setElementFast( rowIdx, col + element, element + 1 );
   }
}

template< typename Matrix >
void setCudaTestMatrix( Matrix& matrix,
                        const int elementsPerRow )
{
   typedef typename Matrix::IndexType IndexType;
   typedef typename Matrix::RealType RealType;
   Matrix* kernel_matrix = tnlCuda::passToDevice( matrix );
   dim3 cudaBlockSize( 256 ), cudaGridSize( tnlCuda::getMaxGridSize() );
   const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
   const IndexType cudaGrids = roundUpDivision( cudaBlocks, tnlCuda::getMaxGridSize() );
   for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
   {
      if( gridIdx == cudaGrids - 1 )
         cudaGridSize.x = cudaBlocks % tnlCuda::getMaxGridSize();
      setCudaTestMatrixKernel< Matrix >
         <<< cudaGridSize, cudaBlockSize >>>
         ( kernel_matrix, elementsPerRow, gridIdx );
      checkCudaDevice;
   }
   tnlCuda::freeFromDevice( kernel_matrix );
}


template< typename Real,
          template< typename, typename, typename > class Matrix,
          template< typename, typename, typename > class Vector = tnlVector >
bool
benchmarkSpMV( Benchmark & benchmark,
               const int & loops,
               const int & size,
               const int elementsPerRow = 5 )
{
   typedef Matrix< Real, tnlHost, int > HostMatrix;
   typedef Matrix< Real, tnlCuda, int > DeviceMatrix;
   typedef tnlVector< Real, tnlHost, int > HostVector;
   typedef tnlVector< Real, tnlCuda, int > CudaVector;

   HostMatrix hostMatrix;
   DeviceMatrix deviceMatrix;
   tnlVector< int, tnlHost, int > hostRowLengths;
   tnlVector< int, tnlCuda, int > deviceRowLengths;
   HostVector hostVector, hostVector2;
   CudaVector deviceVector, deviceVector2;

   // create benchmark group
   tnlList< tnlString > parsedType;
   parseObjectType( HostMatrix::getType(), parsedType );
   benchmark.createHorizontalGroup( parsedType[ 0 ], 2 );

   if( ! hostRowLengths.setSize( size ) ||
       ! deviceRowLengths.setSize( size ) ||
       ! hostMatrix.setDimensions( size, size ) ||
       ! deviceMatrix.setDimensions( size, size ) ||
       ! hostVector.setSize( size ) ||
       ! hostVector2.setSize( size ) ||
       ! deviceVector.setSize( size ) ||
       ! deviceVector2.setSize( size ) )
   {
      const char* msg = "error: allocation of vectors failed";
      cerr << msg << endl;
      benchmark.addErrorMessage( msg, 2 );
      return false;
   }

   hostRowLengths.setValue( elementsPerRow );
   deviceRowLengths.setValue( elementsPerRow );

   if( ! hostMatrix.setCompressedRowsLengths( hostRowLengths ) )
   {
      const char* msg = "error: allocation of host matrix failed";
      cerr << msg << endl;
      benchmark.addErrorMessage( msg, 2 );
      return false;
   }
   if( ! deviceMatrix.setCompressedRowsLengths( deviceRowLengths ) )
   {
      const char* msg = "error: allocation of device matrix failed";
      cerr << msg << endl;
      benchmark.addErrorMessage( msg, 2 );
      return false;
   }

   const int elements = setHostTestMatrix< HostMatrix >( hostMatrix, elementsPerRow );
   setCudaTestMatrix< DeviceMatrix >( deviceMatrix, elementsPerRow );
   const double datasetSize = loops * elements * ( 2 * sizeof( Real ) + sizeof( int ) ) / oneGB;

   // reset function
   auto reset = [&]() {
      hostVector.setValue( 1.0 );
      deviceVector.setValue( 1.0 );
      hostVector2.setValue( 0.0 );
      deviceVector2.setValue( 0.0 );
   };

   // compute functions
   auto spmvHost = [&]() {
      hostMatrix.vectorProduct( hostVector, hostVector2 );
   };
   auto spmvCuda = [&]() {
      deviceMatrix.vectorProduct( deviceVector, deviceVector2 );
   };

   benchmark.setOperation( datasetSize );
   benchmark.time( reset,
                   "CPU", spmvHost,
                   "GPU", spmvCuda );

   return true;
}

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

        // TODO: benchmark all formats from tnl-benchmark-spmv (different parameters of the base formats)
        benchmarkSpMV< Real, tnlCSRMatrix >( benchmark, loops, size, elementsPerRow );
        benchmarkSpMV< Real, tnlEllpackMatrix >( benchmark, loops, size, elementsPerRow );
        benchmarkSpMV< Real, SlicedEllpackMatrix >( benchmark, loops, size, elementsPerRow );
        benchmarkSpMV< Real, tnlChunkedEllpackMatrix >( benchmark, loops, size, elementsPerRow );
    }
}

void
setupConfig( tnlConfigDescription & config )
{
    config.addDelimiter( "Benchmark settings:" );
    config.addEntry< tnlString >( "log-file", "Log file name.", "tnl-cuda-benchmarks.log");
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

    ofstream logFile( parameters.getParameter< tnlString >( "log-file" ).getString() );
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
