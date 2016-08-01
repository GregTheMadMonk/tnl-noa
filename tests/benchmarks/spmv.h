#pragma once

#include "benchmarks.h"

#include <TNL/List.h>
#include <TNL/Matrices/CSRMatrix.h>
#include <TNL/Matrices/EllpackMatrix.h>
#include <TNL/Matrices/SlicedEllpackMatrix.h>
#include <TNL/Matrices/ChunkedEllpackMatrix.h>

namespace TNL
{
namespace benchmarks
{

// silly alias to match the number of template parameters with other formats
template< typename Real, typename Device, typename Index >
using SlicedEllpackMatrix = Matrices::SlicedEllpackMatrix< Real, Device, Index >;

template< typename Matrix >
int setHostTestMatrix( Matrix& matrix,
                       const int elementsPerRow )
{
    const int size = matrix.getRows();
    int elements( 0 );
    for( int row = 0; row < size; row++ ) {
        int col = row - elementsPerRow / 2;
        for( int element = 0; element < elementsPerRow; element++ ) {
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

#ifdef HAVE_CUDA
template< typename Matrix >
__global__ void setCudaTestMatrixKernel( Matrix* matrix,
                                         const int elementsPerRow,
                                         const int gridIdx )
{
    const int rowIdx = ( gridIdx * Devices::Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
    if( rowIdx >= matrix->getRows() )
        return;
    int col = rowIdx - elementsPerRow / 2;
    for( int element = 0; element < elementsPerRow; element++ ) {
        if( col + element >= 0 &&
            col + element < matrix->getColumns() )
           matrix->setElementFast( rowIdx, col + element, element + 1 );
    }
}
#endif

template< typename Matrix >
void setCudaTestMatrix( Matrix& matrix,
                        const int elementsPerRow )
{
#ifdef HAVE_CUDA
    typedef typename Matrix::IndexType IndexType;
    typedef typename Matrix::RealType RealType;
    Matrix* kernel_matrix = Devices::Cuda::passToDevice( matrix );
    dim3 cudaBlockSize( 256 ), cudaGridSize( Devices::Cuda::getMaxGridSize() );
    const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
    const IndexType cudaGrids = roundUpDivision( cudaBlocks, Devices::Cuda::getMaxGridSize() );
    for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ ) {
        if( gridIdx == cudaGrids - 1 )
            cudaGridSize.x = cudaBlocks % Devices::Cuda::getMaxGridSize();
        setCudaTestMatrixKernel< Matrix >
            <<< cudaGridSize, cudaBlockSize >>>
            ( kernel_matrix, elementsPerRow, gridIdx );
        checkCudaDevice;
    }
    Devices::Cuda::freeFromDevice( kernel_matrix );
#endif
}


// TODO: rename as benchmark_SpMV_synthetic and move to spmv-synthetic.h
template< typename Real,
          template< typename, typename, typename > class Matrix,
          template< typename, typename, typename > class Vector = Vectors::Vector >
bool
benchmarkSpMV( Benchmark & benchmark,
               const int & loops,
               const int & size,
               const int elementsPerRow = 5 )
{
    typedef Matrix< Real, Devices::Host, int > HostMatrix;
    typedef Matrix< Real, Devices::Cuda, int > DeviceMatrix;
    typedef Vectors::Vector< Real, Devices::Host, int > HostVector;
    typedef Vectors::Vector< Real, Devices::Cuda, int > CudaVector;

    HostMatrix hostMatrix;
    DeviceMatrix deviceMatrix;
    Vectors::Vector< int, Devices::Host, int > hostRowLengths;
    Vectors::Vector< int, Devices::Cuda, int > deviceRowLengths;
    HostVector hostVector, hostVector2;
    CudaVector deviceVector, deviceVector2;

    // create benchmark group
    List< String > parsedType;
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
        std::cerr << msg << std::endl;
        benchmark.addErrorMessage( msg, 2 );
        return false;
    }

    hostRowLengths.setValue( elementsPerRow );
    deviceRowLengths.setValue( elementsPerRow );

    if( ! hostMatrix.setCompressedRowsLengths( hostRowLengths ) ) {
        const char* msg = "error: allocation of host matrix failed";
        std::cerr << msg << std::endl;
        benchmark.addErrorMessage( msg, 2 );
        return false;
    }
    if( ! deviceMatrix.setCompressedRowsLengths( deviceRowLengths ) ) {
        const char* msg = "error: allocation of device matrix failed";
        std::cerr << msg << std::endl;
        benchmark.addErrorMessage( msg, 2 );
        return false;
    }

    const int elements = setHostTestMatrix< HostMatrix >( hostMatrix, elementsPerRow );
    setCudaTestMatrix< DeviceMatrix >( deviceMatrix, elementsPerRow );
    const double datasetSize = ( double ) loops * elements * ( 2 * sizeof( Real ) + sizeof( int ) ) / oneGB;

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

template< typename Real = double,
          typename Index = int >
bool
benchmarkSpmvSynthetic( Benchmark & benchmark,
                        const int & loops,
                        const int & size,
                        const int & elementsPerRow )
{
    // TODO: benchmark all formats from tnl-benchmark-spmv (different parameters of the base formats)
    benchmarkSpMV< Real, Matrices::CSRMatrix >( benchmark, loops, size, elementsPerRow );
    benchmarkSpMV< Real, Matrices::EllpackMatrix >( benchmark, loops, size, elementsPerRow );
    benchmarkSpMV< Real, SlicedEllpackMatrix >( benchmark, loops, size, elementsPerRow );
    benchmarkSpMV< Real, Matrices::ChunkedEllpackMatrix >( benchmark, loops, size, elementsPerRow );
}

} // namespace benchmarks
} // namespace tnl
