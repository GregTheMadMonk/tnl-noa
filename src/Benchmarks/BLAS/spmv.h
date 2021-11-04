/***************************************************************************
                          spmv.h  -  description
                             -------------------
    begin                : Dec 30, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include "../Benchmarks.h"

#include <TNL/Pointers/DevicePointer.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/CSR.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/Ellpack.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/SlicedEllpack.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/ChunkedEllpack.h>

namespace TNL {
namespace Benchmarks {

// silly alias to match the number of template parameters with other formats
template< typename Real, typename Device, typename Index >
using SlicedEllpack = SpMV::ReferenceFormats::Legacy::SlicedEllpack< Real, Device, Index >;

// Legacy formats
template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Scalar = SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, SpMV::ReferenceFormats::Legacy::CSRScalar >;


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
   const int rowIdx = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
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
   Pointers::DevicePointer< Matrix > kernel_matrix( matrix );
   dim3 cudaBlockSize( 256 ), cudaGridSize( Cuda::getMaxGridSize() );
   const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
   const IndexType cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridSize() );
   for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ ) {
      if( gridIdx == cudaGrids - 1 )
         cudaGridSize.x = cudaBlocks % Cuda::getMaxGridSize();
      setCudaTestMatrixKernel< Matrix >
         <<< cudaGridSize, cudaBlockSize >>>
         ( &kernel_matrix.template modifyData< Devices::Cuda >(), elementsPerRow, gridIdx );
        TNL_CHECK_CUDA_DEVICE;
   }
#endif
}


// TODO: rename as benchmark_SpMV_synthetic and move to spmv-synthetic.h
template< typename Real,
          template< typename, typename, typename > class Matrix >
void
benchmarkSpMV( Benchmark<> & benchmark,
               const int & size,
               const int elementsPerRow = 5 )
{
   typedef Matrix< Real, Devices::Host, int > HostMatrix;
   typedef Matrix< Real, Devices::Cuda, int > DeviceMatrix;
   typedef Containers::Vector< Real, Devices::Host, int > HostVector;
   typedef Containers::Vector< Real, Devices::Cuda, int > CudaVector;

   HostMatrix hostMatrix;
   DeviceMatrix deviceMatrix;
   Containers::Vector< int, Devices::Host, int > hostRowLengths;
   Containers::Vector< int, Devices::Cuda, int > deviceRowLengths;
   HostVector hostVector, hostVector2;
   CudaVector deviceVector, deviceVector2;

   // set metadata
   const std::vector< String > parsedType = parseObjectType( getType< HostMatrix >() );
   benchmark.setMetadataElement({ "format", parsedType[ 0 ] });

   hostRowLengths.setSize( size );
   hostMatrix.setDimensions( size, size );
   hostVector.setSize( size );
   hostVector2.setSize( size );
#ifdef HAVE_CUDA
   deviceRowLengths.setSize( size );
   deviceMatrix.setDimensions( size, size );
   deviceVector.setSize( size );
   deviceVector2.setSize( size );
#endif

   hostRowLengths.setValue( elementsPerRow );
#ifdef HAVE_CUDA
   deviceRowLengths.setValue( elementsPerRow );
#endif

   hostMatrix.setCompressedRowLengths( hostRowLengths );
#ifdef HAVE_CUDA
   deviceMatrix.setCompressedRowLengths( deviceRowLengths );
#endif

   const int elements = setHostTestMatrix< HostMatrix >( hostMatrix, elementsPerRow );
   setCudaTestMatrix< DeviceMatrix >( deviceMatrix, elementsPerRow );
   const double datasetSize = (double) elements * ( 2 * sizeof( Real ) + sizeof( int ) ) / oneGB;

   // reset function
   auto reset = [&]() {
      hostVector.setValue( 1.0 );
      hostVector2.setValue( 0.0 );
#ifdef HAVE_CUDA
      deviceVector.setValue( 1.0 );
      deviceVector2.setValue( 0.0 );
#endif
   };

   // compute functions
   auto spmvHost = [&]() {
      hostMatrix.vectorProduct( hostVector, hostVector2 );
   };
   benchmark.setOperation( datasetSize );
   benchmark.time< Devices::Host >( reset, "CPU", spmvHost );
#ifdef HAVE_CUDA
   auto spmvCuda = [&]() {
      deviceMatrix.vectorProduct( deviceVector, deviceVector2 );
   };
   benchmark.time< Devices::Cuda >( reset, "GPU", spmvCuda );
#endif
}

template< typename Real = double,
          typename Index = int >
void
benchmarkSpmvSynthetic( Benchmark<> & benchmark,
                        const int & size,
                        const int & elementsPerRow )
{
   // TODO: benchmark all formats from tnl-benchmark-spmv (different parameters of the base formats)
   // NOTE: CSR is disabled because it is very slow on GPU
   //benchmarkSpMV< Real, SparseMatrixLegacy_CSR_Scalar >( benchmark, size, elementsPerRow );
   benchmarkSpMV< Real, Benchmarks::SpMV::ReferenceFormats::Legacy::Ellpack >( benchmark, size, elementsPerRow );
   benchmarkSpMV< Real, SlicedEllpack >( benchmark, size, elementsPerRow );
   benchmarkSpMV< Real, Benchmarks::SpMV::ReferenceFormats::Legacy::ChunkedEllpack >( benchmark, size, elementsPerRow );
}

} // namespace Benchmarks
} // namespace TNL
