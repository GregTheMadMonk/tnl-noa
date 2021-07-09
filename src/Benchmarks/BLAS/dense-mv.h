/***************************************************************************
                          dense-mv.h  -  description
                             -------------------
    begin                : Jul 8, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include "../Benchmarks.h"
#include "cublasWrappers.h"

#include <TNL/Containers/Vector.h>
#include <TNL/Pointers/DevicePointer.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Host.h>

namespace TNL {
namespace Benchmarks {

template< typename Matrix >
void setMatrix( Matrix& matrix )
{
   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;
   matrix.forAllElements( [] __cuda_callable__ ( IndexType rowIdx, IndexType columnIdx, IndexType columnIdx_, RealType& value ) {
       value = 1.0; } );
}

template< typename Real >
void
benchmarkDenseMVSynthetic( Benchmark<> & benchmark,
                           const int & size )
{
   using HostMatrix = TNL::Matrices::DenseMatrix< Real, TNL::Devices::Host >;
   using RowMajorCudaMatrix = TNL::Matrices::DenseMatrix< Real, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::RowMajorOrder >;
   using ColumnMajorCudaMatrix = TNL::Matrices::DenseMatrix< Real, TNL::Devices::Cuda >;
   using HostVector = Containers::Vector< Real, Devices::Host, int >;
   using CudaVector = Containers::Vector< Real, Devices::Cuda, int >;

   HostMatrix hostMatrix;
   RowMajorCudaMatrix rowMajorCudaMatrix;
   ColumnMajorCudaMatrix columnMajorCudaMatrix;
   HostVector inHostVector, outHostVector;
   CudaVector inCudaVector, outCudaVector1, outCudaVector2;

   // create benchmark group
   const std::vector< String > parsedType = parseObjectType( getType< HostMatrix >() );
#ifdef HAVE_CUDA
   benchmark.createHorizontalGroup( parsedType[ 0 ], 2 );
#else
   benchmark.createHorizontalGroup( parsedType[ 0 ], 1 );
#endif

   hostMatrix.setDimensions( size, size );
   inHostVector.setSize( size );
   outHostVector.setSize( size );

   setMatrix< HostMatrix >( hostMatrix );
   const double datasetSize = (double) ( size * size ) * ( 2 * sizeof( Real ) + sizeof( int ) ) / oneGB;

   // reset function
   auto reset = [&]() {
      inHostVector = 1.0;
      outHostVector = 0.0;
#ifdef HAVE_CUDA
      inCudaVector = 1.0;
      //outCudaVector1 = 0.0;
      //outCudaVector2 = 0.0;
#endif
   };

   // compute functions
   auto spmvHost = [&]() {
      hostMatrix.vectorProduct( inHostVector, outHostVector );
   };
   benchmark.setOperation( datasetSize );
   benchmark.time< Devices::Host >( reset, "CPU", spmvHost );

#ifdef HAVE_CUDA
   columnMajorCudaMatrix.setDimensions( size, size );
   inCudaVector.setSize( size );
   outCudaVector1.setSize( size );
   outCudaVector2.setSize( size );
   setMatrix< ColumnMajorCudaMatrix >( columnMajorCudaMatrix );

   auto columnMajorMvCuda = [&]() {
      columnMajorCudaMatrix.vectorProduct( inCudaVector, outCudaVector1 );
   };
   benchmark.time< Devices::Cuda >( reset, "GPU col", columnMajorMvCuda );

   columnMajorCudaMatrix.reset();

   rowMajorCudaMatrix.setDimensions( size, size );
   setMatrix< RowMajorCudaMatrix >( rowMajorCudaMatrix );

   auto rowMajorMvCuda = [&]() {
      rowMajorCudaMatrix.vectorProduct( inCudaVector, outCudaVector2 );
   };
   benchmark.time< Devices::Cuda >( reset, "GPU row", rowMajorMvCuda );

   //std::cerr << "Diff. = " << TNL::max( abs( outCudaVector2 - outCudaVector1 ) ) << std::endl;

   rowMajorCudaMatrix.reset();
   columnMajorCudaMatrix.setDimensions( size, size );
   setMatrix< ColumnMajorCudaMatrix >( columnMajorCudaMatrix );

   cublasHandle_t cublasHandle;
   cublasCreate( &cublasHandle );
   auto mvCublas = [&] () {
      Real alpha = 1.0;
      Real beta = 0.0;
      cublasGemv( cublasHandle, CUBLAS_OP_N, size, size, &alpha,
                  columnMajorCudaMatrix.getValues().getData(), size,
                  inCudaVector.getData(), 1, &beta,
                  outCudaVector1.getData(), 1 );
   };
   benchmark.time< Devices::Cuda >( reset, "GPU cublas", mvCublas );

#endif
}

/*template< typename Real = double,
          typename Index = int >
void
benchmarkDenseSynthetic( Benchmark<> & benchmark,
                         const int & size )
{
   // TODO: benchmark all formats from tnl-benchmark-spmv (different parameters of the base formats)
   // NOTE: CSR is disabled because it is very slow on GPU
   //benchmarkSpMV< Real, SparseMatrixLegacy_CSR_Scalar >( benchmark, size, elementsPerRow );
   benchmarkSpMV< Real, Benchmarks::SpMV::ReferenceFormats::Legacy::Ellpack >( benchmark, size, elementsPerRow );
   benchmarkSpMV< Real, SlicedEllpack >( benchmark, size, elementsPerRow );
   benchmarkSpMV< Real, Benchmarks::SpMV::ReferenceFormats::Legacy::ChunkedEllpack >( benchmark, size, elementsPerRow );
}*/

} // namespace Benchmarks
} // namespace TNL
