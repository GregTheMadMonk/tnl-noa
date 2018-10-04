/***************************************************************************
                          SparseOperations_impl.h  -  description
                             -------------------
    begin                : Oct 4, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/Pointers/DevicePointer.h>

namespace TNL {
namespace Matrices {

#ifdef HAVE_CUDA
template< typename Vector, typename Matrix >
__global__ void
SparseMatrixSetRowLengthsVectorKernel( Vector* rowLengths,
                                       const Matrix* matrix,
                                       typename Matrix::IndexType rows,
                                       typename Matrix::IndexType cols )
{
   using IndexType = typename Matrix::IndexType;

   IndexType rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
   const IndexType gridSize = blockDim.x * gridDim.x;

   while( rowIdx < rows ) {
      const auto max_length = matrix->getRowLengthFast( rowIdx );
      const auto row = matrix->getRow( rowIdx );
      IndexType length = 0;
      for( IndexType c_j = 0; c_j < max_length; c_j++ )
         if( row.getElementColumn( c_j ) < cols )
            length++;
         else
            break;
      rowLengths[ rowIdx ] = length;
      rowIdx += gridSize;
   }
}

template< typename Matrix1, typename Matrix2 >
__global__ void
SparseMatrixCopyKernel( Matrix1* A,
                        const Matrix2* B,
                        const typename Matrix2::IndexType* rowLengths,
                        typename Matrix2::IndexType rows )
{
   using IndexType = typename Matrix2::IndexType;

   IndexType rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
   const IndexType gridSize = blockDim.x * gridDim.x;

   while( rowIdx < rows ) {
      const auto length = rowLengths[ rowIdx ];
      const auto rowB = B->getRow( rowIdx );
      auto rowA = A->getRow( rowIdx );
      for( IndexType c = 0; c < length; c++ )
         rowA.setElement( c, rowB.getElementColumn( c ), rowB.getElementValue( c ) );
      rowIdx += gridSize;
   }
}
#endif


template< typename Matrix1, typename Matrix2 >
void
copySparseMatrix( Matrix1& A, const Matrix2& B )
{
   static_assert( std::is_same< typename Matrix1::RealType, typename Matrix2::RealType >::value,
                  "The matrices must have the same RealType." );
   static_assert( std::is_same< typename Matrix1::DeviceType, typename Matrix2::DeviceType >::value,
                  "The matrices must be allocated on the same device." );
   static_assert( std::is_same< typename Matrix1::IndexType, typename Matrix2::IndexType >::value,
                  "The matrices must have the same IndexType." );

   using RealType = typename Matrix1::RealType;
   using DeviceType = typename Matrix1::DeviceType;
   using IndexType = typename Matrix1::IndexType;

   const IndexType rows = B.getRows();
   const IndexType cols = B.getColumns();

   A.setDimensions( rows, cols );

   if( std::is_same< DeviceType, Devices::Host >::value ) {
      // set row lengths
      typename Matrix1::CompressedRowLengthsVector rowLengths;
      rowLengths.setSize( rows );
#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
      for( IndexType i = 0; i < rows; i++ ) {
         const auto max_length = B.getRowLength( i );
         const auto row = B.getRow( i );
         IndexType length = 0;
         for( IndexType c_j = 0; c_j < max_length; c_j++ )
            if( row.getElementColumn( c_j ) < cols )
               length++;
            else
               break;
         rowLengths[ i ] = length;
      }
      A.setCompressedRowLengths( rowLengths );

#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
      for( IndexType i = 0; i < rows; i++ ) {
         const auto length = rowLengths[ i ];
         const auto rowB = B.getRow( i );
         auto rowA = A.getRow( i );
         for( IndexType c = 0; c < length; c++ )
            rowA.setElement( c, rowB.getElementColumn( c ), rowB.getElementValue( c ) );
      }
   }

   if( std::is_same< DeviceType, Devices::Cuda >::value ) {
#ifdef HAVE_CUDA
      dim3 blockSize( 256 );
      dim3 gridSize;
      const IndexType desGridSize = 32 * Devices::CudaDeviceInfo::getCudaMultiprocessors( Devices::CudaDeviceInfo::getActiveDevice() );
      gridSize.x = min( desGridSize, Devices::Cuda::getNumberOfBlocks( rows, blockSize.x ) );

      typename Matrix1::CompressedRowLengthsVector rowLengths;
      rowLengths.setSize( rows );

      Pointers::DevicePointer< Matrix1 > Apointer( A );
      const Pointers::DevicePointer< const Matrix2 > Bpointer( B );

      // set row lengths
      Devices::Cuda::synchronizeDevice();
      SparseMatrixSetRowLengthsVectorKernel<<< gridSize, blockSize >>>(
            rowLengths.getData(),
            &Bpointer.template getData< TNL::Devices::Cuda >(),
            rows,
            cols );
      TNL_CHECK_CUDA_DEVICE;
      Apointer->setCompressedRowLengths( rowLengths );

      // copy rows
      Devices::Cuda::synchronizeDevice();
      SparseMatrixCopyKernel<<< gridSize, blockSize >>>(
            &Apointer.template modifyData< TNL::Devices::Cuda >(),
            &Bpointer.template getData< TNL::Devices::Cuda >(),
            rowLengths.getData(),
            rows );
      TNL_CHECK_CUDA_DEVICE;
#else
      throw Exceptions::CudaSupportMissing();
#endif
   }
}

} // namespace Matrices
} // namespace TNL
