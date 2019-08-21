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

#include <type_traits>
#include <stdexcept>
#include <algorithm>

#include <TNL/Pointers/DevicePointer.h>
#include <TNL/ParallelFor.h>

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

// copy on the same device
template< typename Matrix1,
          typename Matrix2 >
typename std::enable_if< std::is_same< typename Matrix1::DeviceType, typename Matrix2::DeviceType >::value >::type
copySparseMatrix_impl( Matrix1& A, const Matrix2& B )
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
      const IndexType desGridSize = 32 * Cuda::DeviceInfo::getCudaMultiprocessors( Cuda::DeviceInfo::getActiveDevice() );
      gridSize.x = min( desGridSize, Cuda::getNumberOfBlocks( rows, blockSize.x ) );

      typename Matrix1::CompressedRowLengthsVector rowLengths;
      rowLengths.setSize( rows );

      Pointers::DevicePointer< Matrix1 > Apointer( A );
      const Pointers::DevicePointer< const Matrix2 > Bpointer( B );

      // set row lengths
      Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
      SparseMatrixSetRowLengthsVectorKernel<<< gridSize, blockSize >>>(
            rowLengths.getData(),
            &Bpointer.template getData< TNL::Devices::Cuda >(),
            rows,
            cols );
      TNL_CHECK_CUDA_DEVICE;
      Apointer->setCompressedRowLengths( rowLengths );

      // copy rows
      Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
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

// cross-device copy (host -> gpu)
template< typename Matrix1,
          typename Matrix2 >
typename std::enable_if< ! std::is_same< typename Matrix1::DeviceType, typename Matrix2::DeviceType >::value &&
                           std::is_same< typename Matrix2::DeviceType, Devices::Host >::value >::type
copySparseMatrix_impl( Matrix1& A, const Matrix2& B )
{
   typename Matrix2::CudaType B_tmp;
   B_tmp = B;
   copySparseMatrix_impl( A, B_tmp );
}

// cross-device copy (gpu -> host)
template< typename Matrix1,
          typename Matrix2 >
typename std::enable_if< ! std::is_same< typename Matrix1::DeviceType, typename Matrix2::DeviceType >::value &&
                           std::is_same< typename Matrix2::DeviceType, Devices::Cuda >::value >::type
copySparseMatrix_impl( Matrix1& A, const Matrix2& B )
{
   typename Matrix1::CudaType A_tmp;
   copySparseMatrix_impl( A_tmp, B );
   A = A_tmp;
}

template< typename Matrix1, typename Matrix2 >
void
copySparseMatrix( Matrix1& A, const Matrix2& B )
{
   copySparseMatrix_impl( A, B );
}


template< typename Matrix, typename AdjacencyMatrix >
void
copyAdjacencyStructure( const Matrix& A, AdjacencyMatrix& B,
                        bool has_symmetric_pattern,
                        bool ignore_diagonal )
{
   static_assert( std::is_same< typename Matrix::DeviceType, Devices::Host >::value,
                  "The function is not implemented for CUDA matrices - it would require atomic insertions "
                  "of elements into the sparse format." );
   static_assert( std::is_same< typename Matrix::DeviceType, typename AdjacencyMatrix::DeviceType >::value,
                  "The matrices must be allocated on the same device." );
   static_assert( std::is_same< typename Matrix::IndexType, typename AdjacencyMatrix::IndexType >::value,
                  "The matrices must have the same IndexType." );
//   static_assert( std::is_same< typename AdjacencyMatrix::RealType, bool >::value,
//                  "The RealType of the adjacency matrix must be bool." );

   using IndexType = typename Matrix::IndexType;

   if( A.getRows() != A.getColumns() ) {
      throw std::logic_error( "The matrix is not square: " + std::to_string( A.getRows() ) + " rows, "
                              + std::to_string( A.getColumns() ) + " columns." );
   }

   const IndexType N = A.getRows();
   B.setDimensions( N, N );

   // set row lengths
   typename AdjacencyMatrix::CompressedRowLengthsVector rowLengths;
   rowLengths.setSize( N );
   rowLengths.setValue( 0 );
   for( IndexType i = 0; i < A.getRows(); i++ ) {
      const int maxLength = A.getRowLength( i );
      const auto row = A.getRow( i );
      IndexType length = 0;
      for( int c_j = 0; c_j < maxLength; c_j++ ) {
         const IndexType j = row.getElementColumn( c_j );
         if( j >= A.getColumns() )
            break;
         length++;
         if( ! has_symmetric_pattern && i != j )
            if( A.getElement( j, i ) == 0 )
               rowLengths[ j ]++;
      }
      if( ignore_diagonal )
         length--;
      rowLengths[ i ] += length;
   }
   B.setCompressedRowLengths( rowLengths );

   // set non-zeros
   for( IndexType i = 0; i < A.getRows(); i++ ) {
      const int maxLength = A.getRowLength( i );
      const auto row = A.getRow( i );
      for( int c_j = 0; c_j < maxLength; c_j++ ) {
         const IndexType j = row.getElementColumn( c_j );
         if( j >= A.getColumns() )
            break;
         if( ! ignore_diagonal || i != j )
            if( A.getElement( i, j ) != 0 ) {
               B.setElement( i, j, true );
               if( ! has_symmetric_pattern )
                  B.setElement( j, i, true );
            }
      }
   }
}


template< typename Matrix1, typename Matrix2, typename PermutationArray >
void
reorderSparseMatrix( const Matrix1& matrix1, Matrix2& matrix2, const PermutationArray& perm, const PermutationArray& iperm )
{
   // TODO: implement on GPU
   static_assert( std::is_same< typename Matrix1::DeviceType, Devices::Host >::value, "matrix reordering is implemented only for host" );
   static_assert( std::is_same< typename Matrix2::DeviceType, Devices::Host >::value, "matrix reordering is implemented only for host" );
   static_assert( std::is_same< typename PermutationArray::DeviceType, Devices::Host >::value, "matrix reordering is implemented only for host" );

   using IndexType = typename Matrix1::IndexType;

   matrix2.setDimensions( matrix1.getRows(), matrix1.getColumns() );

   // set row lengths
   typename Matrix2::CompressedRowLengthsVector rowLengths;
   rowLengths.setSize( matrix1.getRows() );
   for( IndexType i = 0; i < matrix1.getRows(); i++ ) {
      const IndexType maxLength = matrix1.getRowLength( perm[ i ] );
      const auto row = matrix1.getRow( perm[ i ] );
      IndexType length = 0;
      for( IndexType j = 0; j < maxLength; j++ )
         if( row.getElementColumn( j ) < matrix1.getColumns() )
            length++;
      rowLengths[ i ] = length;
   }
   matrix2.setCompressedRowLengths( rowLengths );

   // set row elements
   for( IndexType i = 0; i < matrix2.getRows(); i++ ) {
      const IndexType rowLength = rowLengths[ i ];

      // extract sparse row
      const auto row1 = matrix1.getRow( perm[ i ] );

      // permute
      typename Matrix2::IndexType columns[ rowLength ];
      typename Matrix2::RealType values[ rowLength ];
      for( IndexType j = 0; j < rowLength; j++ ) {
         columns[ j ] = iperm[ row1.getElementColumn( j ) ];
         values[ j ] = row1.getElementValue( j );
      }

      // sort
      IndexType indices[ rowLength ];
      for( IndexType j = 0; j < rowLength; j++ )
         indices[ j ] = j;
      // nvcc does not allow lambdas to capture VLAs, even in host code (WTF!?)
      //    error: a variable captured by a lambda cannot have a type involving a variable-length array
      IndexType* _columns = columns;
      auto comparator = [=]( IndexType a, IndexType b ) {
         return _columns[ a ] < _columns[ b ];
      };
      std::sort( indices, indices + rowLength, comparator );

      typename Matrix2::IndexType sortedColumns[ rowLength ];
      typename Matrix2::RealType sortedValues[ rowLength ];
      for( IndexType j = 0; j < rowLength; j++ ) {
         sortedColumns[ j ] = columns[ indices[ j ] ];
         sortedValues[ j ] = values[ indices[ j ] ];
      }

      matrix2.setRow( i, sortedColumns, sortedValues, rowLength );
   }
}

template< typename Array1, typename Array2, typename PermutationArray >
void
reorderArray( const Array1& src, Array2& dest, const PermutationArray& perm )
{
   static_assert( std::is_same< typename Array1::DeviceType, typename Array2::DeviceType >::value,
                  "Arrays must reside on the same device." );
   static_assert( std::is_same< typename Array1::DeviceType, typename PermutationArray::DeviceType >::value,
                  "Arrays must reside on the same device." );
   TNL_ASSERT_EQ( src.getSize(), perm.getSize(),
                  "Source array and permutation must have the same size." );
   TNL_ASSERT_EQ( dest.getSize(), perm.getSize(),
                  "Destination array and permutation must have the same size." );

   using DeviceType = typename Array1::DeviceType;
   using IndexType = typename Array1::IndexType;

   auto kernel = [] __cuda_callable__
      ( IndexType i,
        const typename Array1::ValueType* src,
        typename Array2::ValueType* dest,
        const typename PermutationArray::ValueType* perm )
   {
      dest[ i ] = src[ perm[ i ] ];
   };

   ParallelFor< DeviceType >::exec( (IndexType) 0, src.getSize(),
                                    kernel,
                                    src.getData(),
                                    dest.getData(),
                                    perm.getData() );
}

} // namespace Matrices
} // namespace TNL
