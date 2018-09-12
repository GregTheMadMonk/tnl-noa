#pragma once

#include <algorithm>

#include <TNL/Devices/Host.h>
#include <TNL/ParallelFor.h>

using namespace TNL;

template< typename Matrix, typename PermutationVector >
void
getTrivialOrdering( const Matrix& matrix, PermutationVector& perm, PermutationVector& iperm )
{
   using IndexType = typename Matrix::IndexType;

   // allocate permutation vectors
   perm.setSize( matrix.getRows() );
   iperm.setSize( matrix.getRows() );

   const IndexType N = matrix.getRows() / 2;
   for( IndexType i = 0; i < N; i++ ) {
      perm[ 2 * i ] = i;
      perm[ 2 * i + 1 ] = i + N;
      iperm[ i ] = 2 * i;
      iperm[ i + N ] = 2 * i + 1;
   }
}

template< typename Vector, typename PermutationVector >
void
reorderVector( const Vector& src, Vector& dest, const PermutationVector& perm )
{
   TNL_ASSERT_EQ( src.getSize(), perm.getSize(),
                  "Source vector and permutation must have the same size." );
   using RealType = typename Vector::RealType;
   using DeviceType = typename Vector::DeviceType;
   using IndexType = typename Vector::IndexType;

   auto kernel = [] __cuda_callable__
      ( IndexType i,
        const RealType* src,
        RealType* dest,
        const typename PermutationVector::RealType* perm )
   {
      dest[ i ] = src[ perm[ i ] ];
   };

   dest.setLike( src );

   ParallelFor< DeviceType >::exec( (IndexType) 0, src.getSize(),
                                    kernel,
                                    src.getData(),
                                    dest.getData(),
                                    perm.getData() );
}

template< typename Matrix, typename PermutationVector >
void
reorderMatrix( const Matrix& matrix1, Matrix& matrix2, const PermutationVector& _perm, const PermutationVector& _iperm )
{
   // TODO: implement on GPU
   static_assert( std::is_same< typename Matrix::DeviceType, Devices::Host >::value, "matrix reordering is implemented only for host" );
   static_assert( std::is_same< typename PermutationVector::DeviceType, Devices::Host >::value, "matrix reordering is implemented only for host" );

   using namespace TNL;
   using IndexType = typename Matrix::IndexType;

   matrix2.setLike( matrix1 );

   // general multidimensional accessors for permutation indices
   // TODO: this depends on the specific layout of dofs, general reordering of NDArray is needed
   auto perm = [&]( IndexType dof ) {
      TNL_ASSERT_LT( dof, matrix1.getRows(), "invalid dof index" );
      const IndexType i = dof / _perm.getSize();
      return i * _perm.getSize() + _perm[ dof % _perm.getSize() ];
   };
   auto iperm = [&]( IndexType dof ) {
      TNL_ASSERT_LT( dof, matrix1.getRows(), "invalid dof index" );
      const IndexType i = dof / _iperm.getSize();
      return i * _iperm.getSize() + _iperm[ dof % _iperm.getSize() ];
   };

   // set row lengths
   typename Matrix::CompressedRowLengthsVector rowLengths;
   rowLengths.setSize( matrix1.getRows() );
   for( IndexType i = 0; i < matrix1.getRows(); i++ ) {
      const IndexType maxLength = matrix1.getRowLength( perm( i ) );
      const auto row = matrix1.getRow( perm( i ) );
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
      const auto row1 = matrix1.getRow( perm( i ) );

      // permute
      typename Matrix::IndexType columns[ rowLength ];
      typename Matrix::RealType values[ rowLength ];
      for( IndexType j = 0; j < rowLength; j++ ) {
         columns[ j ] = iperm( row1.getElementColumn( j ) );
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

      typename Matrix::IndexType sortedColumns[ rowLength ];
      typename Matrix::RealType sortedValues[ rowLength ];
      for( IndexType j = 0; j < rowLength; j++ ) {
         sortedColumns[ j ] = columns[ indices[ j ] ];
         sortedValues[ j ] = values[ indices[ j ] ];
      }

      matrix2.setRow( i, sortedColumns, sortedValues, rowLength );
   }
}
