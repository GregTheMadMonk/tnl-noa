/***************************************************************************
                          SparseMatrix.h -  description
                             -------------------
    begin                : Nov 29, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <functional>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Algorithms/Reduction.h>

namespace TNL {
namespace Matrices {

   template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
SparseMatrix( const RealAllocatorType& realAllocator,
              const IndexAllocatorType& indexAllocator )
   : Matrix< Real, Device, Index, RealAllocator >( realAllocator ), columnIndexes( indexAllocator )
{
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
SparseMatrix( const SparseMatrix& m )
   : Matrix< Real, Device, Index, RealAllocator >( m ), columnIndexes( m.columnIndexes )
{
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
SparseMatrix( const SparseMatrix&& m )
   : Matrix< Real, Device, Index, RealAllocator >( std::move( m ) ), columnIndexes( std::move( m.columnIndexes ) )
{
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
SparseMatrix( const IndexType rows,
              const IndexType columns,
              const RealAllocatorType& realAllocator,
              const IndexAllocatorType& indexAllocator )
: Matrix< Real, Device, Index, RealAllocator >( rows, columns, realAllocator ), columnIndexes( indexAllocator )
{
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
String
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
getSerializationType()
{
   return String( "Matrices::SparseMatrix< " ) +
             TNL::getSerializationType< RealType >() + ", " +
             TNL::getSerializationType< SegmentsType >() + ", [any_device], " +
             TNL::getSerializationType< IndexType >() + ", [any_allocator] >";
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
String
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
setCompressedRowLengths( ConstCompressedRowLengthsVectorView rowLengths )
{
   TNL_ASSERT_EQ( rowLengths.getSize(), this->getRows(), "Number of matrix rows does not fit with rowLengths vector size." );
   this->segments.setSegmentsSizes( rowLengths );
   this->values.setSize( this->segments.getStorageSize() );
   this->values = ( RealType ) 0;
   this->columnIndexes.setSize( this->segments.getStorageSize() );
   this->columnIndexes = this->getPaddingIndex();
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Vector >
void
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
getCompressedRowLengths( Vector& rowLengths ) const
{
   rowLengths.setSize( this->getRows() );
   rowLengths = 0;
   auto rowLengths_view = rowLengths.getView();
   auto fetch = [] __cuda_callable__ ( IndexType row, IndexType column, const RealType& value ) -> IndexType {
      return ( value != 0.0 );
   };
   auto reduce = [] __cuda_callable__ ( IndexType& aux, const IndexType a ) {
      aux += a;
   };
   auto keep = [=] __cuda_callable__ ( const IndexType rowIdx, const IndexType value ) mutable {
      rowLengths_view[ rowIdx ] = value;
   };
   this->allRowsReduction( fetch, reduce, keep, 0 );
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
Index
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
getRowLength( const IndexType row ) const
{

}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
Index
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
getRowLengthFast( const IndexType row ) const
{

}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
Index
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
getNonZeroRowLength( const IndexType row ) const
{

}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
Index
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
getNonZeroRowLengthFast( const IndexType row ) const
{

}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Real2, template< typename, typename, typename > class Segments2,  typename Device2, typename Index2, typename RealAllocator2, typename IndexAllocator2 >
void
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
setLike( const SparseMatrix< Real2, Segments2, Device2, Index2, RealAllocator2, IndexAllocator2 >& matrix )
{
   Matrix< Real, Device, Index, RealAllocator >::setLike( matrix );
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
Index
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
getNumberOfNonzeroMatrixElements() const
{
   const auto columns_view = this->columnIndexes.getConstView();
   const IndexType paddingIndex = this->getPaddingIndex();
   auto fetch = [=] __cuda_callable__ ( const IndexType i ) -> IndexType {
      return ( columns_view[ i ] != paddingIndex );
   };
   return Algorithms::Reduction< DeviceType >::reduce( this->columnIndexes.getSize(), std::plus<>{}, fetch, 0 );
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
reset()
{
   Matrix< Real, Device, Index >::reset();
   this->columnIndexes.reset();

}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
bool
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
setElementFast( const IndexType row,
                const IndexType column,
                const RealType& value )
{
   return this->addElementFast( row, column, value, 0.0 );
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
bool
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
setElement( const IndexType row,
            const IndexType column,
            const RealType& value )
{
   return this->addElement( row, column, value, 0.0 );
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
bool
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
addElementFast( const IndexType row,
                const IndexType column,
                const RealType& value,
                const RealType& thisElementMultiplicator )
{

}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
bool
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
addElement( const IndexType row,
            const IndexType column,
            const RealType& value,
            const RealType& thisElementMultiplicator )
{
   TNL_ASSERT( row >= 0 && row < this->rows &&
               column >= 0 && column < this->columns,
               std::cerr << " row = " << row
                    << " column = " << column
                    << " this->rows = " << this->rows
                    << " this->columns = " << this->columns );

   const IndexType rowSize = this->segments.getSegmentSize( row );
   IndexType col( this->getPaddingIndex() );
   IndexType i;
   IndexType globalIdx;
   for( i = 0; i < rowSize; i++ )
   {
      globalIdx = this->segments.getGlobalIndex( row, i );
      TNL_ASSERT_LT( globalIdx, this->columnIndexes.getSize(), "" );
      col = this->columnIndexes.getElement( globalIdx );
      if( col == column )
      {
         this->values.setElement( globalIdx, thisElementMultiplicator * this->values.getElement( globalIdx ) + value );
         return true;
      }
      if( col == this->getPaddingIndex() || col > column )
         break;
   }
   if( i == rowSize )
      return false;
   if( col == this->getPaddingIndex() )
   {
      this->columnIndexes.setElement( globalIdx, column );
      this->values.setElement( globalIdx, value );
      return true;
   }
   else
   {
      IndexType j = rowSize - 1;
      while( j > i )
      {
         const IndexType globalIdx1 = this->segments.getGlobalIndex( row, j );
         const IndexType globalIdx2 = this->segments.getGlobalIndex( row, j - 1 );
         TNL_ASSERT_LT( globalIdx1, this->columnIndexes.getSize(), "" );
         TNL_ASSERT_LT( globalIdx2, this->columnIndexes.getSize(), "" );
         this->columnIndexes.setElement( globalIdx1, this->columnIndexes.getElement( globalIdx2 ) );
         this->values.setElement( globalIdx1, this->values.getElement( globalIdx2 ) );
         j--;
      }

      this->columnIndexes.setElement( globalIdx, column );
      this->values.setElement( globalIdx, value );
      return true;
   }
}


template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
bool
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
setRowFast( const IndexType row,
            const IndexType* columnIndexes,
            const RealType* values,
            const IndexType elements )
{
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
bool
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
setRow( const IndexType row,
        const IndexType* columnIndexes,
        const RealType* values,
        const IndexType elements )
{
   const IndexType rowLength = this->segments.getSegmentSize( row );
   if( elements > rowLength )
      return false;

   for( IndexType i = 0; i < elements; i++ )
   {
      const IndexType globalIdx = this->segments.getGlobalIndex( row, i );
      this->columnIndexes.setElement( globalIdx, columnIndexes[ i ] );
      this->values.setElement( globalIdx, values[ i ] );
   }
   for( IndexType i = elements; i < rowLength; i++ )
      this->columnIndexes.setElement( this->segments.getGlobalIndex( row, i ), this->getPaddingIndex() );
   return true;
}


template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
bool
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
addRowFast( const IndexType row,
            const IndexType* columns,
            const RealType* values,
            const IndexType numberOfElements,
            const RealType& thisElementMultiplicator )
{

}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
bool
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
addRow( const IndexType row,
        const IndexType* columns,
        const RealType* values,
        const IndexType numberOfElements,
        const RealType& thisElementMultiplicator )
{

}


template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
Real
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
getElementFast( const IndexType row,
                const IndexType column ) const
{

}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
Real
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
getElement( const IndexType row,
            const IndexType column ) const
{
   const IndexType rowSize = this->segments.getSegmentSize( row );
   for( IndexType i = 0; i < rowSize; i++ )
   {
      const IndexType globalIdx = this->segments.getGlobalIndex( row, i );
      TNL_ASSERT_LT( globalIdx, this->columnIndexes.getSize(), "" );
      const IndexType col = this->columnIndexes.getElement( globalIdx );
      if( col == column )
         return this->values.getElement( globalIdx );
   }
   return 0.0;
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
void
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
getRowFast( const IndexType row,
            IndexType* columns,
            RealType* values ) const
{

}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Vector >
__cuda_callable__
typename Vector::RealType
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
rowVectorProduct( const IndexType row,
                  const Vector& vector ) const
{

}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
template< typename InVector,
       typename OutVector >
void
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
vectorProduct( const InVector& inVector,
               OutVector& outVector,
               const RealType& matrixMultiplicator,
               const RealType& inVectorAddition ) const
{
   const auto inVectorView = inVector.getConstView();
   auto outVectorView = outVector.getView();
   const auto valuesView = this->values.getConstView();
   const auto columnIndexesView = this->columnIndexes.getConstView();
   const IndexType paddingIndex = this->getPaddingIndex();
   auto fetch = [=] __cuda_callable__ ( IndexType row, IndexType offset ) -> RealType {
      const IndexType column = columnIndexesView[ offset ];
      if( column == paddingIndex )
         return 0.0;
      return valuesView[ offset ] * inVectorView[ column ];
   };
   auto reduction = [] __cuda_callable__ ( RealType& sum, const RealType& value ) {
      sum += value;
   };
   auto keeper = [=] __cuda_callable__ ( IndexType row, const RealType& value ) mutable {
      outVectorView[ row ] = value;
   };
   this->segments.segmentsReduction( 0, this->getRows(), fetch, reduction, keeper, ( RealType ) 0.0 );
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
void
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
rowsReduction( IndexType first, IndexType last, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchValue& zero ) const
{
   const auto columns_view = this->columnIndexes.getConstView();
   const auto values_view = this->values.getConstView();
   const IndexType paddingIndex_ = this->getPaddingIndex();
   auto fetch_ = [=] __cuda_callable__ ( IndexType rowIdx, IndexType globalIdx ) mutable -> decltype( fetch( IndexType(), IndexType(), RealType() ) ) {
      IndexType columnIdx = columns_view[ globalIdx ];
      if( columnIdx != paddingIndex_ )
         return fetch( rowIdx, columnIdx, values_view[ globalIdx ] );
      return zero;
   };
   this->segments.segmentsReduction( first, last, fetch_, reduce, keep, zero );
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
allRowsReduction( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const
{
   this->rowsReduction( 0, this->getRows(), fetch, reduce, keep, zero );
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
forRows( IndexType first, IndexType last, Function& function ) const
{
   const auto columns_view = this->columnIndexes.getConstView();
   const auto values_view = this->values.getConstView();
   const IndexType paddingIndex_ = this->getPaddingIndex();
   /*auto fetch_ = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable -> decltype( fetch( IndexType(), IndexType(), RealType() ) ) {
      IndexType columnIdx = columns_view[ globalIdx ];
      if( columnIdx != paddingIndex_ )
         return fetch( rowIdx, columnIdx, values_view[ globalIdx ] );
      return zero;
   };
   this->segments.segmentsReduction( first, last, fetch_, reduce, keep, zero );*/

}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
forAllRows( Function& function ) const
{
   this->forRows( 0, this->getRows(), function );
}

/*template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Real2, template< typename, typename > class Segments2, typename Index2, typename RealAllocator2, typename IndexAllocator2 >
void
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
addMatrix( const SparseMatrix< Real2, Segments2, Device, Index2, RealAllocator2, IndexAllocator2 >& matrix,
           const RealType& matrixMultiplicator,
           const RealType& thisMatrixMultiplicator )
{

}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Real2, typename Index2 >
void
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
getTransposition( const SparseMatrix< Real2, Device, Index2 >& matrix,
                  const RealType& matrixMultiplicator )
{

}*/

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Vector1, typename Vector2 >
bool
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
performSORIteration( const Vector1& b,
                     const IndexType row,
                     Vector2& x,
                     const RealType& omega ) const
{

}

// copy assignment
template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >&
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
operator=( const SparseMatrix& matrix )
{
   Matrix< Real, Device, Index >::operator=( matrix );
   this->columnIndexes = matrix.columnIndexes;
   this->segments = matrix.segments;
   this->indexAllocator = matrix.indexAllocator;
   this->realAllocator = matrix.realAllocator;
}

// cross-device copy assignment
template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Real2,
             template< typename, typename, typename > class Segments2,
             typename Device2,
             typename Index2,
             typename RealAllocator2,
             typename IndexAllocator2 >
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >&
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
operator=( const SparseMatrix< Real2, Segments2, Device2, Index2, RealAllocator2, IndexAllocator2 >& matrix )
{
   using RHSMatrixType = SparseMatrix< Real2, Segments2, Device2, Index2, RealAllocator2, IndexAllocator2 >;
   RowsCapacitiesType rowLengths;
   matrix.getCompressedRowLengths( rowLengths );
   this->setCompressedRowLengths( rowLengths );

   // TODO: Replace this with SparseMatrixView
   const auto matrix_columns_view = matrix.columnIndexes.getConstView();
   const auto matrix_values_view = matrix.values.getConstView();
   const IndexType paddingIndex = this->getPaddingIndex();
   auto this_columns_view = this->columnIndexes.getView();
   auto this_values_view = this->values.getView();

   if( std::is_same< Device, Device2 >::value )
   {
      const auto this_segments_view = this->segments.getView();
      auto f = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable {
         const IndexType column = matrix_columns_view[ globalIdx ];
         if( column != paddingIndex )
         {
            const RealType value = matrix_values_view[ globalIdx ];
            IndexType thisGlobalIdx = this_segments_view.getGlobalIndex( rowIdx, localIdx );
            this_columns_view[ thisGlobalIdx ] = column;
            this_values_view[ thisGlobalIdx ] = value;
         }
      };
      matrix.forAllRows( f );
   }
   else
   {
      const IndexType maxRowLength = max( rowLengths );
      const IndexType bufferRowsCount( 128 );
      const size_t bufferSize = bufferRowsCount * maxRowLength;
      Containers::Vector< Real2, Device2, Index2, RealAllocator2 > matrixValuesBuffer( bufferSize );
      Containers::Vector< Index2, Device2, Index2, IndexAllocator2 > matrixColumnsBuffer( bufferSize );
      Containers::Vector< RealType, DeviceType, IndexType, RealAllocator > thisValuesBuffer( bufferSize );
      Containers::Vector< IndexType, DeviceType, IndexType, IndexAllocator > thisColumnsBuffer( bufferSize );
      auto matrixValuesBuffer_view = matrixValuesBuffer.getView();
      auto matrixColumnsBuffer_view = matrixColumnsBuffer.getView();
      auto thisValuesBuffer_view = thisValuesBuffer.getView();
      auto thisColumnsBuffer_view = thisColumnsBuffer.getView();

      IndexType baseRow( 0 );
      const IndexType rowsCount = this->getRows();
      while( baseRow < rowsCount )
      {
         const IndexType lastRow = min( baseRow + bufferRowsCount, rowsCount );
         thisColumnsBuffer = paddingIndex;

         ////
         // Copy matrix elements into buffer
         auto f1 = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable {
            const IndexType column = matrix_columns_view[ globalIdx ];
            if( column != paddingIndex )
            {
               const IndexType bufferIdx = ( rowIdx - baseRow ) * maxRowLength + localIdx;
               matrixValuesBuffer_view[ bufferIdx ] = matrix_values_view[ globalIdx ];
               matrixColumnsBuffer_view[ bufferIdx ] = column;
            }
         };
         matrix.forRows( baseRow, lastRow, f1 );

         ////
         // Copy the source matrix buffer to this matrix buffer
         thisValuesBuffer_view = matrixValuesBuffer_view;
         thisColumnsBuffer_view = matrixColumnsBuffer_view;

         ////
         // Copy matrix elements from the buffer to the matrix
         auto f2 = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable {
            const IndexType bufferIdx = ( rowIdx - baseRow ) * maxRowLength + localIdx;
            const IndexType column = thisColumnsBuffer_view[ bufferIdx ];
            if( column != paddingIndex )
            {
               this_columns_view[ globalIdx ] = column;
               this_values_view[ globalIdx ] = thisValuesBuffer_view[ bufferIdx ];
            }
         };
         this->forRows( baseRow, lastRow, f2 );
         baseRow += bufferRowsCount;
      }
   }
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
save( File& file ) const
{
   Matrix< RealType, DeviceType, IndexType >::save( file );
   file << this->columnIndexes;
   this->segments.save( file );
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
load( File& file )
{
   Matrix< RealType, DeviceType, IndexType >::load( file );
   file >> this->columnIndexes;
   this->segments.load( file );
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
print( std::ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      const IndexType rowLength = this->segments.getSegmentSize( row );
      for( IndexType i = 0; i < rowLength; i++ )
      {
         const IndexType globalIdx = this->segments.getGlobalIndex( row, i );
         const IndexType column = this->columnIndexes.getElement( globalIdx );
         if( column == this->getPaddingIndex() )
            break;
         str << " Col:" << column << "->" << this->values.getElement( globalIdx ) << "\t";
      }
      str << std::endl;
   }
}

template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
Index
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
getPaddingIndex() const
{
   return -1;
}

   } //namespace Matrices
} // namespace  TNL
