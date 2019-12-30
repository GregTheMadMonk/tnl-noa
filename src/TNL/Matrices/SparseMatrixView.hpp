/***************************************************************************
                          SparseMatrixView.hpp -  description
                             -------------------
    begin                : Dec 28, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <functional>
#include <TNL/Matrices/SparseMatrixView.h>
#include <TNL/Algorithms/Reduction.h>

namespace TNL {
namespace Matrices {

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
__cuda_callable__
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
SparseMatrixView()
{
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
__cuda_callable__
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
SparseMatrixView( const IndexType rows,
                  const IndexType columns,
                  const ValuesViewType& values,
                  const ColumnsIndexesViewType& columnIndexes,
                  const SegmentsViewType& segments )
 : MatrixView< Real, Device, Index >( rows, columns, values ), columnIndexes( columnIndexes ), segments( segments )
{
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
__cuda_callable__
auto
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
getView() -> ViewType
{
   return ViewType( this->getRows(), 
                    this->getColumns(),
                    this->getValues().getView(),
                    this->columnIndexes.getView(),
                    this->segments.getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
__cuda_callable__
auto
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
getConstView() const -> ConstViewType
{
   return ConstViewType( this->getRows(),
                         this->getColumns(),
                         this->getValues().getConstView(),
                         this->getColumnsIndexes().getConstView(),
                         this->segments.getConstView() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
String
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
getSerializationType()
{
   return String( "Matrices::SparseMatrix< " ) +
             TNL::getSerializationType< RealType >() + ", " +
             TNL::getSerializationType< SegmentsViewType >() + ", [any_device], " +
             TNL::getSerializationType< IndexType >() + ", [any_allocator] >";
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
String
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
   template< typename Vector >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
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
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
Index
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
getRowLength( const IndexType row ) const
{

}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
__cuda_callable__
Index
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
getRowLengthFast( const IndexType row ) const
{

}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
Index
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
getNonZeroRowLength( const IndexType row ) const
{

}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
__cuda_callable__
Index
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
getNonZeroRowLengthFast( const IndexType row ) const
{

}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
Index
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
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
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
reset()
{
   Matrix< Real, Device, Index >::reset();
   this->columnIndexes.reset();

}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
__cuda_callable__ auto
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
getRow( const IndexType& rowIdx ) const -> const RowView
{
   TNL_ASSERT_LT( rowIdx, this->getRows(), "Row index is larger than number of matrix rows." );
   return RowView( this->segments.getSegmentView( rowIdx ), this->values.getView(), this->columnIndexes.getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
__cuda_callable__ auto
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
getRow( const IndexType& rowIdx ) -> RowView
{
   TNL_ASSERT_LT( rowIdx, this->getRows(), "Row index is larger than number of matrix rows." );
   return RowView( this->segments.getSegmentView( rowIdx ), this->values.getView(), this->columnIndexes.getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
__cuda_callable__
bool
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
setElementFast( const IndexType row,
                const IndexType column,
                const RealType& value )
{
   return this->addElementFast( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
bool
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
setElement( const IndexType row,
            const IndexType column,
            const RealType& value )
{
   return this->addElement( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
__cuda_callable__
bool
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
addElementFast( const IndexType row,
                const IndexType column,
                const RealType& value,
                const RealType& thisElementMultiplicator )
{

}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
bool
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
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
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
__cuda_callable__
bool
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
setRowFast( const IndexType row,
            const IndexType* columnIndexes,
            const RealType* values,
            const IndexType elements )
{
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
bool
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
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
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
__cuda_callable__
bool
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
addRowFast( const IndexType row,
            const IndexType* columns,
            const RealType* values,
            const IndexType numberOfElements,
            const RealType& thisElementMultiplicator )
{

}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
bool
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
addRow( const IndexType row,
        const IndexType* columns,
        const RealType* values,
        const IndexType numberOfElements,
        const RealType& thisElementMultiplicator )
{

}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
__cuda_callable__
Real
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
getElementFast( const IndexType row,
                const IndexType column ) const
{

}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
Real
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
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
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
__cuda_callable__
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
getRowFast( const IndexType row,
            IndexType* columns,
            RealType* values ) const
{

}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
   template< typename Vector >
__cuda_callable__
typename Vector::RealType
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
rowVectorProduct( const IndexType row,
                  const Vector& vector ) const
{

}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
template< typename InVector,
       typename OutVector >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
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
   auto fetch = [=] __cuda_callable__ ( IndexType row, IndexType offset, bool& compute ) -> RealType {
      const IndexType column = columnIndexesView[ offset ];
      compute = ( column != paddingIndex );
      if( ! compute )
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
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
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
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
allRowsReduction( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const
{
   this->rowsReduction( 0, this->getRows(), fetch, reduce, keep, zero );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
   template< typename Function >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
forRows( IndexType first, IndexType last, Function& function ) const
{
   const auto columns_view = this->columnIndexes.getConstView();
   const auto values_view = this->values.getConstView();
   const IndexType paddingIndex_ = this->getPaddingIndex();
   auto f = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable -> bool {
      function( rowIdx, localIdx, globalIdx );
      return true;
   };
   this->segments.forSegments( first, last, f );

}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
   template< typename Function >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
forAllRows( Function& function ) const
{
   this->forRows( 0, this->getRows(), function );
}

/*template< typename Real,
          template< typename, typename > class SegmentsView,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Real2, template< typename, typename > class Segments2, typename Index2, typename RealAllocator2, typename IndexAllocator2 >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
addMatrix( const SparseMatrixView< Real2, Segments2, Device, Index2, RealAllocator2, IndexAllocator2 >& matrix,
           const RealType& matrixMultiplicator,
           const RealType& thisMatrixMultiplicator )
{

}

template< typename Real,
          template< typename, typename > class SegmentsView,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Real2, typename Index2 >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
getTransposition( const SparseMatrixView< Real2, Device, Index2 >& matrix,
                  const RealType& matrixMultiplicator )
{

}*/

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
template< typename Vector1, typename Vector2 >
bool
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
performSORIteration( const Vector1& b,
                     const IndexType row,
                     Vector2& x,
                     const RealType& omega ) const
{

}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
save( File& file ) const
{
   MatrixView< RealType, DeviceType, IndexType >::save( file );
   file << this->columnIndexes;
   this->segments.save( file );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
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
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView >
__cuda_callable__
Index
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView >::
getPaddingIndex() const
{
   return -1;
}

   } //namespace Matrices
} // namespace  TNL
