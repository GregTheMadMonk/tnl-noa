/***************************************************************************
                          SparseMatrix.hpp -  description
                             -------------------
    begin                : Nov 29, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <functional>
#include <sstream>
#include <TNL/Algorithms/Reduction.h>
#include <TNL/Matrices/SparseMatrix.h>

namespace TNL {
namespace Matrices {

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
SparseMatrix( const RealAllocatorType& realAllocator,
              const IndexAllocatorType& indexAllocator )
   : BaseType( realAllocator ), columnIndexes( indexAllocator )
{
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
SparseMatrix( const SparseMatrix& m )
   : Matrix< Real, Device, Index, RealAllocator >( m ), columnIndexes( m.columnIndexes )
{
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
SparseMatrix( const SparseMatrix&& m )
   : Matrix< Real, Device, Index, RealAllocator >( std::move( m ) ), columnIndexes( std::move( m.columnIndexes ) )
{
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
SparseMatrix( const IndexType rows,
              const IndexType columns,
              const RealAllocatorType& realAllocator,
              const IndexAllocatorType& indexAllocator )
: BaseType( rows, columns, realAllocator ), columnIndexes( indexAllocator )
{
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
SparseMatrix( const std::initializer_list< IndexType >& rowCapacities,
              const IndexType columns,
              const RealAllocatorType& realAllocator,
              const IndexAllocatorType& indexAllocator )
: BaseType( rowCapacities.size(), columns, realAllocator ), columnIndexes( indexAllocator )
{
   this->setCompressedRowLengths( RowsCapacitiesType( rowCapacities ) );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
SparseMatrix( const IndexType rows,
              const IndexType columns,
              const std::initializer_list< std::tuple< IndexType, IndexType, RealType > >& data,
              const RealAllocatorType& realAllocator,
              const IndexAllocatorType& indexAllocator )
: BaseType( rows, columns, realAllocator ), columnIndexes( indexAllocator )
{
   this->setElements( data );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename MapIndex,
             typename MapValue >
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
SparseMatrix( const IndexType rows,
              const IndexType columns,
              const std::map< std::pair< MapIndex, MapIndex > , MapValue >& map )
{
   this->setDimensions( rows, columns );
   this->setElements( map );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
auto
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
getView() const -> ViewType
{
   return ViewType( this->getRows(),
                    this->getColumns(),
                    const_cast< SparseMatrix* >( this )->getValues().getView(),  // TODO: remove const_cast
                    const_cast< SparseMatrix* >( this )->columnIndexes.getView(),
                    const_cast< SparseMatrix* >( this )->segments.getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
auto
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
getConstView() const -> ConstViewType
{
   return ConstViewType( this->getRows(),
                         this->getColumns(),
                         this->getValues().getConstView(),
                         this->columnIndexes.getConstView(),
                         this->segments.getConstView() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
String
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
getSerializationType()
{
   return String( "Matrices::SparseMatrix< " ) +
             TNL::getSerializationType< RealType >() + ", " +
             TNL::getSerializationType< SegmentsType >() + ", [any_device], " +
             TNL::getSerializationType< IndexType >() + ", [any_allocator] >";
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
String
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename RowsCapacitiesVector >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
setCompressedRowLengths( const RowsCapacitiesVector& rowsCapacities )
{
   TNL_ASSERT_EQ( rowsCapacities.getSize(), this->getRows(), "Number of matrix rows does not fit with rowLengths vector size." );
   using RowsCapacitiesVectorDevice = typename RowsCapacitiesVector::DeviceType;
   if( std::is_same< DeviceType, RowsCapacitiesVectorDevice >::value )
      this->segments.setSegmentsSizes( rowsCapacities );
   else
   {
      RowsCapacitiesType thisRowsCapacities;
      thisRowsCapacities = rowsCapacities;
      this->segments.setSegmentsSizes( thisRowsCapacities );
   }
   if( ! isBinary() )
   {
      this->values.setSize( this->segments.getStorageSize() );
      this->values = ( RealType ) 0;
   }
   this->columnIndexes.setSize( this->segments.getStorageSize() );
   this->columnIndexes = this->getPaddingIndex();
   this->view = this->getView();
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
setElements( const std::initializer_list< std::tuple< IndexType, IndexType, RealType > >& data )
{
   const auto& rows = this->getRows();
   const auto& columns = this->getColumns();
   Containers::Vector< IndexType, Devices::Host, IndexType > rowCapacities( rows, 0 );
   for( const auto& i : data )
   {
      if( std::get< 0 >( i ) >= rows )
      {
         std::stringstream s;
         s << "Wrong row index " << std::get< 0 >( i ) << " in an initializer list";
         throw std::logic_error( s.str() );
      }
      rowCapacities[ std::get< 0 >( i ) ]++;
   }
   SparseMatrix< Real, Devices::Host, Index, MatrixType, Segments > hostMatrix( rows, columns );
   hostMatrix.setCompressedRowLengths( rowCapacities );
   for( const auto& i : data )
   {
      if( std::get< 1 >( i ) >= columns )
      {
         std::stringstream s;
         s << "Wrong column index " << std::get< 1 >( i ) << " in an initializer list";
         throw std::logic_error( s.str() );
      }
      hostMatrix.setElement( std::get< 0 >( i ), std::get< 1 >( i ), std::get< 2 >( i ) );
   }
   ( *this ) = hostMatrix;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename MapIndex,
             typename MapValue >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
setElements( const std::map< std::pair< MapIndex, MapIndex > , MapValue >& map )
{
   Containers::Vector< IndexType, Devices::Host, IndexType > rowsCapacities( this->getRows(), 0 );
   for( auto element : map )
      rowsCapacities[ element.first.first ]++;
   if( !std::is_same< DeviceType, Devices::Host >::value )
   {
      SparseMatrix< Real, Devices::Host, Index, MatrixType, Segments > hostMatrix( this->getRows(), this->getColumns() );
      hostMatrix.setCompressedRowLengths( rowsCapacities );
      for( auto element : map )
         hostMatrix.setElement( element.first.first, element.first.second, element.second );
      *this = hostMatrix;
   }
   else
   {
      this->setCompressedRowLengths( rowsCapacities );
      for( auto element : map )
         this->setElement( element.first.first, element.first.second, element.second );
   }
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Vector >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
getCompressedRowLengths( Vector& rowLengths ) const
{
   this->view.getCompressedRowLengths( rowLengths );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Matrix_ >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
setLike( const Matrix_& matrix )
{
   BaseType::setLike( matrix );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
Index
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
getNumberOfNonzeroMatrixElements() const
{
   return this->view.getNumberOfNonzeroMatrixElements();
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
reset()
{
   BaseType::reset();
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__ auto
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
getRow( const IndexType& rowIdx ) const -> const RowView
{
   return this->view.getRow( rowIdx );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__ auto
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
getRow( const IndexType& rowIdx ) -> RowView
{
   return this->view.getRow( rowIdx );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
setElement( const IndexType row,
            const IndexType column,
            const RealType& value )
{
   this->view.setElement( row, column, value );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
addElement( const IndexType row,
            const IndexType column,
            const RealType& value,
            const RealType& thisElementMultiplicator )
{
   this->view.addElement( row, column, value, thisElementMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
Real
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
getElement( const IndexType row,
            const IndexType column ) const
{
   return this->view.getElement( row, column );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Vector >
__cuda_callable__
typename Vector::RealType
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
rowVectorProduct( const IndexType row,
                  const Vector& vector ) const
{
   this->view.rowVectorProduct( row, vector );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
template< typename InVector,
       typename OutVector >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
vectorProduct( const InVector& inVector,
               OutVector& outVector,
               const RealType& matrixMultiplicator,
               const RealType& outVectorMultiplicator ) const
{
   this->view.vectorProduct( inVector, outVector, matrixMultiplicator, outVectorMultiplicator );
   /*TNL_ASSERT_EQ( this->getColumns(), inVector.getSize(), "Matrix columns do not fit with input vector." );
   TNL_ASSERT_EQ( this->getRows(), outVector.getSize(), "Matrix rows do not fit with output vector." );

   const auto inVectorView = inVector.getConstView();
   auto outVectorView = outVector.getView();
   const auto valuesView = this->values.getConstView();
   const auto columnIndexesView = this->columnIndexes.getConstView();
   const IndexType paddingIndex = this->getPaddingIndex();
   auto fetch = [=] __cuda_callable__ ( IndexType row, IndexType localIdx, IndexType globalIdx, bool& compute ) -> RealType {
      const IndexType column = columnIndexesView[ globalIdx ];
      compute = ( column != paddingIndex );
      if( ! compute )
         return 0.0;
      return valuesView[ globalIdx ] * inVectorView[ column ];
   };
   auto reduction = [] __cuda_callable__ ( RealType& sum, const RealType& value ) {
      sum += value;
   };
   auto keeper = [=] __cuda_callable__ ( IndexType row, const RealType& value ) mutable {
      outVectorView[ row ] = value;
   };
   this->segments.segmentsReduction( 0, this->getRows(), fetch, reduction, keeper, ( RealType ) 0.0 );*/
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
rowsReduction( IndexType first, IndexType last, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchValue& zero ) const
{
   this->view.rowsReduction( first, last, fetch, reduce, keep, zero );
   /*const auto columns_view = this->columnIndexes.getConstView();
   const auto values_view = this->values.getConstView();
   const IndexType paddingIndex_ = this->getPaddingIndex();
   auto fetch_ = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType globalIdx, bool& compute ) mutable -> decltype( fetch( IndexType(), IndexType(), IndexType(), RealType() ) ) {
      IndexType columnIdx = columns_view[ globalIdx ];
      if( columnIdx != paddingIndex_ )
         return fetch( rowIdx, columnIdx, globalIdx, values_view[ globalIdx ] );
      return zero;
   };
   this->segments.segmentsReduction( first, last, fetch_, reduce, keep, zero );*/
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
allRowsReduction( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const
{
   this->rowsReduction( 0, this->getRows(), fetch, reduce, keep, zero );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
forRows( IndexType first, IndexType last, Function& function ) const
{
   this->view.forRows( first, last, function );
   /*const auto columns_view = this->columnIndexes.getConstView();
   const auto values_view = this->values.getConstView();
   const IndexType paddingIndex_ = this->getPaddingIndex();
   auto f = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable -> bool {
      function( rowIdx, localIdx, columns_view[ globalIdx ], values_view[ globalIdx ] );
      return true;
   };
   this->segments.forSegments( first, last, f );
    */
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
forRows( IndexType first, IndexType last, Function& function )
{
   this->view.forRows( first, last, function );
   /*auto columns_view = this->columnIndexes.getView();
   auto values_view = this->values.getView();
   const IndexType paddingIndex_ = this->getPaddingIndex();
   auto f = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable -> bool {
      function( rowIdx, localIdx, columns_view[ globalIdx ], values_view[ globalIdx ] );
      return true;
   };
   this->segments.forSegments( first, last, f );*/
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
forAllRows( Function& function ) const
{
   this->forRows( 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
forAllRows( Function& function )
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
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
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
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
getTransposition( const SparseMatrix< Real2, Device, Index2 >& matrix,
                  const RealType& matrixMultiplicator )
{

}*/

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Vector1, typename Vector2 >
bool
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
performSORIteration( const Vector1& b,
                     const IndexType row,
                     Vector2& x,
                     const RealType& omega ) const
{
   return false;
}

// copy assignment
template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >&
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
operator=( const SparseMatrix& matrix )
{
   Matrix< Real, Device, Index >::operator=( matrix );
   this->columnIndexes = matrix.columnIndexes;
   this->segments = matrix.segments;
   this->view = this->getView();
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Real_, typename Device_, typename Index_, bool RowMajorOrder, typename RealAllocator_ >
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >&
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
operator=( const Dense< Real_, Device_, Index_, RowMajorOrder, RealAllocator_ >& matrix )
{
   using RHSMatrix = Dense< Real_, Device_, Index_, RowMajorOrder, RealAllocator_ >;
   using RHSIndexType = typename RHSMatrix::IndexType;
   using RHSRealType = typename RHSMatrix::RealType;
   using RHSDeviceType = typename RHSMatrix::DeviceType;
   using RHSRealAllocatorType = typename RHSMatrix::RealAllocatorType;

   Containers::Vector< RHSIndexType, RHSDeviceType, RHSIndexType > rowLengths;
   matrix.getCompressedRowLengths( rowLengths );
   this->setLike( matrix );
   this->setCompressedRowLengths( rowLengths );
   Containers::Vector< IndexType, DeviceType, IndexType > rowLocalIndexes( matrix.getRows() );
   rowLocalIndexes = 0;

   // TODO: use getConstView when it works
   const auto matrixView = const_cast< RHSMatrix& >( matrix ).getView();
   const IndexType paddingIndex = this->getPaddingIndex();
   auto columns_view = this->columnIndexes.getView();
   auto values_view = this->values.getView();
   auto rowLocalIndexes_view = rowLocalIndexes.getView();
   columns_view = paddingIndex;

   if( std::is_same< DeviceType, RHSDeviceType >::value )
   {
      const auto segments_view = this->segments.getView();
      auto f = [=] __cuda_callable__ ( RHSIndexType rowIdx, RHSIndexType localIdx, RHSIndexType columnIdx, const RHSRealType& value, bool& compute ) mutable {
         if( value != 0.0 )
         {
            IndexType thisGlobalIdx = segments_view.getGlobalIndex( rowIdx, rowLocalIndexes_view[ rowIdx ]++ );
            columns_view[ thisGlobalIdx ] = columnIdx;
            if( ! isBinary() )
               values_view[ thisGlobalIdx ] = value;
         }
      };
      matrix.forAllRows( f );
   }
   else
   {
      const IndexType maxRowLength = matrix.getColumns();
      const IndexType bufferRowsCount( 128 );
      const size_t bufferSize = bufferRowsCount * maxRowLength;
      Containers::Vector< RHSRealType, RHSDeviceType, RHSIndexType, RHSRealAllocatorType > matrixValuesBuffer( bufferSize );
      Containers::Vector< RealType, DeviceType, IndexType, RealAllocatorType > thisValuesBuffer( bufferSize );
      Containers::Vector< IndexType, DeviceType, IndexType, IndexAllocatorType > thisColumnsBuffer( bufferSize );
      auto matrixValuesBuffer_view = matrixValuesBuffer.getView();
      auto thisValuesBuffer_view = thisValuesBuffer.getView();

      IndexType baseRow( 0 );
      const IndexType rowsCount = this->getRows();
      while( baseRow < rowsCount )
      {
         const IndexType lastRow = min( baseRow + bufferRowsCount, rowsCount );
         thisColumnsBuffer = paddingIndex;

         ////
         // Copy matrix elements into buffer
         auto f1 = [=] __cuda_callable__ ( RHSIndexType rowIdx, RHSIndexType localIdx, RHSIndexType columnIndex, const RHSRealType& value, bool& compute ) mutable {
            const IndexType bufferIdx = ( rowIdx - baseRow ) * maxRowLength + localIdx;
            matrixValuesBuffer_view[ bufferIdx ] = value;
         };
         matrix.forRows( baseRow, lastRow, f1 );

         ////
         // Copy the source matrix buffer to this matrix buffer
         thisValuesBuffer_view = matrixValuesBuffer_view;

         ////
         // Copy matrix elements from the buffer to the matrix and ignoring
         // zero matrix elements.
         const IndexType matrix_columns = this->getColumns();
         auto f2 = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType& columnIndex, RealType& value, bool& compute  ) mutable {
            RealType inValue( 0.0 );
            IndexType bufferIdx, column( rowLocalIndexes_view[ rowIdx ] );
            while( inValue == 0.0 && column < matrix_columns )
            {
               bufferIdx = ( rowIdx - baseRow ) * maxRowLength + column++;
               inValue = thisValuesBuffer_view[ bufferIdx ];
            }
            rowLocalIndexes_view[ rowIdx ] = column;
            if( inValue == 0.0 )
            {
               columnIndex = paddingIndex;
               value = 0.0;
            }
            else
            {
               columnIndex = column - 1;
               value = inValue;
            }
         };
         this->forRows( baseRow, lastRow, f2 );
         baseRow += bufferRowsCount;
      }
      //std::cerr << "This matrix = " << std::endl << *this << std::endl;
   }
   this->view = this->getView();
   return *this;

}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename RHSMatrix >
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >&
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
operator=( const RHSMatrix& matrix )
{
   using RHSIndexType = typename RHSMatrix::IndexType;
   using RHSRealType = typename RHSMatrix::RealType;
   using RHSDeviceType = typename RHSMatrix::DeviceType;
   using RHSRealAllocatorType = typename RHSMatrix::RealAllocatorType;

   Containers::Vector< RHSIndexType, RHSDeviceType, RHSIndexType > rowLengths;
   matrix.getCompressedRowLengths( rowLengths );
   this->setDimensions( matrix.getRows(), matrix.getColumns() );
   this->setCompressedRowLengths( rowLengths );
   Containers::Vector< IndexType, DeviceType, IndexType > rowLocalIndexes( matrix.getRows() );
   rowLocalIndexes = 0;

   // TODO: use getConstView when it works
   const auto matrixView = const_cast< RHSMatrix& >( matrix ).getView();
   const IndexType paddingIndex = this->getPaddingIndex();
   auto columns_view = this->columnIndexes.getView();
   auto values_view = this->values.getView();
   auto rowLocalIndexes_view = rowLocalIndexes.getView();
   columns_view = paddingIndex;

   if( std::is_same< DeviceType, RHSDeviceType >::value )
   {
      const auto segments_view = this->segments.getView();
      auto f = [=] __cuda_callable__ ( RHSIndexType rowIdx, RHSIndexType localIdx_, RHSIndexType columnIndex, const RHSRealType& value, bool& compute ) mutable {
         IndexType localIdx( rowLocalIndexes_view[ rowIdx ] );
         if( value != 0.0 && columnIndex != paddingIndex )
         {
            IndexType thisGlobalIdx = segments_view.getGlobalIndex( rowIdx, localIdx++ );
            columns_view[ thisGlobalIdx ] = columnIndex;
            if( ! isBinary() )
               values_view[ thisGlobalIdx ] = value;
            rowLocalIndexes_view[ rowIdx ] = localIdx;
         }
      };
      matrix.forAllRows( f );
   }
   else
   {
      const IndexType maxRowLength = max( rowLengths );
      const IndexType bufferRowsCount( 128 );
      const size_t bufferSize = bufferRowsCount * maxRowLength;
      Containers::Vector< RHSRealType, RHSDeviceType, RHSIndexType, RHSRealAllocatorType > matrixValuesBuffer( bufferSize );
      Containers::Vector< RHSIndexType, RHSDeviceType, RHSIndexType > matrixColumnsBuffer( bufferSize );
      Containers::Vector< RealType, DeviceType, IndexType, RealAllocatorType > thisValuesBuffer( bufferSize );
      Containers::Vector< IndexType, DeviceType, IndexType > thisColumnsBuffer( bufferSize );
      Containers::Vector< IndexType, DeviceType, IndexType > thisRowLengths;
      thisRowLengths = rowLengths;
      auto matrixValuesBuffer_view = matrixValuesBuffer.getView();
      auto matrixColumnsBuffer_view = matrixColumnsBuffer.getView();
      auto thisValuesBuffer_view = thisValuesBuffer.getView();
      auto thisColumnsBuffer_view = thisColumnsBuffer.getView();
      matrixValuesBuffer_view = 0.0;

      IndexType baseRow( 0 );
      const IndexType rowsCount = this->getRows();
      while( baseRow < rowsCount )
      {
         const IndexType lastRow = min( baseRow + bufferRowsCount, rowsCount );
         thisColumnsBuffer = paddingIndex;
         matrixColumnsBuffer_view = paddingIndex;

         ////
         // Copy matrix elements into buffer
         auto f1 = [=] __cuda_callable__ ( RHSIndexType rowIdx, RHSIndexType localIdx, RHSIndexType columnIndex, const RHSRealType& value, bool& compute ) mutable {
            if( columnIndex != paddingIndex )
            {
               const IndexType bufferIdx = ( rowIdx - baseRow ) * maxRowLength + localIdx;
               matrixColumnsBuffer_view[ bufferIdx ] = columnIndex;
               matrixValuesBuffer_view[ bufferIdx ] = value;
               //std::cerr << " <<<<< rowIdx = " << rowIdx << " localIdx = " << localIdx << " value = " << value << " bufferIdx = " << bufferIdx << std::endl;
            }
         };
         matrix.forRows( baseRow, lastRow, f1 );

         ////
         // Copy the source matrix buffer to this matrix buffer
         thisValuesBuffer_view = matrixValuesBuffer_view;
         thisColumnsBuffer_view = matrixColumnsBuffer_view;

         ////
         // Copy matrix elements from the buffer to the matrix and ignoring
         // zero matrix elements
         const IndexType matrix_columns = this->getColumns();
         const auto thisRowLengths_view = thisRowLengths.getConstView();
         auto f2 = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType& columnIndex, RealType& value, bool& compute ) mutable {
            RealType inValue( 0.0 );
            size_t bufferIdx;
            IndexType bufferLocalIdx( rowLocalIndexes_view[ rowIdx ] );
            while( inValue == 0.0 && localIdx < thisRowLengths_view[ rowIdx ] )
            {
               bufferIdx = ( rowIdx - baseRow ) * maxRowLength + bufferLocalIdx++;
               TNL_ASSERT_LT( bufferIdx, bufferSize, "" );
               inValue = thisValuesBuffer_view[ bufferIdx ];
            }
            //std::cerr << "rowIdx = " << rowIdx << " localIdx = " << localIdx << " bufferLocalIdx = " << bufferLocalIdx
            //          << " inValue = " << inValue << " bufferIdx = " << bufferIdx << std::endl;
            rowLocalIndexes_view[ rowIdx ] = bufferLocalIdx;
            if( inValue == 0.0 )
            {
               columnIndex = paddingIndex;
               value = 0.0;
            }
            else
            {
               columnIndex = thisColumnsBuffer_view[ bufferIdx ];//column - 1;
               value = inValue;
            }
         };
         this->forRows( baseRow, lastRow, f2 );
         baseRow += bufferRowsCount;
      }
   }
   this->view = this->getView();
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
save( File& file ) const
{
   this->view.save( file );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
load( File& file )
{
   Matrix< RealType, DeviceType, IndexType >::load( file );
   file >> this->columnIndexes;
   this->segments.load( file );
   this->view = this->getView();
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
print( std::ostream& str ) const
{
   this->view.print( str );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
Index
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
getPaddingIndex() const
{
   return -1;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
auto
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
getSegments() -> SegmentsType&
{
   return this->segments;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename RealAllocator,
          typename IndexAllocator >
auto
SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::
getSegments() const -> const SegmentsType&
{
   return this->segments;
}

   } //namespace Matrices
} // namespace  TNL
