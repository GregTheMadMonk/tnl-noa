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
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
SparseMatrix( const RealAllocatorType& realAllocator,
              const IndexAllocatorType& indexAllocator )
: BaseType( realAllocator ), columnIndexes( indexAllocator ), view( this->getView() )
{
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Index_t, std::enable_if_t< std::is_integral< Index_t >::value, int > >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
SparseMatrix( const Index_t rows,
              const Index_t columns,
              const RealAllocatorType& realAllocator,
              const IndexAllocatorType& indexAllocator )
: BaseType( rows, columns, realAllocator ), columnIndexes( indexAllocator ),
  segments( Containers::Vector< IndexType, DeviceType, IndexType >( rows, 0 ) ),
  view( this->getView() )
{
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename ListIndex >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
SparseMatrix( const std::initializer_list< ListIndex >& rowCapacities,
              const IndexType columns,
              const RealAllocatorType& realAllocator,
              const IndexAllocatorType& indexAllocator )
: BaseType( rowCapacities.size(), columns, realAllocator ), columnIndexes( indexAllocator )
{
   this->setRowCapacities( RowsCapacitiesType( rowCapacities ) );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename RowCapacitiesVector, std::enable_if_t< TNL::IsArrayType< RowCapacitiesVector >::value, int > >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
SparseMatrix( const RowCapacitiesVector& rowCapacities,
              const IndexType columns,
              const RealAllocatorType& realAllocator,
              const IndexAllocatorType& indexAllocator )
: BaseType( rowCapacities.getSize(), columns, realAllocator ), columnIndexes( indexAllocator )
{
   this->setRowCapacities( rowCapacities );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
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
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename MapIndex,
             typename MapValue >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
SparseMatrix( const IndexType rows,
              const IndexType columns,
              const std::map< std::pair< MapIndex, MapIndex > , MapValue >& map,
              const RealAllocatorType& realAllocator,
              const IndexAllocatorType& indexAllocator )
: BaseType( rows, columns, realAllocator ), columnIndexes( indexAllocator )
{
   this->setDimensions( rows, columns );
   this->setElements( map );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
auto
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
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
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
auto
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
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
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
String
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
getSerializationType()
{
   return ViewType::getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
String
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
setDimensions( const IndexType rows,
               const IndexType columns )
{
   BaseType::setDimensions( rows, columns );
   segments.setSegmentsSizes( Containers::Vector< IndexType, DeviceType, IndexType >( rows, 0 ) );
   this->view = this->getView();
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Matrix_ >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
setLike( const Matrix_& matrix )
{
   BaseType::setLike( matrix );
   this->segments.setSegmentsSizes( Containers::Vector< IndexType, DeviceType, IndexType >( matrix.getRows(), 0 ) ),
   this->view = this->getView();
   TNL_ASSERT_EQ( this->getRows(), segments.getSegmentsCount(), "mismatched segments count" );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename RowsCapacitiesVector >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
setRowCapacities( const RowsCapacitiesVector& rowsCapacities )
{
   TNL_ASSERT_EQ( rowsCapacities.getSize(), this->getRows(), "Number of matrix rows does not fit with rowCapacities vector size." );
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
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Vector >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
getRowCapacities( Vector& rowCapacities ) const
{
   this->view.getRowCapacities( rowCapacities );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
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
   hostMatrix.setRowCapacities( rowCapacities );
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
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename MapIndex,
             typename MapValue >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
setElements( const std::map< std::pair< MapIndex, MapIndex > , MapValue >& map )
{
   Containers::Vector< IndexType, Devices::Host, IndexType > rowsCapacities( this->getRows(), 0 );
   for( auto element : map )
      rowsCapacities[ element.first.first ]++;
   if( !std::is_same< DeviceType, Devices::Host >::value )
   {
      SparseMatrix< Real, Devices::Host, Index, MatrixType, Segments > hostMatrix( this->getRows(), this->getColumns() );
      hostMatrix.setRowCapacities( rowsCapacities );
      for( auto element : map )
         hostMatrix.setElement( element.first.first, element.first.second, element.second );
      *this = hostMatrix;
   }
   else
   {
      this->setRowCapacities( rowsCapacities );
      for( auto element : map )
         this->setElement( element.first.first, element.first.second, element.second );
   }
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Vector >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
getCompressedRowLengths( Vector& rowLengths ) const
{
   this->view.getCompressedRowLengths( rowLengths );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
Index
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
getRowCapacity( const IndexType row ) const
{
   return this->view.getRowCapacity( row );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
Index
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
getNonzeroElementsCount() const
{
   return this->view.getNonzeroElementsCount();
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
reset()
{
   BaseType::reset();
   this->segments.reset();
   this->view = this->getView();
   TNL_ASSERT_EQ( this->getRows(), segments.getSegmentsCount(), "mismatched segments count" );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__ auto
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
getRow( const IndexType& rowIdx ) const -> const ConstRowView
{
   return this->view.getRow( rowIdx );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__ auto
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
getRow( const IndexType& rowIdx ) -> RowView
{
   return this->view.getRow( rowIdx );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__ void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
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
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__ void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
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
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
Real
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
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
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
template< typename InVector,
       typename OutVector >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
vectorProduct( const InVector& inVector,
               OutVector& outVector,
               const ComputeRealType& matrixMultiplicator,
               const ComputeRealType& outVectorMultiplicator,
               const IndexType firstRow,
               const IndexType lastRow ) const
{
   this->view.vectorProduct( inVector, outVector, matrixMultiplicator, outVectorMultiplicator, firstRow, lastRow );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
rowsReduction( IndexType begin, IndexType end, Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchValue& zero )
{
   this->view.rowsReduction( begin, end, fetch, reduce, keep, zero );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
rowsReduction( IndexType begin, IndexType end, Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchValue& zero ) const
{
   this->view.rowsReduction( begin, end, fetch, reduce, keep, zero );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
allRowsReduction( Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& zero )
{
   this->rowsReduction( 0, this->getRows(), fetch, reduce, keep, zero );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
allRowsReduction( Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& zero ) const
{
   this->rowsReduction( 0, this->getRows(), fetch, reduce, keep, zero );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
forRows( IndexType begin, IndexType end, Function& function ) const
{
   this->view.forRows( begin, end, function );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
forRows( IndexType begin, IndexType end, Function& function )
{
   this->view.forRows( begin, end, function );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
forAllRows( Function& function ) const
{
   this->forRows( 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
forAllRows( Function& function )
{
   this->forRows( 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
sequentialForRows( IndexType begin, IndexType end, Function& function ) const
{
   this->view.sequentialForRows( begin, end, function );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
sequentialForRows( IndexType first, IndexType last, Function& function )
{
   this->view.sequentialForRows( first, last, function );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
sequentialForAllRows( Function& function ) const
{
   this->sequentialForRows( 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
sequentialForAllRows( Function& function )
{
   this->sequentialForRows( 0, this->getRows(), function );
}


/*template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Real2, template< typename, typename > class Segments2, typename Index2, typename RealAllocator2, typename IndexAllocator2 >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
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
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
getTransposition( const SparseMatrix< Real2, Device, Index2 >& matrix,
                  const RealType& matrixMultiplicator )
{

}*/

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Vector1, typename Vector2 >
bool
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
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
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >&
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
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
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization, typename RealAllocator_ >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >&
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
operator=( const DenseMatrix< Real_, Device_, Index_, Organization, RealAllocator_ >& matrix )
{
   using RHSMatrix = DenseMatrix< Real_, Device_, Index_, Organization, RealAllocator_ >;
   using RHSIndexType = typename RHSMatrix::IndexType;
   using RHSRealType = typename RHSMatrix::RealType;
   using RHSDeviceType = typename RHSMatrix::DeviceType;
   using RHSRealAllocatorType = typename RHSMatrix::RealAllocatorType;

   Containers::Vector< RHSIndexType, RHSDeviceType, RHSIndexType > rowLengths;
   matrix.getCompressedRowLengths( rowLengths );
   this->setLike( matrix );
   this->setRowCapacities( rowLengths );
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
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename RHSMatrix >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >&
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
operator=( const RHSMatrix& matrix )
{
   using RHSIndexType = typename RHSMatrix::IndexType;
   using RHSRealType = typename RHSMatrix::RealType;
   using RHSDeviceType = typename RHSMatrix::DeviceType;
   using RHSRealAllocatorType = typename RHSMatrix::RealAllocatorType;

   Containers::Vector< RHSIndexType, RHSDeviceType, RHSIndexType > rowCapacities;
   matrix.getRowCapacities( rowCapacities );
   this->setDimensions( matrix.getRows(), matrix.getColumns() );
   this->setRowCapacities( rowCapacities );
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
      const IndexType maxRowLength = max( rowCapacities );
      const IndexType bufferRowsCount( 128 );
      const size_t bufferSize = bufferRowsCount * maxRowLength;
      Containers::Vector< RHSRealType, RHSDeviceType, RHSIndexType, RHSRealAllocatorType > matrixValuesBuffer( bufferSize );
      Containers::Vector< RHSIndexType, RHSDeviceType, RHSIndexType > matrixColumnsBuffer( bufferSize );
      Containers::Vector< RealType, DeviceType, IndexType, RealAllocatorType > thisValuesBuffer( bufferSize );
      Containers::Vector< IndexType, DeviceType, IndexType > thisColumnsBuffer( bufferSize );
      Containers::Vector< IndexType, DeviceType, IndexType > thisRowLengths;
      Containers::Vector< RHSIndexType, RHSDeviceType, RHSIndexType > rhsRowLengths;
      matrix.getCompressedRowLengths( rhsRowLengths );
      thisRowLengths= rhsRowLengths;
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
               //printf("SparseMatrix.hpp: localIdx = %d, maxRowLength = %d \n", localIdx, maxRowLength );
               TNL_ASSERT_LT( rowIdx - baseRow, bufferRowsCount, "" );
               TNL_ASSERT_LT( localIdx, maxRowLength, "" );
               const IndexType bufferIdx = ( rowIdx - baseRow ) * maxRowLength + localIdx;
               TNL_ASSERT_LT( bufferIdx, bufferSize, "" );
               matrixColumnsBuffer_view[ bufferIdx ] = columnIndex;
               matrixValuesBuffer_view[ bufferIdx ] = value;
               //printf( "TO BUFFER: rowIdx = %d localIdx = %d bufferIdx = %d column = %d value = %d \n", rowIdx, localIdx, bufferIdx, columnIndex, value );
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
         //const IndexType matrix_columns = this->getColumns();
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
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Matrix >
bool
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
operator==( const Matrix& m ) const
{
   return view == m;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Matrix >
bool
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
operator!=( const Matrix& m ) const
{
   return view != m;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
save( File& file ) const
{
   this->view.save( file );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
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
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
print( std::ostream& str ) const
{
   this->view.print( str );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
Index
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
getPaddingIndex() const
{
   return -1;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
auto
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
getSegments() -> SegmentsType&
{
   return this->segments;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
auto
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
getSegments() const -> const SegmentsType&
{
   return this->segments;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
auto
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
getColumnIndexes() const -> const ColumnsIndexesVectorType&
{
   return this->columnIndexes;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
auto
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::
getColumnIndexes() -> ColumnsIndexesVectorType&
{
   return this->columnIndexes;
}


} // namespace Matrices
} // namespace TNL
