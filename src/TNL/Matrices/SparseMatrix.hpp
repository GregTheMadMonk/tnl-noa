/***************************************************************************
                          SparseMatrix.h -  description
                             -------------------
    begin                : Nov 29, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/SparseMatrix.h>

namespace TNL {
namespace Matrices {

   template< typename Real,
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Real2, template< typename, typename > class Segments2,  typename Device2, typename Index2, typename RealAllocator2, typename IndexAllocator2 >
void
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
setLike( const SparseMatrix< Real2, Segments2, Device2, Index2, RealAllocator2, IndexAllocator2 >& matrix )
{
   Matrix< Real, Device, Index, RealAllocator >::setLike( matrix );
}

template< typename Real,
          template< typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
Index
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
getNumberOfNonzeroMatrixElements() const
{
}

template< typename Real,
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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

/*template< typename Real,
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >&
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
operator=( const SparseMatrix& matrix )
{

}

// cross-device copy assignment
template< typename Real,
          template< typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Real2,
             template< typename, typename > class Segments2,
             typename Device2,
             typename Index2,
             typename RealAllocator2,
             typename IndexAllocator2 >
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >&
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
operator=( const SparseMatrix< Real2, Segments2, Device2, Index2, RealAllocator2, IndexAllocator2 >& matrix )
{

}

template< typename Real,
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
          template< typename, typename > class Segments,
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
