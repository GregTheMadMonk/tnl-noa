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
   : Matrix< Real, Device, Index, RealAllocator >( realAllocator ), columnsVector( indexAllocator )
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
   : Matrix< Real, Device, Index, RealAllocator >( m ), columnsVector( m.columnsVector )
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
   : Matrix< Real, Device, Index, RealAllocator >( std::move( m ) ), columnsVector( std::move( m.columnsVector ) )
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
: Matrix< Real, Device, Index, RealAllocator >( rows, columns, realAllocator ), columnsVector( indexAllocator )
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
setDimensions( const IndexType rows,
               const IndexType columns )
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
setCompressedRowLengths( ConstCompressedRowLengthsVectorView rowLengths )
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
   
}

__cuda_callable__
template< typename Real,
          template< typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
getRowFast( const IndexType row,
            IndexType* columns,
            RealType* values ) const
{
   
}

/*template< typename Real,
          template< typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
MatrixRow 
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
getRow( const IndexType rowIndex )
{
   
}

template< typename Real,
          template< typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
ConstMatrixRow
SparseMatrix< Real, Segments, Device, Index, RealAllocator, IndexAllocator >::
getRow( const IndexType rowIndex ) const
{
   
}*/

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
               OutVector& outVector ) const
// TODO: add const RealType& multiplicator = 1.0 )
{
   
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
   
}


   } //namespace Matrices
} // namespace  TNL
