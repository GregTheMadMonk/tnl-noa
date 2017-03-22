/***************************************************************************
                          tnlSlicedSlicedEllpackGraphMatrix_impl.h  -  description
                             -------------------
    begin                : Dec 8, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLSLICEDELLPACKGRAPHMATRIX_IMPL_H_
#define TNLSLICEDELLPACKGRAPHMATRIX_IMPL_H_

#include <matrices/tnlSlicedEllpackGraphMatrix.h>
#include <core/vectors/tnlVector.h>
#include <core/mfuncs.h>

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::tnlSlicedEllpackGraphMatrix()
: rearranged( false )
{
};

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
tnlString tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::getType()
{
   return tnlString( "tnlSlicedEllpackGraphMatrix< ") +
          tnlString( ::getType< Real >() ) +
          tnlString( ", " ) +
          Device :: getDeviceType() +
          tnlString( " >" );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
tnlString tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::setDimensions( const IndexType rows,
                                                                                   const IndexType columns )
{
   tnlAssert( rows > 0 && columns > 0,
              cerr << "rows = " << rows
                   << " columns = " << columns << endl );
   return tnlSparseMatrix< Real, Device, Index >::setDimensions( rows, columns );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::setRowLengths( const RowLengthsVector& rowLengths )
{
   tnlAssert( this->getRows() > 0, );
   tnlAssert( this->getColumns() > 0, );
   const IndexType slices = roundUpDivision( this->rows, SliceSize );
   if( ! this->sliceRowLengths.setSize( slices ) ||
       ! this->slicePointers.setSize( slices + 1 ) )
      return false;

   this->permutationArray.setSize( this->getRows() );
   for( IndexType i = 0; i < this->getRows(); i++ )
      this->permutationArray.setElement( i, i );

   tnlVector< Index, Device, Index > sliceRowLengths, slicePointers;
   sliceRowLengths.setSize( slices );
   slicePointers.setSize( slices + 1 );
   DeviceDependentCode::computeMaximalRowLengthInSlices( *this, rowLengths, sliceRowLengths, slicePointers );
   this->sliceRowLengths = sliceRowLengths;
   this->slicePointers = slicePointers;

   this->maxRowLength = rowLengths.max();

   this->slicePointers.computeExclusivePrefixSum();
   return this->allocateMatrixElements( this->slicePointers.getElement( slices ) );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
Index tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::getRowLength( const IndexType row ) const
{
   const IndexType slice = row / SliceSize;
   return this->sliceRowLengths[ slice ];
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::setLike( const tnlSlicedEllpackGraphMatrix< Real2, Device2, Index2, SliceSize >& matrix )
{
   if( !tnlSparseMatrix< Real, Device, Index >::setLike( matrix ) ||
       ! this->slicePointers.setLike( matrix.slicePointers ) ||
       ! this->sliceRowLengths.setLike( matrix.sliceRowLengths ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::reset()
{
   tnlSparseMatrix< Real, Device, Index >::reset();
   this->slicePointers.reset();
   this->sliceRowLengths.reset();
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::operator == ( const tnlSlicedEllpackGraphMatrix< Real2, Device2, Index2 >& matrix ) const
{
   tnlAssert( this->getRows() == matrix.getRows() &&
              this->getColumns() == matrix.getColumns(),
              cerr << "this->getRows() = " << this->getRows()
                   << " matrix.getRows() = " << matrix.getRows()
                   << " this->getColumns() = " << this->getColumns()
                   << " matrix.getColumns() = " << matrix.getColumns()
                   << " this->getName() = " << this->getName()
                   << " matrix.getName() = " << matrix.getName() );
   // TODO: implement this
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::operator != ( const tnlSlicedEllpackGraphMatrix< Real2, Device2, Index2 >& matrix ) const
{
   return ! ( ( *this ) == matrix );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::setElementFast( const IndexType row,
                                                                                    const IndexType column,
                                                                                    const Real& value )
{
   return this->addElementFast( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::setElement( const IndexType row,
                                                                                const IndexType column,
                                                                                const Real& value )
{
   return this->addElement( row, column, value, 0.0 );
}


template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::addElementFast( const IndexType row,
                                                                                    const IndexType column,
                                                                                    const RealType& value,
                                                                                    const RealType& thisElementMultiplicator )
{
   tnlAssert( row >= 0 && row < this->rows &&
              column >= 0 && column <= this->rows,
              cerr << " row = " << row
                   << " column = " << column
                   << " this->rows = " << this->rows
                   << " this->columns = " << this-> columns );

   Index elementPtr, rowEnd, step;
   DeviceDependentCode::initRowTraverseFast( *this, this->permutationArray.getElement( row ), elementPtr, rowEnd, step );

   IndexType col;
   while( elementPtr < rowEnd &&
          ( col = this->columnIndexes.getElement( elementPtr ) ) < column &&
          col != this->getPaddingIndex() ) elementPtr += step;
   if( elementPtr == rowEnd )
      return false;
   if( col == column )
   {
      this->values.setElement( elementPtr, thisElementMultiplicator * this->values.getElement( elementPtr ) + value );
      return true;
   }
   if( col == this->getPaddingIndex() )
   {
      this->columnIndexes.setElement( elementPtr, column );
      this->values.setElement( elementPtr, value );
      return true;
   }
   IndexType j = rowEnd - step;
   while( j > elementPtr )
   {
      this->columnIndexes.setElement( j, this->columnIndexes.getElement( j - step ) );
      this->values.setElement( j, this->values.getElement( j - step ) );
      j -= step;
   }
   this->columnIndexes.setElement( elementPtr, column );
   this->values.setElement( elementPtr, value );
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::addElement( const IndexType row,
                                                                                const IndexType column,
                                                                                const RealType& value,
                                                                                const RealType& thisElementMultiplicator )
{
   tnlAssert( row >= 0 && row < this->rows &&
              column >= 0 && column <= this->rows,
              cerr << " row = " << row
                   << " column = " << column
                   << " this->rows = " << this->rows
                   << " this->columns = " << this-> columns );

   Index elementPtr, rowEnd, step;
   DeviceDependentCode::initRowTraverse( *this, this->permutationArray.getElement( row ), elementPtr, rowEnd, step );

   IndexType col;
   while( elementPtr < rowEnd &&
          ( col = this->columnIndexes.getElement( elementPtr ) ) < column &&
          col != this->getPaddingIndex() ) elementPtr += step;
   if( elementPtr == rowEnd )
      return false;
   if( col == column )
   {
      this->values.setElement( elementPtr, thisElementMultiplicator * this->values.getElement( elementPtr ) + value );
      return true;
   }
   if( col == this->getPaddingIndex() )
   {
      this->columnIndexes.setElement( elementPtr, column );
      this->values.setElement( elementPtr, value );
      return true;
   }
   IndexType j = rowEnd - step;
   while( j > elementPtr )
   {
      this->columnIndexes.setElement( j, this->columnIndexes.getElement( j - step ) );
      this->values.setElement( j, this->values.getElement( j - step ) );
      j -= step;
   }
   this->columnIndexes.setElement( elementPtr, column );
   this->values.setElement( elementPtr, value );
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize > :: setRowFast( const IndexType row,
                                                                                  const IndexType* columnIndexes,
                                                                                  const RealType* values,
                                                                                  const IndexType elements )
{
   const IndexType sliceIdx = this->permutationArray.getElement( row ) / SliceSize;
   const IndexType rowLength = this->sliceRowLengths[ sliceIdx ];
   if( elements > rowLength )
      return false;

   Index elementPointer, rowEnd, step;
   DeviceDependentCode::initRowTraverseFast( *this, this->permutationArray.getElement( row ), elementPointer, rowEnd, step );

   for( IndexType i = 0; i < elements; i++ )
   {
      const IndexType column = columnIndexes[ i ];
      if( column < 0 || column >= this->getColumns() )
         return false;
      this->columnIndexes[ elementPointer ] = columnIndexes[ i ];
      this->values[ elementPointer ] = values[ i ];
      elementPointer += step;
   }
   for( IndexType i = elements; i < rowLength; i++ )
   {
      this->columnIndexes[ elementPointer ] = this->getPaddingIndex();
      elementPointer += step;
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize > :: setRow( const IndexType row,
                                                                              const IndexType* columnIndexes,
                                                                              const RealType* values,
                                                                              const IndexType elements )
{
   const IndexType sliceIdx = this->permutationArray.getElement( row ) / SliceSize;
   const IndexType rowLength = this->sliceRowLengths.getElement( sliceIdx );
   if( elements > rowLength )
      return false;

   Index elementPointer, rowEnd, step;
   DeviceDependentCode::initRowTraverse( *this, this->permutationArray.getElement( row ), elementPointer, rowEnd, step );

   for( IndexType i = 0; i < elements; i++ )
   {
      const IndexType column = columnIndexes[ i ];
      if( column < 0 || column >= this->getColumns() )
         return false;
      this->columnIndexes.setElement( elementPointer, column );
      this->values.setElement( elementPointer, values[ i ] );
      elementPointer += step;
   }
   for( IndexType i = elements; i < rowLength; i++ )
   {
      this->columnIndexes.setElement( elementPointer, this->getPaddingIndex() );
      elementPointer += step;
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize > :: addRowFast( const IndexType row,
                                                                                  const IndexType* columns,
                                                                                  const RealType* values,
                                                                                  const IndexType numberOfElements,
                                                                                  const RealType& thisElementMultiplicator )
{
   // TODO: implement
   return false;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize > :: addRow( const IndexType row,
                                                                              const IndexType* columns,
                                                                              const RealType* values,
                                                                              const IndexType numberOfElements,
                                                                              const RealType& thisElementMultiplicator )
{
   // TODO: implement
   return false;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Real tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::getElementFast( const IndexType row,
                                                                                    const IndexType column ) const
{
   Index elementPtr, rowEnd, step;
   DeviceDependentCode::initRowTraverseFast( *this, row, elementPtr, rowEnd, step );

   IndexType col;
   while( elementPtr < rowEnd &&
          ( col = this->columnIndexes[ elementPtr ] ) < column &&
          col != this->getPaddingIndex() )
      elementPtr += step;
   if( elementPtr < rowEnd && col == column )
      return this->values[ elementPtr ];
   return 0.0;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
Real tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::getElement( const IndexType row,
                                                                                const IndexType column ) const
{
   if( row < column )
      return this->getElement( column, row );

   Index elementPtr, rowEnd, step;
   DeviceDependentCode::initRowTraverse( *this, row, elementPtr, rowEnd, step );

   IndexType col;
   while( elementPtr < rowEnd &&
          ( col = this->columnIndexes.getElement( elementPtr ) ) < column &&
          col != this->getPaddingIndex() )
      elementPtr += step;
   if( elementPtr < rowEnd && col == column )
      return this->values.getElement( elementPtr );
   return 0.0;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
void tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::getRowFast( const IndexType row,
                                                                                IndexType* columns,
                                                                                RealType* values ) const
{
   Index elementPtr, rowEnd, step, i( 0 );
   DeviceDependentCode::initRowTraverseFast( *this, this->permutationArray.getElement( row ), elementPtr, rowEnd, step );

   while( elementPtr < rowEnd )
   {
      columns[ i ] = this->columnIndexes[ elementPtr ];
      values[ i ] = this->values[ elementPtr ];
      elementPtr += step;
      i++;
   }
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::getRow( const IndexType row,
                                                                            IndexType* columns,
                                                                            RealType* values ) const
{
   Index elementPtr, rowEnd, step, i( 0 );
   DeviceDependentCode::initRowTraverse( *this, row, elementPtr, rowEnd, step );

   while( elementPtr < rowEnd )
   {
      columns[ i ] = this->columnIndexes.getElement( elementPtr );
      values[ i ] = this->values.getElement( elementPtr );
      elementPtr += step;
      i++;
   }
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
  template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
typename Vector::RealType tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::rowVectorProduct( const IndexType row,
                                                                                                           const Vector& vector ) const
{
   Real result = 0.0;
   Index elementPtr, rowEnd, step;
   DeviceDependentCode::initRowTraverseFast( *this, this->permutationArray.getElement( row ), elementPtr, rowEnd, step );

   IndexType column;
   while( elementPtr < rowEnd &&
          ( column = this->columnIndexes[ elementPtr ] ) < this->columns &&
          column != this->getPaddingIndex() )
   {
      result += this->values[ elementPtr ] * vector[ column ];
      elementPtr += step;
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename InVector,
             typename OutVector >
void tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::vectorProduct( const InVector& inVector,
                                                                                   OutVector& outVector ) const
{
   DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Real2,
             typename Index2 >
void tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::addMatrix( const tnlSlicedEllpackGraphMatrix< Real2, Device, Index2 >& matrix,
                                                                               const RealType& matrixMultiplicator,
                                                                               const RealType& thisMatrixMultiplicator )
{
   tnlAssert( false, cerr << "TODO: implement" );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Real2,
             typename Index2 >
void tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::getTransposition( const tnlSlicedEllpackGraphMatrix< Real2, Device, Index2 >& matrix,
                                                                                      const RealType& matrixMultiplicator )
{
   tnlAssert( false, cerr << "TODO: implement" );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Vector >
bool tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::performSORIteration( const Vector& b,
                                                                                         const IndexType row,
                                                                                         Vector& x,
                                                                                         const RealType& omega ) const
{
   tnlAssert( row >=0 && row < this->getRows(),
              cerr << "row = " << row
                   << " this->getRows() = " << this->getRows()
                   << " this->getName() = " << this->getName() << endl );

   RealType diagonalValue( 0.0 );
   RealType sum( 0.0 );

   const IndexType sliceIdx = this->permutationArray.getElement( row ) / SliceSize;
   const IndexType rowLength = this->sliceRowLengths[ sliceIdx ];
   IndexType elementPtr = this->slicePointers[ sliceIdx ] +
                          rowLength * ( this->permutationArray.getElement( row ) - sliceIdx * SliceSize );
   const IndexType rowEnd( elementPtr + rowLength );
   IndexType column;
   while( elementPtr < rowEnd && ( column = this->columnIndexes[ elementPtr ] ) < this->columns )
   {
      if( column == this->permutationArray.getElement( row ) )
         diagonalValue = this->values.getElement( elementPtr );
      else
         sum += this->values.getElement( this->permutationArray.getElement( row ) * this->diagonalsShift.getSize() + elementPtr ) * x. getElement( column );
      elementPtr++;
   }
   if( diagonalValue == ( Real ) 0.0 )
   {
      cerr << "There is zero on the diagonal in " << this->permutationArray.getElement( row ) << "-th row of thge matrix " << this->getName() << ". I cannot perform SOR iteration." << endl;
      return false;
   }
   x. setElement( this->permutationArray.getElement( row ), x[ this->permutationArray.getElement( row ) ] + omega / diagonalValue * ( b[ this->permutationArray.getElement( row ) ] - sum ) );
   return true;
}


template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::save( tnlFile& file ) const
{
   if( ! tnlSparseMatrix< Real, Device, Index >::save( file ) ||
       ! this->slicePointers.save( file ) ||
       ! this->sliceRowLengths.save( file ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::load( tnlFile& file )
{
   if( ! tnlSparseMatrix< Real, Device, Index >::load( file ) ||
       ! this->slicePointers.load( file ) ||
       ! this->sliceRowLengths.load( file ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::save( const tnlString& fileName ) const
{
   return tnlObject::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::load( const tnlString& fileName )
{
   return tnlObject::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::print( ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      const IndexType sliceIdx = this->permutationArray.getElement( row ) / SliceSize;
      const IndexType rowLength = this->sliceRowLengths.getElement( sliceIdx );
      IndexType elementPtr = this->slicePointers.getElement( sliceIdx ) +
                             rowLength * ( this->permutationArray.getElement( row ) - sliceIdx * SliceSize );
      const IndexType rowEnd( elementPtr + rowLength );
      while( elementPtr < rowEnd &&
             this->columnIndexes.getElement( elementPtr ) < this->columns &&
             this->columnIndexes.getElement( elementPtr ) != this->getPaddingIndex() )
      {
         const Index column = this->columnIndexes.getElement( elementPtr );
         str << " Col:" << column << "->" << this->values.getElement( elementPtr ) << "\t";
         elementPtr++;
      }
      str << endl;
   }
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::computePermutationArray()
{
    tnlVector< Index, Device, Index > colorsVector;
    colorsVector.setSize( this->getRows() );
    for( IndexType i = 0; i < this->getRows(); i++ )
    {
        colorsVector.setElement( i, 0 );
    }

    // compute colors for each row
    tnlMatrix< Real, Device, Index >::computeColorsVector( colorsVector );

    // init color pointers
    this->colorPointers.setSize( this->getNumberOfColors() + 1 );

    // compute permutation
    IndexType position = 0;
    for( IndexType color = 0; color < this->getNumberOfColors(); color++ )
    {
        this->colorPointers.setElement( color, position );
        for (IndexType i = 0; i < this->getRows(); i++)
            if ( colorsVector.getElement( i ) == color)
            {
                IndexType row1 = this->permutationArray.getElement( i );
                IndexType row2 = this->permutationArray.getElement( position );
                IndexType tmp = this->permutationArray.getElement( row1 );
                this->permutationArray.setElement( row1, this->permutationArray.getElement( row2 ) );
                this->permutationArray.setElement( row2, tmp );

                tmp = colorsVector.getElement( position );
                colorsVector.setElement( position, colorsVector.getElement( i ) );
                colorsVector.setElement( i, tmp );
                position++;
            }
    }

    this->colorPointers.setElement( this->getNumberOfColors(), this->getRows() );

    this->inversePermutationArray.setSize( this->getRows() );
    for( IndexType i = 0; i < this->getRows(); i++ )
        this->inversePermutationArray.setElement( this->permutationArray.getElement( i ), i );

    // destroy colors vector
    colorsVector.reset();
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
Index tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::getRealRowLength( const Index row )
{
   const Index sliceIdx = row / SliceSize;
   const Index slicePointer = this->slicePointers.getElement( sliceIdx );
   const Index rowLength = this->sliceRowLengths.getElement( sliceIdx );

   Index rowBegin = slicePointer + rowLength * ( row - sliceIdx * SliceSize );
   Index rowEnd = rowBegin + rowLength;
   Index step = 1;
   Index length = 0;
   for( Index i = rowBegin; i < rowEnd; i++ )
      if( this->columnIndexes.getElement( i ) != this->getPaddingIndex() )
         length++;
      else
         break;

   return length;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
tnlVector< Index, Device, Index > tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::getRealRowLengths()
{
   tnlVector< Index, Device, Index > rowLengths;
   rowLengths.setSize( this->getRows() );
   for( IndexType row = 0; row < this->getRows(); row++ )
      rowLengths.setElement( row, this->getRealRowLength( row ) );

   return rowLengths;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::rearrangeMatrix( bool verbose )
{
    this->computePermutationArray();

    // now based on new permutation array we need to recompute row lengths in slices
    const IndexType slices = roundUpDivision( this->rows, SliceSize );
    tnlVector< Index, Device, Index > sliceRowLengths, slicePointers, rowLengths;
    sliceRowLengths.setSize( slices );
    slicePointers.setSize( slices + 1 );
    rowLengths.setSize( this->getRows() );
    rowLengths = this->getRealRowLengths();
    DeviceDependentCode::computeMaximalRowLengthInSlices( *this, rowLengths, sliceRowLengths, slicePointers );

    slicePointers.computeExclusivePrefixSum();

    // this->testRowLengths( rowLengths, sliceRowLengths );

    // return this->allocateMatrixElements( this->slicePointers.getElement( slices ) );
    tnlVector< Real, Device, Index > valuesVector;
    tnlVector< Index, Device, Index > columnsVector;
    valuesVector.setSize( slicePointers.getElement( slices ) );
    columnsVector.setSize( slicePointers.getElement( slices ) );
    columnsVector.setValue( this->getPaddingIndex() );
    valuesVector.setValue( 0.0 );

    for( IndexType slice = 0; slice < slices; slice++ )
    {
        IndexType step = 1;
        IndexType slicePointerOrig = this->slicePointers.getElement( slice );
        IndexType rowLengthOrig = this->sliceRowLengths.getElement( slice );
        for( IndexType row = slice * SliceSize; row < (slice + 1) * SliceSize && row < this->getRows(); row++ )
        {
            IndexType rowBegin = slicePointerOrig + rowLengthOrig * ( row - slice * SliceSize );
            IndexType rowEnd = rowBegin + rowLengthOrig;
            IndexType elementPointer = rowBegin;

            IndexType sliceNew = this->permutationArray.getElement( row ) / SliceSize;
            IndexType slicePointerNew = slicePointers.getElement( sliceNew );
            IndexType rowLengthNew = sliceRowLengths.getElement( sliceNew );
            IndexType elementPointerNew = slicePointerNew + rowLengthNew * ( this->permutationArray.getElement( row ) - sliceNew * SliceSize );

            for( IndexType i = 0; i < rowLengthOrig; i++ )
            {
                if( this->columnIndexes.getElement( elementPointer ) != this->getPaddingIndex() )
                {
                    valuesVector.setElement(elementPointerNew, this->values.getElement(elementPointer));
                    columnsVector.setElement(elementPointerNew, this->columnIndexes.getElement(elementPointer));
                    elementPointer += step;
                }
                elementPointerNew += step;
            }
        }
    }

    // reset original matrix
    this->values.reset();
    this->columnIndexes.reset();
    this->slicePointers.reset();
    this->sliceRowLengths.reset();

    this->slicePointers.setSize( slicePointers.getSize() );
    this->sliceRowLengths.setSize( sliceRowLengths.getSize() );

    this->sliceRowLengths = sliceRowLengths;
    this->slicePointers = slicePointers;

    // deep copy new matrix
    this->values.setSize( valuesVector.getSize() );
    this->columnIndexes.setSize( columnsVector.getSize() );
    this->values = valuesVector;
    this->columnIndexes = columnsVector;

    // clear memory
    valuesVector.reset();
    columnsVector.reset();
    slicePointers.reset();
    sliceRowLengths.reset();

    this->rearranged = true;
    return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::help( bool verbose )
{
    if( !this->rearranged )
        this->rearrangeMatrix( verbose );
    return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
template< typename InVector,
          typename OutVector >
void tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::vectorProductHost( const InVector& inVector,
                                                                                       OutVector& outVector ) const
{
    // simulated cuda SPMV on CPU
    for( IndexType i = 0; i < this->getNumberOfColors(); i++ )
    {
        IndexType offset = this->colorPointers[ i ];
        IndexType stop = this->colorPointers[ i + 1 ];
        IndexType inSliceIdx = offset % SliceSize;
        IndexType sliceOffset = offset - inSliceIdx;
        IndexType length = this->colorPointers[ i + 1 ] - this->colorPointers[ i ] + inSliceIdx;
        IndexType cudaBlockSize = 256;
        IndexType blocks = roundUpDivision( length, cudaBlockSize );
        for( IndexType blockIdx = 0; blockIdx < blocks; blockIdx++ )
        {
            for( IndexType warpIdx = 0; warpIdx < 8; warpIdx++ )
            {
               IndexType warpSize = 32;
               for (IndexType threadIdx = 0; threadIdx < warpSize; threadIdx++) {
                  IndexType row = blockIdx * cudaBlockSize + warpIdx * warpSize + threadIdx + sliceOffset;
                  if (row >= stop || row < offset)
                     continue;
                  IndexType sliceIdx = row / SliceSize;
                  IndexType sliceLength = this->sliceRowLengths[sliceIdx];
                  IndexType begin = this->slicePointers[sliceIdx] + sliceLength * threadIdx;
                  IndexType rowMapping = this->inversePermutationArray.getElement(row);
                  for (IndexType elementPtr = begin; elementPtr < begin + sliceLength; elementPtr++) {
                     IndexType column = this->columnIndexes[elementPtr];
                     if (column == this->getPaddingIndex())
                        break;
                     outVector[rowMapping] += inVector[column] * this->values[elementPtr];
                     if (rowMapping != column)
                     {
                        outVector[column] += inVector[rowMapping] * this->values[elementPtr];
                     }
                  }
               }
            }
        }
    }
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
__device__ void tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >::computeMaximalRowLengthInSlicesCuda( const RowLengthsVector& rowLengths,
                                                                                                               const IndexType sliceIdx )
{
   Index rowIdx = sliceIdx * SliceSize;
   Index rowInSliceIdx( 0 );
   Index maxRowLength( 0 );
   if( rowIdx >= this->getRows() )
      return;
   while( rowInSliceIdx < SliceSize && rowIdx < this->getRows() )
   {
      maxRowLength = Max( maxRowLength, rowLengths[ rowIdx ] );
      rowIdx++;
      rowInSliceIdx++;
   }
   this->sliceRowLengths[ sliceIdx ] = maxRowLength;
   this->slicePointers[ sliceIdx ] = maxRowLength * SliceSize;
   if( threadIdx.x == 0 )
      this->slicePointers[ this->slicePointers.getSize() - 1 ] = 0;

}
#endif

template<>
class tnlSlicedEllpackGraphMatrixDeviceDependentCode< tnlHost >
{
   public:

      typedef tnlHost Device;

      template< typename Real,
                typename Index,
                int SliceSize >
      static void initRowTraverse( const tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >& matrix,
                                   const Index row,
                                   Index& rowBegin,
                                   Index& rowEnd,
                                   Index& step )
      {
         const Index sliceIdx = matrix.permutationArray.getElement( row ) / SliceSize;
         const Index slicePointer = matrix.slicePointers.getElement( sliceIdx );
         const Index rowLength = matrix.sliceRowLengths.getElement( sliceIdx );

         rowBegin = slicePointer + rowLength * ( matrix.permutationArray.getElement( row ) - sliceIdx * SliceSize );
         rowEnd = rowBegin + rowLength;
         step = 1;
      }

      template< typename Real,
                typename Index,
                int SliceSize >
      static void initRowTraverseFast( const tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >& matrix,
                                       const Index row,
                                       Index& rowBegin,
                                       Index& rowEnd,
                                       Index& step )
      {
         const Index sliceIdx = matrix.permutationArray.getElement( row ) / SliceSize;
         const Index slicePointer = matrix.slicePointers[ sliceIdx ];
         const Index rowLength = matrix.sliceRowLengths[ sliceIdx ];

         rowBegin = slicePointer + rowLength * ( matrix.permutationArray.getElement( row ) - sliceIdx * SliceSize );
         rowEnd = rowBegin + rowLength;
         step = 1;
      }


      template< typename Real,
                typename Index,
                int SliceSize >
      static bool computeMaximalRowLengthInSlices( tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >& matrix,
                                                   const typename tnlSlicedEllpackGraphMatrix< Real, Device, Index >::RowLengthsVector& rowLengths,
                                                   tnlVector< Index, Device, Index >& sliceRowLengths,
                                                   tnlVector< Index, Device, Index >& slicePointers )
      {
         /*Index row( 0 ), slice( 0 ), sliceRowLength( 0 );
         while( row < matrix.getRows() )
         {
            sliceRowLength = Max( rowLengths.getElement( matrix.permutationArray.getElement( row++ ) ), sliceRowLength );
            if( row % SliceSize == 0 )
            {
               sliceRowLengths.setElement( slice, sliceRowLength );
               slicePointers.setElement( slice++, sliceRowLength * SliceSize );
               sliceRowLength = 0;
            }
         }
         if( row % SliceSize != 0 )
         {
            sliceRowLengths.setElement( slice, sliceRowLength );
            slicePointers.setElement( slice++, sliceRowLength * SliceSize );
         }
         slicePointers.setElement( slicePointers.getSize() - 1, 0 );*/

         Index sliceRowLength( 0 );
         Index numberOSlices = roundUpDivision( matrix.getRows(), SliceSize );
         tnlVector< Index, Device, Index > rowMapToSlice;
         rowMapToSlice.setSize( SliceSize );
         for( Index slice = 0; slice < numberOSlices; slice++ )
         {
            rowMapToSlice.setValue( -1 );
            Index elementPtr = 0;
            for( Index row = 0; row < matrix.getRows() && elementPtr < SliceSize; row++ )
            {
               if( matrix.permutationArray.getElement( row ) >= slice * SliceSize &&
                   matrix.permutationArray.getElement( row ) < ( slice + 1 ) * SliceSize )
               {
                  rowMapToSlice.setElement( elementPtr, row );
                  elementPtr++;
               }
            }

            // TODO: pridej sem nejaky logger!

            Index i = 0;
            for( ; i < SliceSize; i++ )
               // sliceRowLength = Max( rowLengths.getElement( matrix.permutationArray.getElement( rowMapToSlice.getElement( row ) ) ), sliceRowLength );
            {
               if( rowMapToSlice.getElement( i ) < 0 )
                  break;
               sliceRowLength = Max( rowLengths.getElement( rowMapToSlice.getElement( i ) ), sliceRowLength );
            }
            if( i % SliceSize == 0 || rowMapToSlice.getElement( i ) < 0 )
            {
               sliceRowLengths.setElement( slice, sliceRowLength );
               slicePointers.setElement( slice, sliceRowLength * SliceSize );
               sliceRowLength = 0;
            }
         }
         slicePointers.setElement( slicePointers.getSize() - 1, 0 );
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector,
                int SliceSize >
      static void vectorProduct( const tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         matrix.vectorProductHost( inVector, outVector );
      }

};

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          int SliceSize >
__global__ void tnlSlicedEllpackGraphMatrix_computeMaximalRowLengthInSlices_CudaKernel( tnlSlicedEllpackMatrix< Real, tnlCuda, Index, SliceSize >* matrix,
                                                                                   const typename tnlSlicedEllpackGraphMatrix< Real, tnlCuda, Index, SliceSize >::RowLengthsVector* rowLengths,
                                                                                   int gridIdx )
{
   const Index sliceIdx = gridIdx * tnlCuda::getMaxGridSize() * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
   matrix->computeMaximalRowLengthInSlicesCuda( *rowLengths, sliceIdx );
}
#endif

template<>
class tnlSlicedEllpackGraphMatrixDeviceDependentCode< tnlCuda >
{
   public:

      typedef tnlCuda Device;

      template< typename Real,
                typename Index,
                int SliceSize >
      static void initRowTraverse( const tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >& matrix,
                                   const Index row,
                                   Index& rowBegin,
                                   Index& rowEnd,
                                   Index& step )
      {
         const Index sliceIdx = matrix.permutationArray.getElement( row ) / SliceSize;
         const Index slicePointer = matrix.slicePointers.getElement( sliceIdx );
         const Index rowLength = matrix.sliceRowLengths.getElement( sliceIdx );

         rowBegin = slicePointer + matrix.permutationArray.getElement( row ) - sliceIdx * SliceSize;
         rowEnd = rowBegin + rowLength * SliceSize;
         step = SliceSize;
      }

      template< typename Real,
                typename Index,
                int SliceSize >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      static void initRowTraverseFast( const tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >& matrix,
                                       const Index row,
                                       Index& rowBegin,
                                       Index& rowEnd,
                                       Index& step )
      {
         const Index sliceIdx = matrix.permutationArray.getElement( row ) / SliceSize;
         const Index slicePointer = matrix.slicePointers[ sliceIdx ];
         const Index rowLength = matrix.sliceRowLengths[ sliceIdx ];

         rowBegin = slicePointer + matrix.permutationArray.getElement( row ) - sliceIdx * SliceSize;
         rowEnd = rowBegin + rowLength * SliceSize;
         step = SliceSize;

      }

      template< typename Real,
                typename Index,
                int SliceSize >
      static bool computeMaximalRowLengthInSlices( tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >& matrix,
                                                   const typename tnlSlicedEllpackGraphMatrix< Real, Device, Index >::RowLengthsVector& rowLengths )
      {
#ifdef HAVE_CUDA
         typedef tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize > Matrix;
         typedef typename Matrix::RowLengthsVector RowLengthsVector;
         Matrix* kernel_matrix = tnlCuda::passToDevice( matrix );
         RowLengthsVector* kernel_rowLengths = tnlCuda::passToDevice( rowLengths );
         const Index numberOfSlices = roundUpDivision( matrix.getRows(), SliceSize );
         dim3 cudaBlockSize( 256 ), cudaGridSize( tnlCuda::getMaxGridSize() );
         const Index cudaBlocks = roundUpDivision( numberOfSlices, cudaBlockSize.x );
         const Index cudaGrids = roundUpDivision( cudaBlocks, tnlCuda::getMaxGridSize() );
         for( int gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
         {
            if( gridIdx == cudaGrids - 1 )
               cudaGridSize.x = cudaBlocks % tnlCuda::getMaxGridSize();
            tnlSlicedEllpackGraphMatrix_computeMaximalRowLengthInSlices_CudaKernel< Real, Index, SliceSize ><<< cudaGridSize, cudaBlockSize >>>
                                                                             ( kernel_matrix,
                                                                               kernel_rowLengths,
                                                                               gridIdx );
         }
         tnlCuda::freeFromDevice( kernel_matrix );
         tnlCuda::freeFromDevice( kernel_rowLengths );
         checkCudaDevice;
#endif
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector,
                int SliceSize >
      static void vectorProduct( const tnlSlicedEllpackGraphMatrix< Real, Device, Index, SliceSize >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         tnlMatrixVectorProductCuda( matrix, inVector, outVector );
      }

};



#endif /* TNLSLICEDELLPACKMATRIX_IMPL_H_ */
