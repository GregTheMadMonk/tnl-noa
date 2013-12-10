/***************************************************************************
                          tnlSlicedSlicedEllpackMatrix_impl.h  -  description
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

#ifndef TNLSLICEDELLPACKMATRIX_IMPL_H_
#define TNLSLICEDELLPACKMATRIX_IMPL_H_

#include <matrices/tnlSlicedEllpackMatrix.h>
#include <core/vectors/tnlVector.h>
#include <core/mfuncs.h>

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::tnlSlicedEllpackMatrix()
: rows( 0 ),
  columns( 0 )
{
};

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
tnlString tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::getType()
{
   return tnlString( "tnlSlicedEllpackMatrix< ") +
          tnlString( GetParameterType( Real( 0.0 ) ) ) +
          tnlString( ", " ) +
          Device :: getDeviceType() +
          tnlString( " >" );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
tnlString tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::setDimensions( const IndexType rows,
                                                                              const IndexType columns )
{
   tnlAssert( rows > 0 && columns > 0,
              cerr << "rows = " << rows
                   << " columns = " << columns << endl );
   this->rows = rows;
   this->columns = columns;
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Vector >
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::setRowLengths( const Vector& rowLengths )
{
   const IndexType slices = roundUpDivision( this->rows, SliceSize );
   if( ! this->sliceRowLengths.setSize( slices ) ||
       ! this->slicePointers.setSize( slices + 1 ) )
      return false;
   IndexType row( 0 ), slice( 0 ), sliceRowLength( 0 );

   /****
    * Compute maximal row length in each slice
    */
   while( row < this->rows )
   {
      sliceRowLength = Max( rowLengths.getElement( row++ ), sliceRowLength );
      if( row % SliceSize == 0 )
      {
         this->sliceRowLengths.setElement( slice, sliceRowLength );
         this->slicePointers.setElement( slice++, sliceRowLength*SliceSize );
         sliceRowLength = 0;
      }
   }
   if( row % SliceSize != 0 )
   {
      this->sliceRowLengths.setElement( slice, sliceRowLength );
      this->slicePointers.setElement( slice++, sliceRowLength*SliceSize );
   }

   /****
    * Compute the slice pointers using the exclusive prefix sum
    */
   this->slicePointers.setElement( slices, 0 );
   this->slicePointers.computeExclusivePrefixSum();

   /****
    * Allocate values and column indexes
    */
   if( ! this->values.setSize( this->slicePointers[ slices ] ) ||
       ! this->columnIndexes.setSize( this->slicePointers[ slices ] ) )
      return false;
   this->columnIndexes.setValue( this->columns );
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::setLike( const tnlSlicedEllpackMatrix< Real2, Device2, Index2, SliceSize >& matrix )
{
   if( ! this->setDimensions( matrix.getRows(), matrix.getColumns() ) ||
       ! this->values.setLike( matrix.values ) ||
       ! this->columnIndexes.setLike( matrix.columnIndexes ) ||
       ! this->slicePointers.setLike( matrix.slicePointers ) ||
       ! this->sliceRowLengths.setLike( matrix.sliceRowLengths ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
Index tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::getNumberOfAllocatedElements() const
{
   return this->values.getSize();
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::reset()
{
   this->columns = 0;
   this->rows = 0;
   this->values.reset();
   this->columnIndexes.reset();
   this->slicePointers.reset();
   this->sliceRowLengths.reset();
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
Index tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::getRows() const
{
   return this->rows;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
Index tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::getColumns() const
{
   return this->columns;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::operator == ( const tnlSlicedEllpackMatrix< Real2, Device2, Index2 >& matrix ) const
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
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::operator != ( const tnlSlicedEllpackMatrix< Real2, Device2, Index2 >& matrix ) const
{
   return ! ( ( *this ) == matrix );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::setElement( const IndexType row,
                                                                           const IndexType column,
                                                                           const Real& value )
{
   return this->addToElement( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
Real tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::getElement( const IndexType row,
                                                                           const IndexType column ) const
{
   const IndexType sliceIdx = row / SliceSize;
   const IndexType rowLength = this->sliceRowLengths[ sliceIdx ];
   IndexType elementPtr = this->slicePointers[ sliceIdx ] +
                          rowLength * ( row - sliceIdx * SliceSize );
   const IndexType rowEnd = elementPtr + rowLength;
   while( elementPtr < rowEnd && this->columnIndexes[ elementPtr ] < column )
      elementPtr++;
   if( this->columnIndexes[ elementPtr ] == column )
      return this->values[ elementPtr ];
   return 0.0;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::addToElement( const IndexType row,
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

   const IndexType sliceIdx = row / SliceSize;
   const IndexType rowLength = this->sliceRowLengths[ sliceIdx ];
   IndexType elementPtr = this->slicePointers[ sliceIdx ] +
                          rowLength * ( row - sliceIdx * SliceSize );
   const IndexType rowEnd( elementPtr + rowLength );
   while( elementPtr < rowEnd && this->columnIndexes[ elementPtr ] < column ) elementPtr++;
   if( elementPtr == rowEnd )
      return false;
   if( this->columnIndexes[ elementPtr ] == column )
   {
      this->values[ elementPtr ] = thisElementMultiplicator * this->values[ elementPtr ] + value;
      return true;
   }
   else
      if( this->columnIndexes[ elementPtr ] == this->columns )
      {
         this->columnIndexes[ elementPtr ] = column;
         this->values[ elementPtr ] = value;
         return true;
      }
      else
      {
         IndexType j = rowEnd - 1;
         while( j > elementPtr )
         {
            this->columnIndexes[ j ] = this->columnIndexes[ j - 1 ];
            this->values[ j ] = this->values[ j - 1 ];
            j--;
         }
         this->columnIndexes[ elementPtr ] = column;
         this->values[ elementPtr ] = value;
         return true;
      }
   return false;
}


template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Vector >
void tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::vectorProduct( const Vector& inVector,
                                                                              Vector& outVector ) const
{
   for( Index row = 0; row < this->getRows(); row ++ )
   {
      Real result = 0.0;
      const IndexType sliceIdx = row / SliceSize;
      const IndexType rowLength = this->sliceRowLengths[ sliceIdx ];
      IndexType elementPtr = this->slicePointers[ sliceIdx ] +
                             rowLength * ( row - sliceIdx * SliceSize );
      const IndexType rowEnd( elementPtr + rowLength );
      while( elementPtr < rowEnd && this->columnIndexes[ elementPtr ] < this->columns )
      {
         const Index column = this->columnIndexes.getElement( elementPtr );
         result += this->values.getElement( elementPtr++ ) * inVector.getElement( column );
      }
      outVector.setElement( row, result );
   }
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Real2,
             typename Index2 >
void tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::addMatrix( const tnlSlicedEllpackMatrix< Real2, Device, Index2 >& matrix,
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
void tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::getTransposition( const tnlSlicedEllpackMatrix< Real2, Device, Index2 >& matrix,
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
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::performSORIteration( const Vector& b,
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

   const IndexType sliceIdx = row / SliceSize;
   const IndexType rowLength = this->sliceRowLengths[ sliceIdx ];
   IndexType elementPtr = this->slicePointers[ sliceIdx ] +
                          rowLength * ( row - sliceIdx * SliceSize );
   const IndexType rowEnd( elementPtr + rowLength );
   IndexType column;
   while( elementPtr < rowEnd && ( column = this->columnIndexes[ elementPtr ] ) < this->columns )
   {
      if( column == row )
         diagonalValue = this->values.getElement( elementPtr );
      else
         sum += this->values.getElement( row * this->diagonalsShift.getSize() + elementPtr ) * x. getElement( column );
      elementPtr++;
   }
   if( diagonalValue == ( Real ) 0.0 )
   {
      cerr << "There is zero on the diagonal in " << row << "-th row of thge matrix " << this->getName() << ". I cannot perform SOR iteration." << endl;
      return false;
   }
   x. setElement( row, x[ row ] + omega / diagonalValue * ( b[ row ] - sum ) );
   return true;
}


template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::save( tnlFile& file ) const
{
   if( ! file.write( &this->rows ) ) return false;
   if( ! file.write( &this->columns ) ) return false;
   if( ! this->values.save( file ) ) return false;
   if( ! this->columnIndexes.save( file ) ) return false;
   if( ! this->slicePointers.save( file ) ) return false;
   if( ! this->sliceRowLengths.save( file ) ) return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::load( tnlFile& file )
{
   if( ! file.read( &this->rows ) ) return false;
   if( ! file.read( &this->columns ) ) return false;
   if( ! this->values.load( file ) ) return false;
   if( ! this->columnIndexes.load( file ) ) return false;
   if( ! this->slicePointers.load( file ) ) return false;
   if( ! this->sliceRowLengths.load( file ) ) return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::save( const tnlString& fileName ) const
{
   return tnlObject::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::load( const tnlString& fileName )
{
   return tnlObject::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::print( ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      const IndexType sliceIdx = row / SliceSize;
      const IndexType rowLength = this->sliceRowLengths[ sliceIdx ];
      IndexType elementPtr = this->slicePointers[ sliceIdx ] +
                             rowLength * ( row - sliceIdx * SliceSize );
      const IndexType rowEnd( elementPtr + rowLength );
      while( elementPtr < rowEnd && this->columnIndexes[ elementPtr ] < this->columns )
      {
         const Index column = this->columnIndexes.getElement( elementPtr );
         str << " Col:" << column << "->" << this->values.getElement( elementPtr ) << "\t";
         elementPtr++;
      }
      str << endl;
   }
}

#endif /* TNLSLICEDELLPACKMATRIX_IMPL_H_ */
