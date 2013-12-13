/***************************************************************************
                          tnlChunkedEllpackMatrix_impl.h  -  description
                             -------------------
    begin                : Dec 12, 2013
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

#ifndef TNLCHUNKEDELLPACKMATRIX_IMPL_H_
#define TNLCHUNKEDELLPACKMATRIX_IMPL_H_


#include <matrices/tnlChunkedEllpackMatrix.h>
#include <core/vectors/tnlVector.h>
#include <core/mfuncs.h>

template< typename Real,
          typename Device,
          typename Index >
tnlChunkedEllpackMatrix< Real, Device, Index >::tnlChunkedEllpackMatrix()
: rows( 0 ),
  columns( 0 ),
  chunksInSlice( 256 ),
  desiredChunkSize( 16 )
{
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlChunkedEllpackMatrix< Real, Device, Index >::getType()
{
   return tnlString( "tnlChunkedEllpackMatrix< ") +
          tnlString( GetParameterType( Real( 0.0 ) ) ) +
          tnlString( ", " ) +
          Device :: getDeviceType() +
          tnlString( " >" );
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlChunkedEllpackMatrix< Real, Device, Index >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::setDimensions( const IndexType rows,
                                                                    const IndexType columns )
{
   tnlAssert( rows > 0 && columns > 0,
              cerr << "rows = " << rows
                   << " columns = " << columns << endl );
   this->rows = rows;
   this->columns = columns;

   /****
    * Allocate slice info array. Note that there cannot be
    * more slices than rows.
    */
   if( ! this->slices.setSize( rows ) ||
       ! this->chunksToRowsMapping.setSize( this-> rows ) ||
       ! this->slicesToRowsMapping.setSize( this->rows ) ||
       ! this->rowPointers.setSize( this->rows + 1 ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::setRowLengths( const Vector& rowLengths )
{
   /****
    * Iterate over rows and allocate slices so that each slice has
    * approximately the same number of allocated elements
    */

   IndexType row( 0 ),
             sliceIndex( 0 ),
             sliceBegin( 0 ),
             sliceEnd( 0 ),
             sliceSize( 0 ),
             allocatedElementsInSlice( 0 ),
             elementsToAllocation( 0 );
   const IndexType desiredElementsInSlice =
            this->chunksInSlice * this->desiredChunkSize;
   this->rowPointers[ 0 ] = 0;
   while( true )
   {
      /****
       * Add one row to the current slice until we reach the desired
       * number of elements in a slice.
       */
      allocatedElementsInSlice += rowLengths[ row ];
      if( allocatedElementsInSlice < desiredElementsInSlice )
      {
         row++;
         sliceSize++;
         if( row < this->rows && sliceSize < chunksInSlice ) continue;
      }

      /****
       * Now, compute the number of chunks per each row.
       * Each row get one chunk by default.
       * Then each row will get additional chunks w.r. to the
       * number of the elements in the row. If there are some
       * free chunks left, repeat it again.
       */
      IndexType freeChunks = this->chunksInSlice - sliceSize;
      const IndexType sliceBegin = row - sliceSize;
      const IndexType sliceEnd = row;
      for( IndexType i = sliceBegin; i < sliceEnd; i++ )
         this->chunksToRowsMapping.setElement( i, 1 );
      while( freeChunks )
      {
         IndexType allocatedChunks( 0 );
         for( IndexType i = sliceBegin; i < sliceEnd; i++ )
         {
            RealType rowRatio( 0.0 );
            if( allocatedElementsInSlice != 0 )
               rowRatio = ( RealType ) rowLengths[ i ] / ( RealType ) allocatedElementsInSlice;
            allocatedChunks += this->chunksToRowsMapping[ i ] = freeChunks * rowRatio;
         }
         freeChunks -= allocatedChunks;
         tnlAssert( allocatedChunks != 0, );
         tnlAssert( freeChunks >= 0, );
      }

      /****
       * Compute the chunk size
       */
      IndexType maxChunkInSlice( 0 );
      for( IndexType i = sliceBegin; i < sliceEnd; i++ )
         maxChunkInSlice = Max( maxChunkInSlice,
                             ceil( ( RealType ) rowLengths[ i ] /
                                   ( RealType ) this->chunksToRowsMapping[ i ] ) );

      /****
       * Set-up the slice info.
       */
      this->slices[ sliceIndex ].size = sliceSize;
      this->slices[ sliceIndex ].chunkSize = maxChunkInSlice;
      this->slices[ sliceIndex ].firstRow = sliceBegin;
      this->slices[ sliceIndex ].pointer = elementsToAllocation;
      elementsToAllocation += this->chunksInSlice * maxChunkInSlice;

      for( IndexType i = sliceBegin; i < sliceEnd; i++ )
         this->slicesToRowsMapping[ i ] = sliceIndex;
      sliceIndex++;

      for( IndexType i = sliceBegin; i < sliceEnd; i++ )
         this->rowPointers[ i + 1 ] =
            this->rowPointers[ i ] + maxChunkInSlice*chunksToRowsMapping[ i ];

      /****
       * Finish the chunks to rows mapping by computing the prefix sum.
       */
      this->chunksToRowsMapping.computePrefixSum( sliceBegin, sliceEnd );

      /****
       * Proceed to the next row
       */
      row++;
      sliceSize = 0;
      if( row < this->rows ) continue;
      else break;
   }

   /****
    * Allocate values and column indexes
    */
   if( ! this->values.setSize( elementsToAllocation ) ||
       ! this->columnIndexes.setSize( elementsToAllocation ) )
      return false;
   this->columnIndexes.setValue( this->columns );
   //this->numberOfSlices = sliceIndex;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::setLike( const tnlChunkedEllpackMatrix< Real2, Device2, Index2 >& matrix )
{
   if( ! this->setDimensions( matrix.getRows(), matrix.getColumns() ) ||
       ! this->values.setLike( matrix.values ) ||
       ! this->columnIndexes.setLike( matrix.columnIndexes ) ||
       ! this->chunksToRowsMapping.setLike( matrix.chunksToRowsMapping ) ||
       ! this->slicesToRowsMapping.setLike( matrix.slicesToRowsMapping ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlChunkedEllpackMatrix< Real, Device, Index >::getNumberOfAllocatedElements() const
{
   return this->values.getSize();
}

template< typename Real,
          typename Device,
          typename Index >
void tnlChunkedEllpackMatrix< Real, Device, Index >::reset()
{
   this->columns = 0;
   this->rows = 0;
   this->values.reset();
   this->columnIndexes.reset();
   this->chunksToRowsMapping.reset();
   this->slicesToRowsMapping.reset();
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlChunkedEllpackMatrix< Real, Device, Index >::getRows() const
{
   return this->rows;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlChunkedEllpackMatrix< Real, Device, Index >::getColumns() const
{
   return this->columns;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlChunkedEllpackMatrix< Real, Device, Index >::setNumberOfChunksInSlice( const IndexType chunksInSlice )
{
   this->chunksInSlice = chunksInSlice;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlChunkedEllpackMatrix< Real, Device, Index >::getNumberOfChunksInSlice() const
{
   return this->numberOfChunksInSlice;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlChunkedEllpackMatrix< Real, Device, Index >::setDesiredChunkSize( const IndexType desiredChunkSize )
{
   this->desiredChunkSize = desiredChunkSize;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlChunkedEllpackMatrix< Real, Device, Index >::getDesiredChunkSize() const
{
   return this->desiredChunkSize;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::operator == ( const tnlChunkedEllpackMatrix< Real2, Device2, Index2 >& matrix ) const
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
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::operator != ( const tnlChunkedEllpackMatrix< Real2, Device2, Index2 >& matrix ) const
{
   return ! ( ( *this ) == matrix );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::setElement( const IndexType row,
                                                                           const IndexType column,
                                                                           const Real& value )
{
   return this->addToElement( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlChunkedEllpackMatrix< Real, Device, Index >::getElement( const IndexType row,
                                                                             const IndexType column ) const
{
   const IndexType& sliceIndex = slicesToRowsMapping[ row ];
   const IndexType& chunkSize = slices[ sliceIndex ].chunkSize;
   IndexType elementPtr = rowPointers[ row ];
   const IndexType rowEnd = rowPointers[ row + 1 ];
   while( elementPtr < rowEnd && this->columnIndexes[ elementPtr ] < column )
      elementPtr++;
   if( this->columnIndexes[ elementPtr ] == column )
      return this->values[ elementPtr ];
   return 0.0;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::addToElement( const IndexType row,
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

   const IndexType& sliceIndex = slicesToRowsMapping[ row ];
   const IndexType& chunkSize = slices[ sliceIndex ].chunkSize;
   IndexType elementPtr = rowPointers[ row ];
   const IndexType rowEnd = rowPointers[ row + 1 ];

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
          typename Index >
   template< typename Vector >
typename Vector::RealType tnlChunkedEllpackMatrix< Real, Device, Index >::rowVectorProduct( const IndexType row,
                                                                                            const Vector& vector ) const
{
   tnlAssert( row >=0 && row < this->rows,
            cerr << " row = " << row << " this->rows = " << this->rows );

   const IndexType& sliceIndex = slicesToRowsMapping[ row ];
   const IndexType& chunkSize = slices[ sliceIndex ].chunkSize;
   IndexType elementPtr = rowPointers[ row ];
   const IndexType rowEnd = rowPointers[ row + 1 ];

   typename Vector::RealType result( 0.0 );
   while( elementPtr < rowEnd && this->columnIndexes[ elementPtr ] < this->columns )
   {
      const Index column = this->columnIndexes.getElement( elementPtr );
      result += this->values.getElement( elementPtr++ ) * vector.getElement( column );
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
void tnlChunkedEllpackMatrix< Real, Device, Index >::vectorProduct( const Vector& inVector,
                                                                    Vector& outVector ) const
{
   for( Index row = 0; row < this->getRows(); row ++ )
      outVector.setElement( row, this->rowVectorProduct( row, inVector ) );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Index2 >
void tnlChunkedEllpackMatrix< Real, Device, Index >::addMatrix( const tnlChunkedEllpackMatrix< Real2, Device, Index2 >& matrix,
                                                                          const RealType& matrixMultiplicator,
                                                                          const RealType& thisMatrixMultiplicator )
{
   tnlAssert( false, cerr << "TODO: implement" );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Index2 >
void tnlChunkedEllpackMatrix< Real, Device, Index >::getTransposition( const tnlChunkedEllpackMatrix< Real2, Device, Index2 >& matrix,
                                                                       const RealType& matrixMultiplicator )
{
   tnlAssert( false, cerr << "TODO: implement" );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::performSORIteration( const Vector& b,
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

   const IndexType& sliceIndex = slicesToRowsMapping[ row ];
   const IndexType& chunkSize = slices[ sliceIndex ].chunkSize;
   IndexType elementPtr = rowPointers[ row ];
   const IndexType rowEnd = rowPointers[ row + 1 ];
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
          typename Index >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::save( tnlFile& file ) const
{
   if( ! file.write( &this->rows ) ) return false;
   if( ! file.write( &this->columns ) ) return false;
   if( ! this->values.save( file ) ) return false;
   if( ! this->columnIndexes.save( file ) ) return false;
   if( ! this->chunksToRowsMapping.save( file ) ) return false;
   if( ! this->slicesToRowsMapping.save( file ) ) return false;
   if( ! this->rowPointers.save( file ) ) return false;
   if( ! this->slices.save( file ) ) return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::load( tnlFile& file )
{
   if( ! file.read( &this->rows ) ) return false;
   if( ! file.read( &this->columns ) ) return false;
   if( ! this->values.load( file ) ) return false;
   if( ! this->columnIndexes.load( file ) ) return false;
   if( ! this->chunksToRowsMapping.load( file ) ) return false;
   if( ! this->slicesToRowsMapping.load( file ) ) return false;
   if( ! this->rowPointers.load( file ) ) return false;
   if( ! this->slices.load( file ) ) return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::save( const tnlString& fileName ) const
{
   return tnlObject::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::load( const tnlString& fileName )
{
   return tnlObject::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlChunkedEllpackMatrix< Real, Device, Index >::print( ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";

      const IndexType& sliceIndex = slicesToRowsMapping[ row ];
      const IndexType& chunkSize = slices[ sliceIndex ].chunkSize;
      IndexType elementPtr = rowPointers[ row ];
      const IndexType rowEnd = rowPointers[ row + 1 ];

      while( elementPtr < rowEnd && this->columnIndexes[ elementPtr ] < this->columns )
      {
         const Index column = this->columnIndexes.getElement( elementPtr );
         str << " Col:" << column << "->" << this->values.getElement( elementPtr ) << "\t";
         elementPtr++;
      }
      str << endl;
   }
}


#endif /* TNLCHUNKEDELLPACKMATRIX_IMPL_H_ */
