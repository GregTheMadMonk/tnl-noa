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
: chunksInSlice( 256 ),
  desiredChunkSize( 16 )
{
   this->values.setName( "tnlChunkedEllpackMatrix::values" );
   this->columnIndexes.setName( "tnlChunkedEllpackMatrix::columnIndexes" );
   chunksToRowsMapping.setName( "tnlChunkedEllpackMatrix::chunksToRowsMapping" );
   slicesToRowsMapping.setName( "tnlChunkedEllpackMatrix::slicesToRowsMapping" );
   rowPointers.setName( "tnlChunkedEllpackMatrix::rowPointers" );
   slices.setName( "tnlChunkedEllpackMatrix::slices" );
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
   if( ! tnlSparseMatrix< Real, Device, Index >::setDimensions( rows, columns ) )
      return false;

   /****
    * Allocate slice info array. Note that there cannot be
    * more slices than rows.
    */
   if( ! this->slices.setSize( this->rows ) ||
       ! this->chunksToRowsMapping.setSize( this-> rows ) ||
       ! this->slicesToRowsMapping.setSize( this->rows ) ||
       ! this->rowPointers.setSize( this->rows + 1 ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::setRowLengths( const RowLengthsVector& rowLengths )
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
      sliceSize++;
      row++;
      if( allocatedElementsInSlice < desiredElementsInSlice )
         if( row < this->rows && sliceSize < chunksInSlice ) continue;
      tnlAssert( sliceSize >0, );

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
         for( IndexType i = sliceBegin; i < sliceEnd && freeChunks > 0; i++ )
         {
            RealType rowRatio( 0.0 );
            if( allocatedElementsInSlice != 0 )
               rowRatio = ( RealType ) rowLengths[ i ] / ( RealType ) allocatedElementsInSlice;
            const IndexType addedChunks = ceil( freeChunks * rowRatio );
            freeChunks -= addedChunks;
            this->chunksToRowsMapping[ i ] += addedChunks;
            tnlAssert( chunksToRowsMapping[ i ] > 0,
                       cerr << " chunksToRowsMapping[ i ] = " << chunksToRowsMapping[ i ] << endl );
         }
         tnlAssert( freeChunks >= 0, );
         //cout << "freeChunks = " << freeChunks << endl;
      }

      /****
       * Compute the chunk size
       */
      IndexType maxChunkInSlice( 0 );
      for( IndexType i = sliceBegin; i < sliceEnd; i++ )
         maxChunkInSlice = Max( maxChunkInSlice,
                             ceil( ( RealType ) rowLengths[ i ] /
                                   ( RealType ) this->chunksToRowsMapping[ i ] ) );
      tnlAssert( maxChunkInSlice > 0,
                 cerr << " maxChunkInSlice = " << maxChunkInSlice << endl );


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
      {
         this->rowPointers[ i + 1 ] =
            this->rowPointers[ i ] + maxChunkInSlice*chunksToRowsMapping[ i ];
         tnlAssert( this->rowPointers[ i ] >= 0,
                    cerr << "this->rowPointers[ i ] = " << this->rowPointers[ i ] );
         tnlAssert( this->rowPointers[ i + 1 ] >= 0,
                    cerr << "this->rowPointers[ i + 1 ] = " << this->rowPointers[ i + 1 ] );
      }

      /****
       * Finish the chunks to rows mapping by computing the prefix sum.
       */
      this->chunksToRowsMapping.computePrefixSum( sliceBegin, sliceEnd );

      /****
       * Proceed to the next row
       */
      sliceSize = 0;
      allocatedElementsInSlice = 0;
      if( row < this->rows ) continue;
      else break;
   }

   return tnlSparseMatrix< Real, Device, Index >::allocateMatrixElements( elementsToAllocation );
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlChunkedEllpackMatrix< Real, Device, Index >::getRowLength( const IndexType row ) const
{
   const IndexType& sliceIndex = slicesToRowsMapping[ row ];
   tnlAssert( sliceIndex < this->rows, );
   const IndexType& chunkSize = slices.getElement( sliceIndex ).chunkSize;
   return rowPointers[ row + 1 ] - rowPointers[ row ];
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::setLike( const tnlChunkedEllpackMatrix< Real2, Device2, Index2 >& matrix )
{
   this->chunksInSlice = matrix.chunksInSlice;
   this->desiredChunkSize = matrix.desiredChunkSize;
   if( ! tnlSparseMatrix< Real, Device, Index >::setLike( matrix ) ||
       ! this->chunksToRowsMapping.setLike( matrix.chunksToRowsMapping ) ||
       ! this->slicesToRowsMapping.setLike( matrix.slicesToRowsMapping ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlChunkedEllpackMatrix< Real, Device, Index >::reset()
{
   tnlSparseMatrix< Real, Device, Index >::reset();
   this->chunksToRowsMapping.reset();
   this->slicesToRowsMapping.reset();
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
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlChunkedEllpackMatrix< Real, Device, Index >::setElementFast( const IndexType row,
                                                                     const IndexType column,
                                                                     const Real& value )
{
   return this->addElementFast( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::setElement( const IndexType row,
                                                                 const IndexType column,
                                                                 const Real& value )
{
   return this->addElement( row, column, value, 0.0 );
}


template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlChunkedEllpackMatrix< Real, Device, Index >::addElementFast( const IndexType row,
                                                                     const IndexType column,
                                                                     const RealType& value,
                                                                     const RealType& thisElementMultiplicator )
{
   // TODO: return this back when CUDA kernels support cerr
   /*tnlAssert( row >= 0 && row < this->rows &&
              column >= 0 && column <= this->rows,
              cerr << " row = " << row
                   << " column = " << column
                   << " this->rows = " << this->rows
                   << " this->columns = " << this-> columns );*/

   const IndexType& sliceIndex = slicesToRowsMapping[ row ];
   tnlAssert( sliceIndex < this->rows, );
   const IndexType& chunkSize = slices.getElement( sliceIndex ).chunkSize;
   IndexType elementPtr = rowPointers[ row ];
   const IndexType rowEnd = rowPointers[ row + 1 ];

   // TODO: return this back when CUDA kernels support cerr
   /*tnlAssert( elementPtr >= 0,
            cerr << "elementPtr = " << elementPtr );
   tnlAssert( rowEnd <= this->columnIndexes.getSize(),
            cerr << "rowEnd = " << rowEnd << " this->columnIndexes.getSize() = " << this->columnIndexes.getSize() );*/

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

bool tnlChunkedEllpackMatrix< Real, Device, Index >::addElement( const IndexType row,
                                                                 const IndexType column,
                                                                 const RealType& value,
                                                                 const RealType& thisElementMultiplicator )
{
   return this->addElementFast( row, column, value, thisElementMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlChunkedEllpackMatrix< Real, Device, Index > :: setRowFast( const IndexType row,
                                                                   const IndexType* columnIndexes,
                                                                   const RealType* values,
                                                                   const IndexType elements )
{
   const IndexType& sliceIndex = slicesToRowsMapping[ row ];
   tnlAssert( sliceIndex < this->rows, );
   const IndexType& chunkSize = slices.getElement( sliceIndex ).chunkSize;
   IndexType elementPointer = rowPointers[ row ];
   const IndexType rowEnd = rowPointers[ row + 1 ];
   const IndexType rowLength = rowEnd - elementPointer;
   if( elements >  rowLength )
      return false;

   for( IndexType i = 0; i < elements; i++ )
   {
      this->columnIndexes[ elementPointer ] = columnIndexes[ i ];
      this->values[ elementPointer ] = values[ i ];
      elementPointer++;
   }
   for( IndexType i = elements; i < rowLength; i++ )
      this->columnIndexes[ elementPointer++ ] = this->getColumns();
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlChunkedEllpackMatrix< Real, Device, Index > :: setRow( const IndexType row,
                                                               const IndexType* columnIndexes,
                                                               const RealType* values,
                                                               const IndexType elements )
{
   return this->setRowFast( row, columnIndexes, values, elements );
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlChunkedEllpackMatrix< Real, Device, Index > :: addRowFast( const IndexType row,
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
          typename Index >
bool tnlChunkedEllpackMatrix< Real, Device, Index > :: addRow( const IndexType row,
                                                               const IndexType* columns,
                                                               const RealType* values,
                                                               const IndexType numberOfElements,
                                                               const RealType& thisElementMultiplicator )
{
   return this->addRowFast( row, columns, values, numberOfElements, thisElementMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Real tnlChunkedEllpackMatrix< Real, Device, Index >::getElementFast( const IndexType row,
                                                                     const IndexType column ) const
{
   const IndexType& sliceIndex = slicesToRowsMapping[ row ];
   tnlAssert( sliceIndex < this->rows, );
   const IndexType& chunkSize = slices.getElement( sliceIndex ).chunkSize;
   IndexType elementPtr = rowPointers[ row ];
   const IndexType rowEnd = rowPointers[ row + 1 ];
   // TODO: return this back when CUDA kernels support cerr
   /*tnlAssert( rowEnd <= this->columnIndexes.getSize(),
            cerr << "rowEnd = " << rowEnd << " this->columnIndexes.getSize() = " << this->columnIndexes.getSize() );*/
   while( elementPtr < rowEnd && this->columnIndexes[ elementPtr ] < column )
      elementPtr++;
   if( elementPtr < rowEnd && this->columnIndexes[ elementPtr ] == column )
      return this->values[ elementPtr ];
   return 0.0;
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlChunkedEllpackMatrix< Real, Device, Index >::getElement( const IndexType row,
                                                                 const IndexType column ) const
{
   return this->getElementFast( row, column );
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
void tnlChunkedEllpackMatrix< Real, Device, Index >::getRowFast( const IndexType row,
                                                                 IndexType* columns,
                                                                 RealType* values ) const
{
   const IndexType& sliceIndex = slicesToRowsMapping[ row ];
   tnlAssert( sliceIndex < this->rows, );
   const IndexType& chunkSize = slices.getElement( sliceIndex ).chunkSize;
   IndexType elementPointer = rowPointers[ row ];
   const IndexType rowLength = rowPointers[ row + 1 ] - elementPointer;

   for( IndexType i = 0; i < rowLength; i++ )
   {
      columns[ i ] = this->columnIndexes[ elementPointer ];
      values[ i ] = this->values[ elementPointer ];
      elementPointer++;
   }
}

template< typename Real,
          typename Device,
          typename Index >
void tnlChunkedEllpackMatrix< Real, Device, Index >::getRow( const IndexType row,
                                                             IndexType* columns,
                                                             RealType* values ) const
{
   return this->getRowFast( row, columns, values );
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
   tnlAssert( sliceIndex < this->rows, );
   const IndexType& chunkSize = slices.getElement( sliceIndex ).chunkSize;
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
   tnlAssert( sliceIndex < this->rows, );
   const IndexType& chunkSize = slices.getElement( sliceIndex ).chunkSize;
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
   if( ! tnlSparseMatrix< Real, Device, Index >::save( file ) ||
       ! this->chunksToRowsMapping.save( file ) ||
       ! this->slicesToRowsMapping.save( file ) ||
       ! this->rowPointers.save( file ) ||
       ! this->slices.save( file ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::load( tnlFile& file )
{
   if( ! tnlSparseMatrix< Real, Device, Index >::load( file ) ||
       ! this->chunksToRowsMapping.load( file ) ||
       ! this->slicesToRowsMapping.load( file ) ||
       ! this->rowPointers.load( file ) ||
       ! this->slices.load( file ) )
      return false;
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
      tnlAssert( sliceIndex < this->rows, );
      const IndexType& chunkSize = slices.getElement( sliceIndex ).chunkSize;
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
