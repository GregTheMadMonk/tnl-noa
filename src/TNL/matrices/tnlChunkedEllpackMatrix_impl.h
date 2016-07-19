/***************************************************************************
                          tnlChunkedEllpackMatrix_impl.h  -  description
                             -------------------
    begin                : Dec 12, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/matrices/tnlChunkedEllpackMatrix.h>
#include <TNL/core/vectors/tnlVector.h>
#include <TNL/core/mfuncs.h>

#ifdef HAVE_CUDA
#include <cuda.h>
#endif

namespace TNL {

template< typename Real,
          typename Index,
          typename Vector >
void tnlChunkedEllpackMatrixVectorProductCuda( const tnlChunkedEllpackMatrix< Real, tnlCuda, Index >& matrix,
                                               const Vector& inVector,
                                               Vector& outVector );


template< typename Real,
          typename Device,
          typename Index >
tnlChunkedEllpackMatrix< Real, Device, Index >::tnlChunkedEllpackMatrix()
: chunksInSlice( 256 ),
  desiredChunkSize( 16 ),
  numberOfSlices( 0 )
{
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlChunkedEllpackMatrix< Real, Device, Index >::getType()
{
   return tnlString( "tnlChunkedEllpackMatrix< ") +
          tnlString( TNL::getType< Real >() ) +
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
              std::cerr << "rows = " << rows
                   << " columns = " << columns << std::endl );
   if( ! tnlSparseMatrix< Real, Device, Index >::setDimensions( rows, columns ) )
      return false;

   /****
    * Allocate slice info array. Note that there cannot be
    * more slices than rows.
    */
   if( ! this->slices.setSize( this->rows ) ||
       ! this->rowToChunkMapping.setSize( this-> rows ) ||
       ! this->rowToSliceMapping.setSize( this->rows ) ||
       ! this->rowPointers.setSize( this->rows + 1 ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlChunkedEllpackMatrix< Real, Device, Index >::resolveSliceSizes( const tnlVector< Index, tnlHost, Index >& rowLengths )
{
   /****
    * Iterate over rows and allocate slices so that each slice has
    * approximately the same number of allocated elements
    */
   const IndexType desiredElementsInSlice =
            this->chunksInSlice * this->desiredChunkSize;

   IndexType row( 0 ),
             sliceSize( 0 ),
             allocatedElementsInSlice( 0 );
   numberOfSlices = 0;
   while( row < this->rows )
   {
      /****
       * Add one row to the current slice until we reach the desired
       * number of elements in a slice.
       */
      allocatedElementsInSlice += rowLengths[ row ];
      sliceSize++;
      row++;
      if( allocatedElementsInSlice < desiredElementsInSlice  )
          if( row < this->rows && sliceSize < chunksInSlice ) continue;
      tnlAssert( sliceSize >0, );
      this->slices[ numberOfSlices ].size = sliceSize;
      this->slices[ numberOfSlices ].firstRow = row - sliceSize;
      this->slices[ numberOfSlices ].pointer = allocatedElementsInSlice; // this is only temporary
      sliceSize = 0;
      numberOfSlices++;
      allocatedElementsInSlice = 0;
   }
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::setSlice( const CompressedRowsLengthsVector& rowLengths,
                                                               const IndexType sliceIndex,
                                                               IndexType& elementsToAllocation )
{
   /****
    * Now, compute the number of chunks per each row.
    * Each row get one chunk by default.
    * Then each row will get additional chunks w.r. to the
    * number of the elements in the row. If there are some
    * free chunks left, repeat it again.
    */
   const IndexType sliceSize = this->slices[ sliceIndex ].size;
   const IndexType sliceBegin = this->slices[ sliceIndex ].firstRow;
   const IndexType allocatedElementsInSlice = this->slices[ sliceIndex ].pointer;
   const IndexType sliceEnd = sliceBegin + sliceSize;

   IndexType freeChunks = this->chunksInSlice - sliceSize;
   for( IndexType i = sliceBegin; i < sliceEnd; i++ )
      this->rowToChunkMapping.setElement( i, 1 );

   int totalAddedChunks( 0 );
   int maxRowLength( rowLengths[ sliceBegin ] );
   for( IndexType i = sliceBegin; i < sliceEnd; i++ )
   {
      double rowRatio( 0.0 );
      if( allocatedElementsInSlice != 0 )
         rowRatio = ( double ) rowLengths[ i ] / ( double ) allocatedElementsInSlice;
      const IndexType addedChunks = freeChunks * rowRatio;
      totalAddedChunks += addedChunks;
      this->rowToChunkMapping[ i ] += addedChunks;
      if( maxRowLength < rowLengths[ i ] )
         maxRowLength = rowLengths[ i ];
   }
   freeChunks -= totalAddedChunks;
   while( freeChunks )
      for( IndexType i = sliceBegin; i < sliceEnd && freeChunks; i++ )
         if( rowLengths[ i ] == maxRowLength )
         {
            this->rowToChunkMapping[ i ]++;
            freeChunks--;
         }

   /****
    * Compute the chunk size
    */
   IndexType maxChunkInSlice( 0 );
   for( IndexType i = sliceBegin; i < sliceEnd; i++ )
      maxChunkInSlice = max( maxChunkInSlice,
                          ceil( ( RealType ) rowLengths[ i ] /
                                ( RealType ) this->rowToChunkMapping[ i ] ) );
   tnlAssert( maxChunkInSlice > 0,
              std::cerr << " maxChunkInSlice = " << maxChunkInSlice << std::endl );

   /****
    * Set-up the slice info.
    */
   this->slices[ sliceIndex ].chunkSize = maxChunkInSlice;
   this->slices[ sliceIndex ].pointer = elementsToAllocation;
   elementsToAllocation += this->chunksInSlice * maxChunkInSlice;

   for( IndexType i = sliceBegin; i < sliceEnd; i++ )
      this->rowToSliceMapping[ i ] = sliceIndex;

   for( IndexType i = sliceBegin; i < sliceEnd; i++ )
   {
      this->rowPointers[ i + 1 ] = maxChunkInSlice*rowToChunkMapping[ i ];
      tnlAssert( this->rowPointers[ i ] >= 0,
                 std::cerr << "this->rowPointers[ i ] = " << this->rowPointers[ i ] );
      tnlAssert( this->rowPointers[ i + 1 ] >= 0,
                 std::cerr << "this->rowPointers[ i + 1 ] = " << this->rowPointers[ i + 1 ] );
   }

   /****
    * Finish the row to chunk mapping by computing the prefix sum.
    */
   for( IndexType j = sliceBegin + 1; j < sliceEnd; j++ )
      rowToChunkMapping[ j ] += rowToChunkMapping[ j - 1 ];
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::setCompressedRowsLengths( const CompressedRowsLengthsVector& rowLengths )
{
   tnlAssert( this->getRows() > 0, );
   tnlAssert( this->getColumns() > 0, );

   IndexType elementsToAllocation( 0 );

   if( DeviceType::DeviceType == ( int ) tnlHostDevice )
   {
      DeviceDependentCode::resolveSliceSizes( *this, rowLengths );
      this->rowPointers.setElement( 0, 0 );
      for( IndexType sliceIndex = 0; sliceIndex < numberOfSlices; sliceIndex++ )
         this->setSlice( rowLengths, sliceIndex, elementsToAllocation );
      this->rowPointers.computePrefixSum();
   }
   if( DeviceType::DeviceType == ( int ) tnlCudaDevice )
   {
      tnlChunkedEllpackMatrix< RealType, tnlHost, IndexType > hostMatrix;
      hostMatrix.setDimensions( this->getRows(), this->getColumns() );
      tnlVector< IndexType, tnlHost, IndexType > hostCompressedRowsLengths;
      hostCompressedRowsLengths.setLike( rowLengths);
      hostCompressedRowsLengths = rowLengths;
      hostMatrix.setNumberOfChunksInSlice( this->chunksInSlice );
      hostMatrix.setDesiredChunkSize( this->desiredChunkSize );
      hostMatrix.setCompressedRowsLengths( hostCompressedRowsLengths );

      this->rowToChunkMapping.setLike( hostMatrix.rowToChunkMapping );
      this->rowToChunkMapping = hostMatrix.rowToChunkMapping;
      this->rowToSliceMapping.setLike( hostMatrix.rowToSliceMapping );
      this->rowToSliceMapping = hostMatrix.rowToSliceMapping;
      this->rowPointers.setLike( hostMatrix.rowPointers );
      this->rowPointers = hostMatrix.rowPointers;
      this->slices.setLike( hostMatrix.slices );
      this->slices = hostMatrix.slices;
      this->numberOfSlices = hostMatrix.numberOfSlices;
      elementsToAllocation = hostMatrix.values.getSize();
   }
   this->maxRowLength = rowLengths.max();
   return tnlSparseMatrix< Real, Device, Index >::allocateMatrixElements( elementsToAllocation );
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlChunkedEllpackMatrix< Real, Device, Index >::getRowLength( const IndexType row ) const
{
   const IndexType& sliceIndex = rowToSliceMapping[ row ];
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
       ! this->rowToChunkMapping.setLike( matrix.rowToChunkMapping ) ||
       ! this->rowToSliceMapping.setLike( matrix.rowToSliceMapping ) ||
       ! this->slices.setLike( matrix.slices ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlChunkedEllpackMatrix< Real, Device, Index >::reset()
{
   tnlSparseMatrix< Real, Device, Index >::reset();
   this->slices.reset();
   this->rowToChunkMapping.reset();
   this->rowToSliceMapping.reset();
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
__cuda_callable__
Index tnlChunkedEllpackMatrix< Real, Device, Index >::getNumberOfChunksInSlice() const
{
   return this->chunksInSlice;
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
__cuda_callable__
Index tnlChunkedEllpackMatrix< Real, Device, Index >::getNumberOfSlices() const
{
   return this->numberOfSlices;
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
              std::cerr << "this->getRows() = " << this->getRows()
                   << " matrix.getRows() = " << matrix.getRows()
                   << " this->getColumns() = " << this->getColumns()
                   << " matrix.getColumns() = " << matrix.getColumns() );
   // TODO: implement this
   return false;
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
__cuda_callable__
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
__cuda_callable__
bool tnlChunkedEllpackMatrix< Real, Device, Index >::addElementFast( const IndexType row,
                                                                     const IndexType _column,
                                                                     const RealType& _value,
                                                                     const RealType& _thisElementMultiplicator )
{
   // TODO: return this back when CUDA kernels support std::cerr
   /*tnlAssert( row >= 0 && row < this->rows &&
              _column >= 0 && _column <= this->columns,
              std::cerr << " row = " << row
                   << " column = " << _column
                   << " this->rows = " << this->rows
                   << " this->columns = " << this-> columns );*/

   const IndexType& sliceIndex = rowToSliceMapping[ row ];
   tnlAssert( sliceIndex < this->rows, );
   IndexType chunkIndex( 0 );
   if( row != slices[ sliceIndex ].firstRow )
      chunkIndex = rowToChunkMapping[ row - 1 ];
   const IndexType lastChunk = rowToChunkMapping[ row ];
   const IndexType sliceOffset = slices[ sliceIndex ].pointer;
   const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
   IndexType column( _column );
   RealType value( _value ), thisElementMultiplicator( _thisElementMultiplicator );
   while( chunkIndex < lastChunk - 1 &&
          ! addElementToChunkFast( sliceOffset, chunkIndex, chunkSize, column, value, thisElementMultiplicator ) )
      chunkIndex++;
   if( chunkIndex < lastChunk - 1 )
      return true;
   return addElementToChunkFast( sliceOffset, chunkIndex, chunkSize, column, value, thisElementMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool tnlChunkedEllpackMatrix< Real, Device, Index >::addElementToChunkFast( const IndexType sliceOffset,
                                                                            const IndexType chunkIndex,
                                                                            const IndexType chunkSize,
                                                                            IndexType& column,
                                                                            RealType& value,
                                                                            RealType& thisElementMultiplicator )
{
   IndexType elementPtr, chunkEnd, step;

   DeviceDependentCode::initChunkTraverse( sliceOffset,
                                           chunkIndex,
                                           chunkSize,
                                           this->getNumberOfChunksInSlice(),
                                           elementPtr,
                                           chunkEnd,
                                           step );
   IndexType col;
   while( elementPtr < chunkEnd &&
          ( col = this->columnIndexes[ elementPtr ] ) < column &&
          col != this->getPaddingIndex() )
      elementPtr += step;

   if( col == column )
   {
      if( thisElementMultiplicator != 0.0 )
         this->values[ elementPtr ] = value + thisElementMultiplicator * this->values[ elementPtr ];
      else
         this->values[ elementPtr ] = value;
      return true;
   }
   if( col < column )
      return false;

   IndexType i( chunkEnd - step );

   /****
    * Check if the chunk is already full. In this case, the last element
    * will be inserted to the next chunk.
    */
   IndexType elementColumn( column );
   RealType elementValue( value );
   bool chunkOverflow( false );
   if( ( col = this->columnIndexes[ i ] ) != this->getPaddingIndex() )
   {
      chunkOverflow = true;
      column = col;
      value = this->values[ i ];
      thisElementMultiplicator = 0;
   }

   while( i > elementPtr )
   {
      this->columnIndexes[ i ] = this->columnIndexes[ i - step ];
      this->values[ i ] = this->values[ i - step ];
      i -= step;
   }
   this->values[ elementPtr ] = elementValue;
   this->columnIndexes[ elementPtr ] = elementColumn;
   return ! chunkOverflow;
}


template< typename Real,
          typename Device,
          typename Index >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::addElement( const IndexType row,
                                                                 const IndexType _column,
                                                                 const RealType& _value,
                                                                 const RealType& _thisElementMultiplicator )
{
   tnlAssert( row >= 0 && row < this->rows &&
              _column >= 0 && _column <= this->columns,
              std::cerr << " row = " << row
                   << " column = " << _column
                   << " this->rows = " << this->rows
                   << " this->columns = " << this-> columns );

   const IndexType& sliceIndex = rowToSliceMapping.getElement( row );
   tnlAssert( sliceIndex < this->rows, );
   IndexType chunkIndex( 0 );
   if( row != slices.getElement( sliceIndex ).firstRow )
      chunkIndex = rowToChunkMapping.getElement( row - 1 );
   const IndexType lastChunk = rowToChunkMapping.getElement( row );
   const IndexType sliceOffset = slices.getElement( sliceIndex ).pointer;
   const IndexType chunkSize = slices.getElement( sliceIndex ).chunkSize;
   IndexType column( _column );
   RealType value( _value ), thisElementMultiplicator( _thisElementMultiplicator );
   while( chunkIndex < lastChunk - 1 &&
          ! addElementToChunk( sliceOffset, chunkIndex, chunkSize, column, value, thisElementMultiplicator ) )
      chunkIndex++;
   if( chunkIndex < lastChunk - 1 )
      return true;
   return addElementToChunk( sliceOffset, chunkIndex, chunkSize, column, value, thisElementMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::addElementToChunk( const IndexType sliceOffset,
                                                                        const IndexType chunkIndex,
                                                                        const IndexType chunkSize,
                                                                        IndexType& column,
                                                                        RealType& value,
                                                                        RealType& thisElementMultiplicator )
{
   IndexType elementPtr, chunkEnd, step;
   DeviceDependentCode::initChunkTraverse( sliceOffset,
                                           chunkIndex,
                                           chunkSize,
                                           this->getNumberOfChunksInSlice(),
                                           elementPtr,
                                           chunkEnd,
                                           step );
   IndexType col;
   while( elementPtr < chunkEnd &&
          ( col = this->columnIndexes.getElement( elementPtr ) ) < column &&
          col != this->getPaddingIndex() )
      elementPtr += step;
   if( col == column )
   {
      if( thisElementMultiplicator != 0.0 )
         this->values.setElement( elementPtr, value + thisElementMultiplicator * this->values.getElement( elementPtr ) );
      else
         this->values.setElement( elementPtr, value );
      return true;
   }
   if( col < column )
      return false;

   IndexType i( chunkEnd - step );

   /****
    * Check if the chunk is already full. In this case, the last element
    * will be inserted to the next chunk.
    */
   IndexType elementColumn( column );
   RealType elementValue( value );
   bool chunkOverflow( false );
   if( ( col = this->columnIndexes.getElement( i ) ) != this->getPaddingIndex() )
   {
      chunkOverflow = true;
      column = col;
      value = this->values.getElement( i );
      thisElementMultiplicator = 0;
   }

   while( i > elementPtr )
   {
      this->columnIndexes.setElement( i, this->columnIndexes.getElement( i - step ) );
      this->values.setElement( i, this->values.getElement( i - step ) );
      i -= step;
   }
   this->values.setElement( elementPtr, elementValue );
   this->columnIndexes.setElement( elementPtr, elementColumn );
   return ! chunkOverflow;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool tnlChunkedEllpackMatrix< Real, Device, Index >::setRowFast( const IndexType row,
                                                                 const IndexType* columnIndexes,
                                                                 const RealType* values,
                                                                 const IndexType elements )
{
   // TODO: return this back when CUDA kernels support std::cerr
   /*tnlAssert( row >= 0 && row < this->rows,
              std::cerr << " row = " << row
                   << " this->rows = " << this->rows );*/
   const IndexType sliceIndex = rowToSliceMapping[ row ];
   //tnlAssert( sliceIndex < this->rows, );
   IndexType chunkIndex( 0 );
   if( row != slices[ sliceIndex ].firstRow )
      chunkIndex = rowToChunkMapping[ row - 1 ];
   const IndexType lastChunk = rowToChunkMapping[ row ];
   const IndexType sliceOffset = slices[ sliceIndex ].pointer;
   const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
   if( chunkSize * ( lastChunk - chunkIndex ) < elements )
      return false;
   IndexType offset( 0 );
   while( chunkIndex < lastChunk )
   {
      /****
       * Note, if elements - offset is non-positive then setChunkFast
       * just erase the chunk.
       */
      setChunkFast( sliceOffset,
                    chunkIndex,
                    chunkSize,
                    &columnIndexes[ offset ],
                    &values[ offset ],
                    elements - offset );
      chunkIndex++;
      offset += chunkSize;
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
void tnlChunkedEllpackMatrix< Real, Device, Index >::setChunkFast( const IndexType sliceOffset,
                                                                   const IndexType chunkIndex,
                                                                   const IndexType chunkSize,
                                                                   const IndexType* columnIndexes,
                                                                   const RealType* values,
                                                                   const IndexType elements )
{
   IndexType elementPtr, chunkEnd, step;
   DeviceDependentCode::initChunkTraverse( sliceOffset,
                                           chunkIndex,
                                           chunkSize,
                                           this->getNumberOfChunksInSlice(),
                                           elementPtr,
                                           chunkEnd,
                                           step );
   IndexType i( 0 );
   while( i < chunkSize && i < elements )
   {
      this->values[ elementPtr ] = values[ i ];
      this->columnIndexes[ elementPtr ] = columnIndexes[ i ];
      i++;
      elementPtr += step;
   }
   while( i < chunkSize )
   {
      this->columnIndexes[ elementPtr ] = this->getPaddingIndex();
      elementPtr += step;
      i++;
   }
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::setRow( const IndexType row,
                                                             const IndexType* columnIndexes,
                                                             const RealType* values,
                                                             const IndexType elements )
{
   tnlAssert( row >= 0 && row < this->rows,
              std::cerr << " row = " << row
                   << " this->rows = " << this->rows );

   const IndexType sliceIndex = rowToSliceMapping.getElement( row );
   tnlAssert( sliceIndex < this->rows, );
   IndexType chunkIndex( 0 );
   if( row != slices.getElement( sliceIndex ).firstRow )
      chunkIndex = rowToChunkMapping.getElement( row - 1 );
   const IndexType lastChunk = rowToChunkMapping.getElement( row );
   const IndexType sliceOffset = slices.getElement( sliceIndex ).pointer;
   const IndexType chunkSize = slices.getElement( sliceIndex ).chunkSize;
   if( chunkSize * ( lastChunk - chunkIndex ) < elements )
      return false;
   IndexType offset( 0 );
   while( chunkIndex < lastChunk )
   {
      /****
       * Note, if elements - offset is non-positive then setChunkFast
       * just erase the chunk.
       */
      setChunk( sliceOffset,
                chunkIndex,
                chunkSize,
                &columnIndexes[ offset ],
                &values[ offset ],
                elements - offset );
      chunkIndex++;
      offset += chunkSize;
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlChunkedEllpackMatrix< Real, Device, Index >::setChunk( const IndexType sliceOffset,
                                                               const IndexType chunkIndex,
                                                               const IndexType chunkSize,
                                                               const IndexType* columnIndexes,
                                                               const RealType* values,
                                                               const IndexType elements )
{
   IndexType elementPtr, chunkEnd, step;
   DeviceDependentCode::initChunkTraverse( sliceOffset,
                                           chunkIndex,
                                           chunkSize,
                                           this->getNumberOfChunksInSlice(),
                                           elementPtr,
                                           chunkEnd,
                                           step );
   IndexType i( 0 );
   while( i < chunkSize && i < elements )
   {
      this->values.setElement( elementPtr, values[ i ] );
      this->columnIndexes.setElement( elementPtr, columnIndexes[ i ] );
      i++;
      elementPtr += step;
   }
   while( i < chunkSize )
   {
      this->columnIndexes.setElement( elementPtr, this->getPaddingIndex() );
      elementPtr += step;
      i++;
   }
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
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
__cuda_callable__
Real tnlChunkedEllpackMatrix< Real, Device, Index >::getElementFast( const IndexType row,
                                                                     const IndexType column ) const
{
   const IndexType sliceIndex = rowToSliceMapping[ row ];
   tnlAssert( sliceIndex < this->rows, );
   IndexType chunkIndex( 0 );
   if( row != slices[ sliceIndex ].firstRow )
      chunkIndex = rowToChunkMapping[ row - 1 ];
   const IndexType lastChunk = rowToChunkMapping[ row ];
   const IndexType sliceOffset = slices[ sliceIndex ].pointer;
   const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
   RealType value( 0.0 );
   while( chunkIndex < lastChunk &&
          ! getElementInChunk( sliceOffset, chunkIndex, chunkSize, column, value ) )
      chunkIndex++;
   return value;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool tnlChunkedEllpackMatrix< Real, Device, Index >::getElementInChunkFast( const IndexType sliceOffset,
                                                                            const IndexType chunkIndex,
                                                                            const IndexType chunkSize,
                                                                            const IndexType column,
                                                                            RealType& value) const
{
   IndexType elementPtr, chunkEnd, step;
   DeviceDependentCode::initChunkTraverse( sliceOffset, chunkIndex, chunkSize, elementPtr, chunkEnd, step );
   while( elementPtr < chunkEnd )
   {
      const IndexType col = this->columnIndexes[ elementPtr ];
      if( col == column )
         value = this->values[ elementPtr ];
      if( col >= column || col == this->getPaddingIndex() )
         return true;
      elementPtr += step;
   }
   return false;
}


template< typename Real,
          typename Device,
          typename Index >
Real tnlChunkedEllpackMatrix< Real, Device, Index >::getElement( const IndexType row,
                                                                 const IndexType column ) const
{
   const IndexType& sliceIndex = rowToSliceMapping.getElement( row );
   tnlAssert( sliceIndex < this->rows,
              std::cerr << " sliceIndex = " << sliceIndex
                   << " this->rows = " << this->rows << std::endl; );
   IndexType chunkIndex( 0 );
   if( row != slices.getElement( sliceIndex ).firstRow )
      chunkIndex = rowToChunkMapping.getElement( row - 1 );
   const IndexType lastChunk = rowToChunkMapping.getElement( row );
   const IndexType sliceOffset = slices.getElement( sliceIndex ).pointer;
   const IndexType chunkSize = slices.getElement( sliceIndex ).chunkSize;
   RealType value( 0.0 );
   while( chunkIndex < lastChunk &&
          ! getElementInChunk( sliceOffset, chunkIndex, chunkSize, column, value ) )
      chunkIndex++;
   return value;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlChunkedEllpackMatrix< Real, Device, Index >::getElementInChunk( const IndexType sliceOffset,
                                                                        const IndexType chunkIndex,
                                                                        const IndexType chunkSize,
                                                                        const IndexType column,
                                                                        RealType& value) const
{
   IndexType elementPtr, chunkEnd, step;
   DeviceDependentCode::initChunkTraverse( sliceOffset,
                                           chunkIndex,
                                           chunkSize,
                                           this->getNumberOfChunksInSlice(),
                                           elementPtr,
                                           chunkEnd,
                                           step );
   while( elementPtr < chunkEnd )
   {
      const IndexType col = this->columnIndexes.getElement( elementPtr );
      if( col == column )
         value = this->values.getElement( elementPtr );
      if( col >= column || col == this->getPaddingIndex() )
         return true;
      elementPtr += step;
   }
   return false;
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
void tnlChunkedEllpackMatrix< Real, Device, Index >::getRowFast( const IndexType row,
                                                                 IndexType* columns,
                                                                 RealType* values ) const
{
   const IndexType& sliceIndex = rowToSliceMapping[ row ];
   tnlAssert( sliceIndex < this->rows, );
   IndexType chunkIndex( 0 );
   if( row != slices[ sliceIndex ].firstRow )
      chunkIndex = rowToChunkMapping[ row - 1 ];
   const IndexType lastChunk = rowToChunkMapping[ row ];
   const IndexType sliceOffset = slices[ sliceIndex ].pointer;
   const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
   RealType value( 0.0 );
   IndexType offset( 0 );
   while( chunkIndex < lastChunk )
   {
      getChunk( sliceOffset,
                chunkIndex,
                min( chunkSize, this->getColumns - offset ),
                &columns[ offset ],
                &values[ offset ] );
      chunkIndex++;
      offset += chunkSize;
   }
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
void tnlChunkedEllpackMatrix< Real, Device, Index >::getChunkFast( const IndexType sliceOffset,
                                                                   const IndexType chunkIndex,
                                                                   const IndexType chunkSize,
                                                                   IndexType* columnIndexes,
                                                                   RealType* values ) const
{
   IndexType elementPtr, chunkEnd, step;
   DeviceDependentCode::initChunkTraverse( sliceOffset, chunkIndex, chunkSize, elementPtr, chunkEnd, step );
   IndexType i( 0 );
   while( i < chunkSize )
   {
      columnIndexes[ i ] = this->columnIndexes[ elementPtr ];
      values[ i ] = this->values[ elementPtr ];
      i++;
      elementPtr += step;
   }
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename tnlChunkedEllpackMatrix< Real, Device, Index >::MatrixRow
tnlChunkedEllpackMatrix< Real, Device, Index >::
getRow( const IndexType rowIndex )
{
   const IndexType rowOffset = this->rowPointers[ rowIndex ];
   const IndexType rowLength = this->rowPointers[ rowIndex + 1 ] - rowOffset;
   return MatrixRow( &this->columns[ rowOffset ],
                     &this->values[ rowOffset ],
                     rowLength,
                     1 );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const typename tnlChunkedEllpackMatrix< Real, Device, Index >::MatrixRow
tnlChunkedEllpackMatrix< Real, Device, Index >::
getRow( const IndexType rowIndex ) const
{
   const IndexType rowOffset = this->rowPointers[ rowIndex ];
   const IndexType rowLength = this->rowPointers[ rowIndex + 1 ] - rowOffset;
   return MatrixRow( &this->columns[ rowOffset ],
                     &this->values[ rowOffset ],
                     rowLength,
                     1 );
}


/*template< typename Real,
          typename Device,
          typename Index >
void tnlChunkedEllpackMatrix< Real, Device, Index >::getRow( const IndexType row,
                                                             IndexType* columns,
                                                             RealType* values ) const
{
   const IndexType& sliceIndex = rowToSliceMapping.getElement( row );
   tnlAssert( sliceIndex < this->rows, );
   IndexType chunkIndex( 0 );
   if( row != slices.getElement( sliceIndex ).firstRow )
      chunkIndex = rowToChunkMapping.getElement( row - 1 );
   const IndexType lastChunk = rowToChunkMapping.getElement( row );
   const IndexType sliceOffset = slices.getElement( sliceIndex ).pointer;
   const IndexType chunkSize = slices.getElement( sliceIndex ).chunkSize;
   RealType value( 0.0 );
   IndexType offset( 0 );
   while( chunkIndex < lastChunk )
   {
      getChunk( sliceOffset,
                chunkIndex,
                min( chunkSize, this->getColumns() - offset ),
                &columns[ offset ],
                &values[ offset ] );
      chunkIndex++;
      offset += chunkSize;
   }
}*/

template< typename Real,
          typename Device,
          typename Index >
void tnlChunkedEllpackMatrix< Real, Device, Index >::getChunk( const IndexType sliceOffset,
                                                               const IndexType chunkIndex,
                                                               const IndexType chunkSize,
                                                               IndexType* columnIndexes,
                                                               RealType* values ) const
{
   IndexType elementPtr, chunkEnd, step;
   DeviceDependentCode::initChunkTraverse( sliceOffset,
                                           chunkIndex,
                                           chunkSize,
                                           this->getNumberOfChunksInSlice(),
                                           elementPtr,
                                           chunkEnd,
                                           step );
   IndexType i( 0 );
   while( i < chunkSize )
   {
      columnIndexes[ i ] = this->columnIndexes.getElement( elementPtr );
      values[ i ] = this->values.getElement( elementPtr );
      i++;
      elementPtr += step;
   }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
__cuda_callable__
typename Vector::RealType tnlChunkedEllpackMatrix< Real, Device, Index >::rowVectorProduct( const IndexType row,
                                                                                            const Vector& vector ) const
{
   /*tnlAssert( row >=0 && row < this->rows,
            std::cerr << " row = " << row << " this->rows = " << this->rows );*/

   const IndexType sliceIndex = rowToSliceMapping[ row ];
   //tnlAssert( sliceIndex < this->rows, );
   IndexType chunkIndex( 0 );
   if( row != slices[ sliceIndex ].firstRow )
      chunkIndex = rowToChunkMapping[ row - 1 ];
   const IndexType lastChunk = rowToChunkMapping[ row ];
   const IndexType sliceOffset = slices[ sliceIndex ].pointer;
   const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
   RealType result( 0.0 );
   while( chunkIndex < lastChunk )
   {
      result += chunkVectorProduct( sliceOffset,
                                    chunkIndex,
                                    chunkSize,
                                    vector );
      chunkIndex++;
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
__cuda_callable__
typename Vector::RealType tnlChunkedEllpackMatrix< Real, Device, Index >::chunkVectorProduct( const IndexType sliceOffset,
                                                                                              const IndexType chunkIndex,
                                                                                              const IndexType chunkSize,
                                                                                              const Vector& vector ) const
{
   IndexType elementPtr, chunkEnd, step;
   DeviceDependentCode::initChunkTraverse( sliceOffset,
                                           chunkIndex,
                                           chunkSize,
                                           this->getNumberOfChunksInSlice(),
                                           elementPtr,
                                           chunkEnd,
                                           step );
   IndexType i( 0 ), col;
   typename Vector::RealType result( 0.0 );
   while( i < chunkSize && ( col = this->columnIndexes[ elementPtr ] ) != this->getPaddingIndex() )
   {
      result += this->values[ elementPtr ] * vector[ col ];
      i++;
      elementPtr += step;
   }
   return result;
}


#ifdef HAVE_CUDA
template< typename Real,
          typename Device,
          typename Index >
   template< typename InVector,
             typename OutVector >
__device__ void tnlChunkedEllpackMatrix< Real, Device, Index >::computeSliceVectorProduct( const InVector* inVector,
                                                                                           OutVector* outVector,
                                                                                           int sliceIdx  ) const
{
   static_assert( DeviceType::DeviceType == tnlCudaDevice, "" );

   RealType* chunkProducts = getSharedMemory< RealType >();
   ChunkedEllpackSliceInfo* sliceInfo = ( ChunkedEllpackSliceInfo* ) & chunkProducts[ blockDim.x ];

   if( threadIdx.x == 0 )
      ( *sliceInfo ) = this->slices[ sliceIdx ];
   __syncthreads();
   chunkProducts[ threadIdx.x ] = this->chunkVectorProduct( sliceInfo->pointer,
                                                            threadIdx.x,
                                                            sliceInfo->chunkSize,
                                                            *inVector );
   __syncthreads();
   if( threadIdx.x < sliceInfo->size )
   {
      const IndexType row = sliceInfo->firstRow + threadIdx.x;
      IndexType chunkIndex( 0 );
      if( threadIdx.x != 0 )
         chunkIndex = this->rowToChunkMapping[ row - 1 ];
      const IndexType lastChunk = this->rowToChunkMapping[ row ];
      RealType result( 0.0 );
      while( chunkIndex < lastChunk )
         result += chunkProducts[ chunkIndex++ ];
      ( *outVector )[ row ] = result;
   }
}
#endif

template< typename Real,
          typename Device,
          typename Index >
   template< typename InVector,
             typename OutVector >
void tnlChunkedEllpackMatrix< Real, Device, Index >::vectorProduct( const InVector& inVector,
                                                                    OutVector& outVector ) const
{
   DeviceDependentCode::vectorProduct( *this, inVector, outVector );
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
   tnlAssert( false, std::cerr << "TODO: implement" );
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
   tnlAssert( false, std::cerr << "TODO: implement" );
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
              std::cerr << "row = " << row
                   << " this->getRows() = " << this->getRows() << std::endl );

   RealType diagonalValue( 0.0 );
   RealType sum( 0.0 );

   const IndexType& sliceIndex = rowToSliceMapping[ row ];
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
      std::cerr << "There is zero on the diagonal in " << row << "-th row of a matrix. I cannot perform SOR iteration." << std::endl;
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
       ! this->rowToChunkMapping.save( file ) ||
       ! this->rowToSliceMapping.save( file ) ||
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
       ! this->rowToChunkMapping.load( file ) ||
       ! this->rowToSliceMapping.load( file ) ||
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
void tnlChunkedEllpackMatrix< Real, Device, Index >::print( std::ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";

      const IndexType& sliceIndex = rowToSliceMapping.getElement( row );
      //tnlAssert( sliceIndex < this->rows, );
      const IndexType& chunkSize = slices.getElement( sliceIndex ).chunkSize;
      IndexType elementPtr = rowPointers.getElement( row );
      const IndexType rowEnd = rowPointers.getElement( row + 1 );

      while( elementPtr < rowEnd &&
             this->columnIndexes.getElement( elementPtr ) < this->columns &&
             this->columnIndexes.getElement( elementPtr ) != this->getPaddingIndex() )
      {
         const Index column = this->columnIndexes.getElement( elementPtr );
         str << " Col:" << column << "->" << this->values.getElement( elementPtr ) << "\t";
         elementPtr++;
      }
      str << std::endl;
   }
}

template< typename Real,
          typename Device,
          typename Index >
void tnlChunkedEllpackMatrix< Real, Device, Index >::printStructure( std::ostream& str,
                                                                     const tnlString& name ) const
{
   const IndexType numberOfSlices = this->getNumberOfSlices();
   str << "Matrix type: " << getType() << std::endl
       << "Marix name: " << name << std::endl
       << "Rows: " << this->getRows() << std::endl
       << "Columns: " << this->getColumns() << std::endl
       << "Slices: " << numberOfSlices << std::endl;
   for( IndexType i = 0; i < numberOfSlices; i++ )
      str << "   Slice " << i
          << " : size = " << this->slices.getElement( i ).size
          << " chunkSize = " << this->slices.getElement( i ).chunkSize
          << " firstRow = " << this->slices.getElement( i ).firstRow
          << " pointer = " << this->slices.getElement( i ).pointer << std::endl;
   for( IndexType i = 0; i < this->getRows(); i++ )
      str << "Row " << i
          << " : slice = " << this->rowToSliceMapping.getElement( i )
          << " chunk = " << this->rowToChunkMapping.getElement( i ) << std::endl;
}

template<>
class tnlChunkedEllpackMatrixDeviceDependentCode< tnlHost >
{
   public:

      typedef tnlHost Device;

      template< typename Real,
                typename Index >
      static void resolveSliceSizes( tnlChunkedEllpackMatrix< Real, Device, Index >& matrix,
                                     const typename tnlChunkedEllpackMatrix< Real, Device, Index >::CompressedRowsLengthsVector& rowLengths )
      {
         matrix.resolveSliceSizes( rowLengths );
      }

      template< typename Index >
      __cuda_callable__
      static void initChunkTraverse( const Index sliceOffset,
                                     const Index chunkIndex,
                                     const Index chunkSize,
                                     const Index chunksInSlice,
                                     Index& chunkBegining,
                                     Index& chunkEnd,
                                     Index& step )
      {
         chunkBegining = sliceOffset + chunkIndex * chunkSize;
         chunkEnd = chunkBegining + chunkSize;
         step = 1;
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const tnlChunkedEllpackMatrix< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         for( Index row = 0; row < matrix.getRows(); row ++ )
            outVector[ row ] = matrix.rowVectorProduct( row, inVector );
      }
};

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename InVector,
          typename OutVector >
__global__ void tnlChunkedEllpackMatrixVectorProductCudaKernel( const tnlChunkedEllpackMatrix< Real, tnlCuda, Index >* matrix,
                                                                const InVector* inVector,
                                                                OutVector* outVector,
                                                                int gridIdx )
{
   const Index sliceIdx = gridIdx * tnlCuda::getMaxGridSize() + blockIdx.x;
   if( sliceIdx < matrix->getNumberOfSlices() )
      matrix->computeSliceVectorProduct( inVector, outVector, sliceIdx );

}
#endif


template<>
class tnlChunkedEllpackMatrixDeviceDependentCode< tnlCuda >
{
   public:

      typedef tnlCuda Device;
 
      template< typename Real,
                typename Index >
      static void resolveSliceSizes( tnlChunkedEllpackMatrix< Real, Device, Index >& matrix,
                                     const typename tnlChunkedEllpackMatrix< Real, Device, Index >::CompressedRowsLengthsVector& rowLengths )
      {
      }
 
      template< typename Index >
      __cuda_callable__
      static void initChunkTraverse( const Index sliceOffset,
                                     const Index chunkIndex,
                                     const Index chunkSize,
                                     const Index chunksInSlice,
                                     Index& chunkBegining,
                                     Index& chunkEnd,
                                     Index& step )
      {
         chunkBegining = sliceOffset + chunkIndex;
         chunkEnd = chunkBegining + chunkSize * chunksInSlice;
         step = chunksInSlice;

         /*chunkBegining = sliceOffset + chunkIndex * chunkSize;
         chunkEnd = chunkBegining + chunkSize;
         step = 1;*/
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const tnlChunkedEllpackMatrix< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         #ifdef HAVE_CUDA
            typedef tnlChunkedEllpackMatrix< Real, tnlCuda, Index > Matrix;
            typedef Index IndexType;
            typedef Real RealType;
            Matrix* kernel_this = tnlCuda::passToDevice( matrix );
            InVector* kernel_inVector = tnlCuda::passToDevice( inVector );
            OutVector* kernel_outVector = tnlCuda::passToDevice( outVector );
            dim3 cudaBlockSize( matrix.getNumberOfChunksInSlice() ),
                 cudaGridSize( tnlCuda::getMaxGridSize() );
            const IndexType cudaBlocks = matrix.getNumberOfSlices();
            const IndexType cudaGrids = roundUpDivision( cudaBlocks, tnlCuda::getMaxGridSize() );
            const IndexType sharedMemory = cudaBlockSize.x * sizeof( RealType ) +
                                           sizeof( tnlChunkedEllpackSliceInfo< IndexType > );
            for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
            {
               if( gridIdx == cudaGrids - 1 )
                  cudaGridSize.x = cudaBlocks % tnlCuda::getMaxGridSize();
               tnlChunkedEllpackMatrixVectorProductCudaKernel< Real, Index, InVector, OutVector >
                                                             <<< cudaGridSize, cudaBlockSize, sharedMemory  >>>
                                                             ( kernel_this,
                                                               kernel_inVector,
                                                               kernel_outVector,
                                                               gridIdx );
            }
            tnlCuda::freeFromDevice( kernel_this );
            tnlCuda::freeFromDevice( kernel_inVector );
            tnlCuda::freeFromDevice( kernel_outVector );
            checkCudaDevice;
         #endif
      }

};

} // namespace TNL
