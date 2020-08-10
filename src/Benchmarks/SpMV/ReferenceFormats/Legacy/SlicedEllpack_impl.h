/***************************************************************************
                          SlicedSlicedEllpack_impl.h  -  description
                             -------------------
    begin                : Dec 8, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <Benchmarks/SpMV/ReferenceFormats/Legacy/SlicedEllpack.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Math.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Matrices {
   namespace Legacy {

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
SlicedEllpack< Real, Device, Index, SliceSize >::SlicedEllpack()
{
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
String SlicedEllpack< Real, Device, Index, SliceSize >::getSerializationType()
{
   return String( "Matrices::SlicedEllpack< ") +
          TNL::getType< Real >() +
          ", [any_device], " +
          String( TNL::getType< Index >() ) +
          String( " >" );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
String SlicedEllpack< Real, Device, Index, SliceSize >::getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpack< Real, Device, Index, SliceSize >::setDimensions( const IndexType rows,
                                                                     const IndexType columns )
{
   TNL_ASSERT( rows > 0 && columns > 0,
              std::cerr << "rows = " << rows
                   << " columns = " << columns << std::endl );
   Sparse< Real, Device, Index >::setDimensions( rows, columns );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpack< Real, Device, Index, SliceSize >::setCompressedRowLengths( ConstCompressedRowLengthsVectorView rowLengths )
{
   TNL_ASSERT_GT( this->getRows(), 0, "cannot set row lengths of an empty matrix" );
   TNL_ASSERT_GT( this->getColumns(), 0, "cannot set row lengths of an empty matrix" );
   TNL_ASSERT_EQ( this->getRows(), rowLengths.getSize(), "wrong size of the rowLengths vector" );

   const IndexType slices = roundUpDivision( this->rows, SliceSize );
   this->sliceCompressedRowLengths.setSize( slices );
   this->slicePointers.setSize( slices + 1 );

   DeviceDependentCode::computeMaximalRowLengthInSlices( *this, rowLengths );

   this->maxRowLength = max( rowLengths );

   this->slicePointers.template scan< Algorithms::ScanType::Exclusive >();
   this->allocateMatrixElements( this->slicePointers.getElement( slices ) );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpack< Real, Device, Index, SliceSize >::getCompressedRowLengths( CompressedRowLengthsVectorView rowLengths ) const
{
   TNL_ASSERT_EQ( rowLengths.getSize(), this->getRows(), "invalid size of the rowLengths vector" );
   for( IndexType row = 0; row < this->getRows(); row++ )
      rowLengths.setElement( row, this->getRowLength( row ) );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
Index SlicedEllpack< Real, Device, Index, SliceSize >::getRowLength( const IndexType row ) const
{
   const IndexType slice = row / SliceSize;
   return this->sliceCompressedRowLengths.getElement( slice );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
__cuda_callable__
Index SlicedEllpack< Real, Device, Index, SliceSize >::getRowLengthFast( const IndexType row ) const
{
   const IndexType slice = row / SliceSize;
   return this->sliceCompressedRowLengths[ slice ];
}

template< typename Real,
          typename Device,
          typename Index ,
          int SliceSize >
Index SlicedEllpack< Real, Device, Index, SliceSize >::getNonZeroRowLength( const IndexType row ) const
{
    ConstMatrixRow matrixRow = getRow( row );
    return matrixRow.getNonZeroElementsCount( getType< Device >() );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Real2,
             typename Device2,
             typename Index2 >
void SlicedEllpack< Real, Device, Index, SliceSize >::setLike( const SlicedEllpack< Real2, Device2, Index2, SliceSize >& matrix )
{
   Sparse< Real, Device, Index >::setLike( matrix );
   this->slicePointers.setLike( matrix.slicePointers );
   this->sliceCompressedRowLengths.setLike( matrix.sliceCompressedRowLengths );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpack< Real, Device, Index, SliceSize >::reset()
{
   Sparse< Real, Device, Index >::reset();
   this->slicePointers.reset();
   this->sliceCompressedRowLengths.reset();
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool SlicedEllpack< Real, Device, Index, SliceSize >::operator == ( const SlicedEllpack< Real2, Device2, Index2 >& matrix ) const
{
   TNL_ASSERT( this->getRows() == matrix.getRows() &&
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
          typename Index,
          int SliceSize >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool SlicedEllpack< Real, Device, Index, SliceSize >::operator != ( const SlicedEllpack< Real2, Device2, Index2 >& matrix ) const
{
   return ! ( ( *this ) == matrix );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
__cuda_callable__
bool SlicedEllpack< Real, Device, Index, SliceSize >::setElementFast( const IndexType row,
                                                                               const IndexType column,
                                                                               const Real& value )
{
   return this->addElementFast( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool SlicedEllpack< Real, Device, Index, SliceSize >::setElement( const IndexType row,
                                                                           const IndexType column,
                                                                           const Real& value )
{
   return this->addElement( row, column, value, 0.0 );
}


template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
__cuda_callable__
bool SlicedEllpack< Real, Device, Index, SliceSize >::addElementFast( const IndexType row,
                                                                               const IndexType column,
                                                                               const RealType& value,
                                                                               const RealType& thisElementMultiplicator )
{
   TNL_ASSERT( row >= 0 && row < this->rows &&
              column >= 0 && column <= this->columns,
              std::cerr << " row = " << row
                   << " column = " << column
                   << " this->rows = " << this->rows
                   << " this->columns = " << this-> columns );

   Index elementPtr, rowEnd, step;
   DeviceDependentCode::initRowTraverseFast( *this, row, elementPtr, rowEnd, step );

   IndexType col = 0;
   while( elementPtr < rowEnd &&
          ( col = this->columnIndexes[ elementPtr ] ) < column &&
          col != this->getPaddingIndex() ) elementPtr += step;
   if( elementPtr == rowEnd )
      return false;
   if( col == column )
   {
      this->values[ elementPtr ] = thisElementMultiplicator * this->values[ elementPtr ] + value;
      return true;
   }
   if( col == this->getPaddingIndex() )
   {
      this->columnIndexes[ elementPtr ] = column;
      this->values[ elementPtr ] = value;
      return true;
   }
   IndexType j = rowEnd - step;
   while( j > elementPtr )
   {
      this->columnIndexes[ j ] = this->columnIndexes[ j - step ];
      this->values[ j ] = this->values[ j - step ];
      j -= step;
   }
   this->columnIndexes[ elementPtr ] = column;
   this->values[ elementPtr ] = value;
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool SlicedEllpack< Real, Device, Index, SliceSize >::addElement( const IndexType row,
                                                                           const IndexType column,
                                                                           const RealType& value,
                                                                           const RealType& thisElementMultiplicator )
{
   TNL_ASSERT( row >= 0 && row < this->rows &&
              column >= 0 && column <= this->columns,
              std::cerr << " row = " << row
                   << " column = " << column
                   << " this->rows = " << this->rows
                   << " this->columns = " << this-> columns );

   Index elementPtr, rowEnd, step;
   DeviceDependentCode::initRowTraverse( *this, row, elementPtr, rowEnd, step );

   IndexType col = 0;
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
__cuda_callable__
bool SlicedEllpack< Real, Device, Index, SliceSize > :: setRowFast( const IndexType row,
                                                                             const IndexType* columnIndexes,
                                                                             const RealType* values,
                                                                             const IndexType elements )
{
   const IndexType sliceIdx = row / SliceSize;
   const IndexType rowLength = this->sliceCompressedRowLengths[ sliceIdx ];
   if( elements > rowLength )
      return false;

   Index elementPointer, rowEnd, step;
   DeviceDependentCode::initRowTraverseFast( *this, row, elementPointer, rowEnd, step );

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
bool SlicedEllpack< Real, Device, Index, SliceSize > :: setRow( const IndexType row,
                                                                         const IndexType* columnIndexes,
                                                                         const RealType* values,
                                                                         const IndexType elements )
{
   const IndexType sliceIdx = row / SliceSize;
   const IndexType rowLength = this->sliceCompressedRowLengths.getElement( sliceIdx );
   if( elements > rowLength )
      return false;

   Index elementPointer, rowEnd, step;
   DeviceDependentCode::initRowTraverse( *this, row, elementPointer, rowEnd, step );

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
__cuda_callable__
bool SlicedEllpack< Real, Device, Index, SliceSize > :: addRowFast( const IndexType row,
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
bool SlicedEllpack< Real, Device, Index, SliceSize > :: addRow( const IndexType row,
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
__cuda_callable__
Real SlicedEllpack< Real, Device, Index, SliceSize >::getElementFast( const IndexType row,
                                                                               const IndexType column ) const
{
   Index elementPtr, rowEnd, step;
   DeviceDependentCode::initRowTraverseFast( *this, row, elementPtr, rowEnd, step );

   IndexType col = 0;
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
Real SlicedEllpack< Real, Device, Index, SliceSize >::getElement( const IndexType row,
                                                                           const IndexType column ) const
{

   Index elementPtr, rowEnd, step;
   DeviceDependentCode::initRowTraverse( *this, row, elementPtr, rowEnd, step );

   IndexType col = 0;
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
__cuda_callable__
void SlicedEllpack< Real, Device, Index, SliceSize >::getRowFast( const IndexType row,
                                                                           IndexType* columns,
                                                                           RealType* values ) const
{
   Index elementPtr, rowEnd, step, i( 0 );
   DeviceDependentCode::initRowTraverseFast( *this, row, elementPtr, rowEnd, step );

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
__cuda_callable__
typename SlicedEllpack< Real, Device, Index, SliceSize >::MatrixRow
SlicedEllpack< Real, Device, Index, SliceSize >::
getRow( const IndexType rowIndex )
{
   Index rowBegin, rowEnd, step;
   DeviceDependentCode::initRowTraverseFast( *this, rowIndex, rowBegin, rowEnd, step );
   const IndexType slice = rowIndex / SliceSize;
   return MatrixRow( &this->columnIndexes[ rowBegin ],
                     &this->values[ rowBegin ],
                     this->sliceCompressedRowLengths[ slice ],
                     step );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
__cuda_callable__
typename SlicedEllpack< Real, Device, Index, SliceSize >::ConstMatrixRow
SlicedEllpack< Real, Device, Index, SliceSize >::
getRow( const IndexType rowIndex ) const
{
   Index rowBegin, rowEnd, step;
   DeviceDependentCode::initRowTraverseFast( *this, rowIndex, rowBegin, rowEnd, step );
   const IndexType slice = rowIndex / SliceSize;
   return ConstMatrixRow( &this->columnIndexes[ rowBegin ],
                          &this->values[ rowBegin ],
                          this->sliceCompressedRowLengths[ slice ],
                          step );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
  template< typename Vector >
__cuda_callable__
typename Vector::RealType SlicedEllpack< Real, Device, Index, SliceSize >::rowVectorProduct( const IndexType row,
                                                                                                      const Vector& vector ) const
{
   Real result = 0.0;
   Index elementPtr, rowEnd, step;
   DeviceDependentCode::initRowTraverseFast( *this, row, elementPtr, rowEnd, step );

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
void SlicedEllpack< Real, Device, Index, SliceSize >::vectorProduct( const InVector& inVector,
                                                                     OutVector& outVector,
                                                                     RealType multiplicator ) const
{
   DeviceDependentCode::vectorProduct( *this, inVector, outVector, multiplicator );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Real2,
             typename Index2 >
void SlicedEllpack< Real, Device, Index, SliceSize >::addMatrix( const SlicedEllpack< Real2, Device, Index2 >& matrix,
                                                                          const RealType& matrixMultiplicator,
                                                                          const RealType& thisMatrixMultiplicator )
{
   throw Exceptions::NotImplementedError( "SlicedEllpack::addMatrix is not implemented." );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Real2,
             typename Index2 >
void SlicedEllpack< Real, Device, Index, SliceSize >::getTransposition( const SlicedEllpack< Real2, Device, Index2 >& matrix,
                                                                      const RealType& matrixMultiplicator )
{
   throw Exceptions::NotImplementedError( "SlicedEllpack::getTransposition is not implemented." );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Vector1, typename Vector2 >
bool SlicedEllpack< Real, Device, Index, SliceSize >::performSORIteration( const Vector1& b,
                                                                           const IndexType row,
                                                                           Vector2& x,
                                                                           const RealType& omega ) const
{
   TNL_ASSERT( row >=0 && row < this->getRows(),
              std::cerr << "row = " << row
                   << " this->getRows() = " << this->getRows() << std::endl );

   RealType diagonalValue( 0.0 );
   RealType sum( 0.0 );

   /*const IndexType sliceIdx = row / SliceSize;
   const IndexType rowLength = this->sliceCompressedRowLengths[ sliceIdx ];
   IndexType elementPtr = this->slicePointers[ sliceIdx ] +
                          rowLength * ( row - sliceIdx * SliceSize );
   const IndexType rowEnd( elementPtr + rowLength );*/
   IndexType elementPtr, rowEnd, step;
   DeviceDependentCode::initRowTraverseFast( *this, row, elementPtr, rowEnd, step );
   IndexType column;
   while( elementPtr < rowEnd && ( column = this->columnIndexes[ elementPtr ] ) < this->columns )
   {
      if( column == row )
         diagonalValue = this->values[  elementPtr ];
      else
         sum += this->values[ elementPtr ] * x[ column ];
      elementPtr += step;
   }
   if( diagonalValue == ( Real ) 0.0 )
   {
      std::cerr << "There is zero on the diagonal in " << row << "-th row of a matrix. I cannot perform SOR iteration." << std::endl;
      return false;
   }
   x[ row ] = ( 1.0 - omega ) * x[ row ] + omega / diagonalValue * ( b[ row ] - sum );
   return true;
}


// copy assignment
template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
SlicedEllpack< Real, Device, Index, SliceSize >&
SlicedEllpack< Real, Device, Index, SliceSize >::operator=( const SlicedEllpack& matrix )
{
   this->setLike( matrix );
   this->values = matrix.values;
   this->columnIndexes = matrix.columnIndexes;
   this->slicePointers = matrix.slicePointers;
   this->sliceCompressedRowLengths = matrix.sliceCompressedRowLengths;
   return *this;
}

// cross-device copy assignment
template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Real2, typename Device2, typename Index2, typename >
SlicedEllpack< Real, Device, Index, SliceSize >&
SlicedEllpack< Real, Device, Index, SliceSize >::operator=( const SlicedEllpack< Real2, Device2, Index2, SliceSize >& matrix )
{
   this->setLike( matrix );
   this->slicePointers = matrix.slicePointers;
   this->sliceCompressedRowLengths = matrix.sliceCompressedRowLengths;

   // host -> cuda
   if( std::is_same< Device, Devices::Cuda >::value ) {
      typename ValuesVector::template Self< typename ValuesVector::ValueType, Devices::Sequential > tmpValues;
      typename ColumnIndexesVector::template Self< typename ColumnIndexesVector::ValueType, Devices::Sequential > tmpColumnIndexes;
      tmpValues.setLike( matrix.values );
      tmpColumnIndexes.setLike( matrix.columnIndexes );

#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
      for( Index sliceIdx = 0; sliceIdx < matrix.sliceCompressedRowLengths.getSize(); sliceIdx++ ) {
         const Index rowLength = matrix.sliceCompressedRowLengths[ sliceIdx ];
         const Index offset = matrix.slicePointers[ sliceIdx ];
         for( Index j = 0; j < rowLength; j++ )
            for( Index i = 0; i < SliceSize; i++ ) {
               tmpValues[ offset + j * SliceSize + i ] = matrix.values[ offset + i * rowLength + j ];
               tmpColumnIndexes[ offset + j * SliceSize + i ] = matrix.columnIndexes[ offset + i * rowLength + j ];
            }
      }

      this->values = tmpValues;
      this->columnIndexes = tmpColumnIndexes;
   }

   // cuda -> host
   else {
      ValuesVector tmpValues;
      ColumnIndexesVector tmpColumnIndexes;
      tmpValues.setLike( matrix.values );
      tmpColumnIndexes.setLike( matrix.columnIndexes );
      tmpValues = matrix.values;
      tmpColumnIndexes = matrix.columnIndexes;

#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
      for( Index sliceIdx = 0; sliceIdx < sliceCompressedRowLengths.getSize(); sliceIdx++ ) {
         const Index rowLength = sliceCompressedRowLengths[ sliceIdx ];
         const Index offset = slicePointers[ sliceIdx ];
         for( Index i = 0; i < SliceSize; i++ )
            for( Index j = 0; j < rowLength; j++ ) {
               this->values[ offset + i * rowLength + j ] = tmpValues[ offset + j * SliceSize + i ];
               this->columnIndexes[ offset + i * rowLength + j ] = tmpColumnIndexes[ offset + j * SliceSize + i ];
            }
      }
   }

   return *this;
}


template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpack< Real, Device, Index, SliceSize >::save( File& file ) const
{
   Sparse< Real, Device, Index >::save( file );
   file << this->slicePointers << this->sliceCompressedRowLengths;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpack< Real, Device, Index, SliceSize >::load( File& file )
{
   Sparse< Real, Device, Index >::load( file );
   file >> this->slicePointers >> this->sliceCompressedRowLengths;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpack< Real, Device, Index, SliceSize >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpack< Real, Device, Index, SliceSize >::load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpack< Real, Device, Index, SliceSize >::print( std::ostream& str ) const
{
   if( ! std::is_same< Device, Devices::Cuda >::value ) {
      for( IndexType row = 0; row < this->getRows(); row++ )
      {
         str <<"Row: " << row << " -> ";
         const IndexType sliceIdx = row / SliceSize;
         const IndexType rowLength = this->sliceCompressedRowLengths.getElement( sliceIdx );
         IndexType elementPtr = this->slicePointers.getElement( sliceIdx ) +
                                rowLength * ( row - sliceIdx * SliceSize );
         const IndexType rowEnd( elementPtr + rowLength );
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
   else {
      Self< Real, Devices::Sequential > hostMatrix;
      hostMatrix = *this;
      hostMatrix.print( str );
   }
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
__device__ void SlicedEllpack< Real, Device, Index, SliceSize >::computeMaximalRowLengthInSlicesCuda( ConstCompressedRowLengthsVectorView rowLengths,
                                                                                                      const IndexType sliceIdx )
{
   Index rowIdx = sliceIdx * SliceSize;
   Index rowInSliceIdx( 0 );
   Index maxRowLength( 0 );
   if( rowIdx >= this->getRows() )
      return;
   while( rowInSliceIdx < SliceSize && rowIdx < this->getRows() )
   {
      maxRowLength = max( maxRowLength, rowLengths[ rowIdx ] );
      rowIdx++;
      rowInSliceIdx++;
   }
   this->sliceCompressedRowLengths[ sliceIdx ] = maxRowLength;
   this->slicePointers[ sliceIdx ] = maxRowLength * SliceSize;
   if( threadIdx.x == 0 )
      this->slicePointers[ this->slicePointers.getSize() - 1 ] = 0;

}
#endif

// implementation for host types
template< typename Device_ >
class SlicedEllpackDeviceDependentCode
{
   public:

      typedef Device_ Device;

      template< typename Real,
                typename Index,
                int SliceSize >
      static void initRowTraverse( const SlicedEllpack< Real, Device, Index, SliceSize >& matrix,
                                   const Index row,
                                   Index& rowBegin,
                                   Index& rowEnd,
                                   Index& step )
      {
         const Index sliceIdx = row / SliceSize;
         const Index slicePointer = matrix.slicePointers.getElement( sliceIdx );
         const Index rowLength = matrix.sliceCompressedRowLengths.getElement( sliceIdx );

         rowBegin = slicePointer + rowLength * ( row - sliceIdx * SliceSize );
         rowEnd = rowBegin + rowLength;
         step = 1;
      }

      template< typename Real,
                typename Index,
                int SliceSize >
      __cuda_callable__
      static void initRowTraverseFast( const SlicedEllpack< Real, Device, Index, SliceSize >& matrix,
                                       const Index row,
                                       Index& rowBegin,
                                       Index& rowEnd,
                                       Index& step )
      {
         const Index sliceIdx = row / SliceSize;
         const Index slicePointer = matrix.slicePointers[ sliceIdx ];
         const Index rowLength = matrix.sliceCompressedRowLengths[ sliceIdx ];

         rowBegin = slicePointer + rowLength * ( row - sliceIdx * SliceSize );
         rowEnd = rowBegin + rowLength;
         step = 1;
      }


      template< typename Real,
                typename Index,
                int SliceSize >
      static bool computeMaximalRowLengthInSlices( SlicedEllpack< Real, Device, Index, SliceSize >& matrix,
                                                   typename SlicedEllpack< Real, Device, Index >::ConstCompressedRowLengthsVectorView rowLengths )
      {
         Index row( 0 ), slice( 0 ), sliceRowLength( 0 );
         while( row < matrix.getRows() )
         {
            sliceRowLength = max( rowLengths.getElement( row++ ), sliceRowLength );
            if( row % SliceSize == 0 )
            {
               matrix.sliceCompressedRowLengths.setElement( slice, sliceRowLength );
               matrix.slicePointers.setElement( slice++, sliceRowLength * SliceSize );
               sliceRowLength = 0;
            }
         }
         if( row % SliceSize != 0 )
         {
            matrix.sliceCompressedRowLengths.setElement( slice, sliceRowLength );
            matrix.slicePointers.setElement( slice++, sliceRowLength * SliceSize );
         }
         matrix.slicePointers.setElement( matrix.slicePointers.getSize() - 1, 0 );
         return true;
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector,
                int SliceSize >
      static void vectorProduct( const SlicedEllpack< Real, Device, Index, SliceSize >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector,
                                 Real multiplicator )
      {
#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
         for( Index row = 0; row < matrix.getRows(); row ++ )
            outVector[ row ] = matrix.rowVectorProduct( row, inVector ) * multiplicator;
      }

};

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          int SliceSize >
__global__ void SlicedEllpack_computeMaximalRowLengthInSlices_CudaKernel( SlicedEllpack< Real, Devices::Cuda, Index, SliceSize >* matrix,
                                                                          typename SlicedEllpack< Real, Devices::Cuda, Index, SliceSize >::ConstCompressedRowLengthsVectorView rowLengths,
                                                                          int gridIdx )
{
   const Index sliceIdx = gridIdx * Cuda::getMaxGridSize() * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
   matrix->computeMaximalRowLengthInSlicesCuda( rowLengths, sliceIdx );
}
#endif

#ifdef HAVE_CUDA
template<
   typename Real,
   typename Index,
   int SliceSize >
__global__ void SlicedEllpackVectorProductCudaKernel(
   const Index rows,
   const Index columns,
   const Index* slicePointers,
   const Index* sliceCompressedRowLengths,
   const Index paddingIndex,
   const Index* columnIndexes,
   const Real* values,
   const Real* inVector,
   Real* outVector,
   Real multiplicator,
   const Index gridIdx )
{
   const Index rowIdx = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( rowIdx >= rows )
      return;
   const Index sliceIdx = rowIdx / SliceSize;
   const Index slicePointer = slicePointers[ sliceIdx ];
   const Index rowLength = sliceCompressedRowLengths[ sliceIdx ];
   Index i = slicePointer + rowIdx - sliceIdx * SliceSize;
   const Index rowEnd = i + rowLength * SliceSize;
   Real result( 0.0 );
   Index columnIndex;
   while( i < rowEnd &&
         ( columnIndex = columnIndexes[ i ] ) < columns &&
         columnIndex < paddingIndex )
   {
      result += values[ i ] * inVector[ columnIndex ];
      i += SliceSize;
   }
   outVector[ rowIdx ] = result * multiplicator;
}
#endif


template<>
class SlicedEllpackDeviceDependentCode< Devices::Cuda >
{
   public:

      typedef Devices::Cuda Device;

      template< typename Real,
                typename Index,
                int SliceSize >
      static void initRowTraverse( const SlicedEllpack< Real, Device, Index, SliceSize >& matrix,
                                   const Index row,
                                   Index& rowBegin,
                                   Index& rowEnd,
                                   Index& step )
      {
         const Index sliceIdx = row / SliceSize;
         const Index slicePointer = matrix.slicePointers.getElement( sliceIdx );
         const Index rowLength = matrix.sliceCompressedRowLengths.getElement( sliceIdx );

         rowBegin = slicePointer + row - sliceIdx * SliceSize;
         rowEnd = rowBegin + rowLength * SliceSize;
         step = SliceSize;
      }

      template< typename Real,
                typename Index,
                int SliceSize >
      __cuda_callable__
      static void initRowTraverseFast( const SlicedEllpack< Real, Device, Index, SliceSize >& matrix,
                                       const Index row,
                                       Index& rowBegin,
                                       Index& rowEnd,
                                       Index& step )
      {
         const Index sliceIdx = row / SliceSize;
         const Index slicePointer = matrix.slicePointers[ sliceIdx ];
         const Index rowLength = matrix.sliceCompressedRowLengths[ sliceIdx ];

         rowBegin = slicePointer + row - sliceIdx * SliceSize;
         rowEnd = rowBegin + rowLength * SliceSize;
         step = SliceSize;

      }

      template< typename Real,
                typename Index,
                int SliceSize >
      static bool computeMaximalRowLengthInSlices( SlicedEllpack< Real, Device, Index, SliceSize >& matrix,
                                                   typename SlicedEllpack< Real, Device, Index >::ConstCompressedRowLengthsVectorView rowLengths )
      {
#ifdef HAVE_CUDA
         typedef SlicedEllpack< Real, Device, Index, SliceSize > Matrix;
         typedef typename Matrix::CompressedRowLengthsVector CompressedRowLengthsVector;
         Matrix* kernel_matrix = Cuda::passToDevice( matrix );
         const Index numberOfSlices = roundUpDivision( matrix.getRows(), SliceSize );
         dim3 cudaBlockSize( 256 ), cudaGridSize( Cuda::getMaxGridSize() );
         const Index cudaBlocks = roundUpDivision( numberOfSlices, cudaBlockSize.x );
         const Index cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridSize() );
         for( int gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
         {
            if( gridIdx == cudaGrids - 1 )
               cudaGridSize.x = cudaBlocks % Cuda::getMaxGridSize();
            SlicedEllpack_computeMaximalRowLengthInSlices_CudaKernel< Real, Index, SliceSize ><<< cudaGridSize, cudaBlockSize >>>
                                                                             ( kernel_matrix,
                                                                               rowLengths,
                                                                               gridIdx );
         }
         Cuda::freeFromDevice( kernel_matrix );
         TNL_CHECK_CUDA_DEVICE;
#endif
         return true;
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector,
                int SliceSize >
      static void vectorProduct( const SlicedEllpack< Real, Device, Index, SliceSize >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector,
                                 Real multiplicator )
      {
         //MatrixVectorProductCuda( matrix, inVector, outVector );
         #ifdef HAVE_CUDA
            typedef SlicedEllpack< Real, Device, Index, SliceSize > Matrix;
            typedef typename Matrix::IndexType IndexType;
            //Matrix* kernel_this = Cuda::passToDevice( matrix );
            //InVector* kernel_inVector = Cuda::passToDevice( inVector );
            //OutVector* kernel_outVector = Cuda::passToDevice( outVector );
            dim3 cudaBlockSize( 256 ), cudaGridSize( Cuda::getMaxGridSize() );
            const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
            const IndexType cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridSize() );
            for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
            {
               if( gridIdx == cudaGrids - 1 )
                  cudaGridSize.x = cudaBlocks % Cuda::getMaxGridSize();
               SlicedEllpackVectorProductCudaKernel
               < Real, Index, SliceSize >
                <<< cudaGridSize, cudaBlockSize >>>
                ( matrix.getRows(),
                  matrix.getColumns(),
                  matrix.slicePointers.getData(),
                  matrix.sliceCompressedRowLengths.getData(),
                  matrix.getPaddingIndex(),
                  matrix.columnIndexes.getData(),
                  matrix.values.getData(),
                  inVector.getData(),
                  outVector.getData(),
                  multiplicator,
                  gridIdx );
               TNL_CHECK_CUDA_DEVICE;
            }
            //Cuda::freeFromDevice( kernel_this );
            //Cuda::freeFromDevice( kernel_inVector );
            //Cuda::freeFromDevice( kernel_outVector );
            TNL_CHECK_CUDA_DEVICE;
            cudaDeviceSynchronize();
         #endif
      }
};

} //namespace Legacy
} // namespace Matrices
} // namespace TNL
