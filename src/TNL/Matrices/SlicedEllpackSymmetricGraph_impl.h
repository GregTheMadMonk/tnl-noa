/***************************************************************************
                          SlicedEllpackSymmetricGraph_impl.h  -  description
                             -------------------
    begin                : Aug 30, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/SlicedEllpackSymmetricGraph.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Math.h>

namespace TNL {
namespace Matrices {

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::SlicedEllpackSymmetricGraph()
: rearranged( false )
{
};

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
String SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::getType()
{
   return String( "SlicedEllpackSymmetricGraph< ") +
          String( TNL::getType< Real >() ) +
          String( ", " ) +
          Device::getDeviceType() +
          String( " >" );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
String SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::setDimensions( const IndexType rows,
                                                                                   const IndexType columns )
{
   TNL_ASSERT( rows > 0 && columns > 0,
             std::cerr << "rows = " << rows
                   << " columns = " << columns <<std::endl );
   Sparse< Real, Device, Index >::setDimensions( rows, columns );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::setCompressedRowLengths( const CompressedRowLengthsVector& rowLengths )
{
   TNL_ASSERT( this->getRows() > 0, );
   TNL_ASSERT( this->getColumns() > 0, );
   const IndexType slices = roundUpDivision( this->rows, SliceSize );
   this->sliceRowLengths.setSize( slices );
   this->slicePointers.setSize( slices + 1 );

   this->permutationArray.setSize( this->getRows() );
   for( IndexType i = 0; i < this->getRows(); i++ )
      this->permutationArray.setElement( i, i );

   Containers::Vector< Index, Device, Index > sliceRowLengths, slicePointers;
   sliceRowLengths.setSize( slices );
   slicePointers.setSize( slices + 1 );
   // TODO: fix this
   //DeviceDependentCode::computeMaximalRowLengthInSlices( *this, rowLengths, sliceRowLengths, slicePointers );
   this->sliceRowLengths = sliceRowLengths;
   this->slicePointers = slicePointers;

   this->maxRowLength = rowLengths.max();

   this->slicePointers.computeExclusivePrefixSum();
   this->allocateMatrixElements( this->slicePointers.getElement( slices ) );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
Index SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::getRowLength( const IndexType row ) const
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
bool SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::setLike( const SlicedEllpackSymmetricGraph< Real2, Device2, Index2, SliceSize >& matrix )
{
   if( !Sparse< Real, Device, Index >::setLike( matrix ) ||
       ! this->slicePointers.setLike( matrix.slicePointers ) ||
       ! this->sliceRowLengths.setLike( matrix.sliceRowLengths ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::reset()
{
   Sparse< Real, Device, Index >::reset();
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
bool SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::operator == ( const SlicedEllpackSymmetricGraph< Real2, Device2, Index2 >& matrix ) const
{
   TNL_ASSERT( this->getRows() == matrix.getRows() &&
              this->getColumns() == matrix.getColumns(),
             std::cerr << "this->getRows() = " << this->getRows()
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
bool SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::operator != ( const SlicedEllpackSymmetricGraph< Real2, Device2, Index2 >& matrix ) const
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
bool SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::setElementFast( const IndexType row,
                                                                                    const IndexType column,
                                                                                    const Real& value )
{
   return this->addElementFast( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::setElement( const IndexType row,
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
bool SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::addElementFast( const IndexType row,
                                                                                    const IndexType column,
                                                                                    const RealType& value,
                                                                                    const RealType& thisElementMultiplicator )
{
   TNL_ASSERT( row >= 0 && row < this->rows &&
              column >= 0 && column <= this->rows,
             std::cerr << " row = " << row
                   << " column = " << column
                   << " this->rows = " << this->rows
                   << " this->columns = " << this-> columns );

   Index elementPtr, rowEnd, step;
   DeviceDependentCode::initRowTraverseFast( *this, row, elementPtr, rowEnd, step );

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
bool SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::addElement( const IndexType row,
                                                                                const IndexType column,
                                                                                const RealType& value,
                                                                                const RealType& thisElementMultiplicator )
{
   TNL_ASSERT( row >= 0 && row < this->rows &&
              column >= 0 && column <= this->rows,
             std::cerr << " row = " << row
                   << " column = " << column
                   << " this->rows = " << this->rows
                   << " this->columns = " << this-> columns );

   Index elementPtr, rowEnd, step;
   DeviceDependentCode::initRowTraverse( *this, row, elementPtr, rowEnd, step );

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
bool SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize > :: setRowFast( const IndexType row,
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
bool SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize > :: setRow( const IndexType row,
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
bool SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize > :: addRowFast( const IndexType row,
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
bool SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize > :: addRow( const IndexType row,
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
Real SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::getElementFast( const IndexType row,
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
Real SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::getElement( const IndexType row,
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
void SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::getRowFast( const IndexType row,
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
void SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::getRow( const IndexType row,
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
typename Vector::RealType SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::rowVectorProduct( const IndexType row,
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
void SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::vectorProduct( const InVector& inVector,
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
void SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::addMatrix( const SlicedEllpackSymmetricGraph< Real2, Device, Index2 >& matrix,
                                                                               const RealType& matrixMultiplicator,
                                                                               const RealType& thisMatrixMultiplicator )
{
   TNL_ASSERT( false,std::cerr << "TODO: implement" );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Real2,
             typename Index2 >
void SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::getTransposition( const SlicedEllpackSymmetricGraph< Real2, Device, Index2 >& matrix,
                                                                                      const RealType& matrixMultiplicator )
{
   TNL_ASSERT( false,std::cerr << "TODO: implement" );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Vector >
bool SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::performSORIteration( const Vector& b,
                                                                                         const IndexType row,
                                                                                         Vector& x,
                                                                                         const RealType& omega ) const
{
   TNL_ASSERT( row >=0 && row < this->getRows(),
             std::cerr << "row = " << row
                   << " this->getRows() = " << this->getRows()
                   << " this->getName() = " << this->getName() <<std::endl );

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
     std::cerr << "There is zero on the diagonal in " << this->permutationArray.getElement( row ) << "-th row of thge matrix " << this->getName() << ". I cannot perform SOR iteration." <<std::endl;
      return false;
   }
   x. setElement( this->permutationArray.getElement( row ), x[ this->permutationArray.getElement( row ) ] + omega / diagonalValue * ( b[ this->permutationArray.getElement( row ) ] - sum ) );
   return true;
}


template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::save( File& file ) const
{
   if( ! Sparse< Real, Device, Index >::save( file ) ||
       ! this->slicePointers.save( file ) ||
       ! this->sliceRowLengths.save( file ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::load( File& file )
{
   if( ! Sparse< Real, Device, Index >::load( file ) ||
       ! this->slicePointers.load( file ) ||
       ! this->sliceRowLengths.load( file ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::save( const String& fileName ) const
{
   return Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::load( const String& fileName )
{
   return Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::print( std::ostream& str ) const
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
      str <<std::endl;
   }
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::computePermutationArray()
{
    Containers::Vector< Index, Device, Index > colorsVector;
    colorsVector.setSize( this->getRows() );
    for( IndexType i = 0; i < this->getRows(); i++ )
    {
        colorsVector.setElement( i, 0 );
    }

    // compute colors for each row
    Matrix< Real, Device, Index >::computeColorsVector( colorsVector );

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
Index SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::getRealRowLength( const Index row )
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
Containers::Vector< Index, Device, Index > SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::getRealRowLengths()
{
   Containers::Vector< Index, Device, Index > rowLengths;
   rowLengths.setSize( this->getRows() );
   for( IndexType row = 0; row < this->getRows(); row++ )
      rowLengths.setElement( row, this->getRealRowLength( row ) );

   return rowLengths;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::rearrangeMatrix( bool verbose )
{
    this->computePermutationArray();

    // now based on new permutation array we need to recompute row lengths in slices
    const IndexType slices = roundUpDivision( this->rows, SliceSize );
    Containers::Vector< Index, Device, Index > sliceRowLengths, slicePointers, rowLengths;
    sliceRowLengths.setSize( slices );
    slicePointers.setSize( slices + 1 );
    rowLengths.setSize( this->getRows() );
    rowLengths = this->getRealRowLengths();
    // TODO: fix this
    //DeviceDependentCode::computeMaximalRowLengthInSlices( *this, rowLengths, sliceRowLengths, slicePointers );

    slicePointers.computeExclusivePrefixSum();

    // this->testRowLengths( rowLengths, sliceRowLengths );

    // return this->allocateMatrixElements( this->slicePointers.getElement( slices ) );
    Containers::Vector< Real, Device, Index > valuesVector;
    Containers::Vector< Index, Device, Index > columnsVector;
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
bool SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::help( bool verbose )
{
    if( !this->rearranged )
        this->rearrangeMatrix( verbose );
    return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
Containers::Vector< Index, Device, Index > SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::getSlicePointers()
{
    return this->slicePointers;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
Containers::Vector< Index, Device, Index > SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::getSliceRowLengths()
{
    return this->sliceRowLengths;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
Containers::Vector< Index, Device, Index > SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::getPermutationArray()
{
    return this->permutationArray;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
Containers::Vector< Index, Device, Index > SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::getInversePermutationArray()
{
    return this->inversePermutationArray;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
Containers::Vector< Index, Device, Index > SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::getColorPointers()
{
    return this->colorPointers;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::copyFromHostToCuda( SlicedEllpackSymmetricGraph<Real, Devices::Host, Index, SliceSize>& matrix )
{
    Sparse< Real, Device, Index >::copyFromHostToCuda( matrix );

    this->rearranged = true;

    Containers::Vector< Index, Device, Index > colorPointers = matrix.getColorPointers();
    this->colorPointers.setSize( colorPointers.getSize() );
    for( IndexType i = 0; i < colorPointers.getSize(); i++ )
        this->colorPointers.setElement( i, colorPointers[ i ] );

    Containers::Vector< Index, Device, Index > slicePointers = matrix.getSlicePointers();
    this->slicePointers.setSize( slicePointers.getSize() );
    for( IndexType i = 0; i < slicePointers.getSize(); i++ )
        this->slicePointers.setElement( i, slicePointers[ i ] );

    Containers::Vector< Index, Device, Index > sliceRowLengths = matrix.getSliceRowLengths();
    this->sliceRowLengths.setSize( sliceRowLengths.getSize() );
    for( IndexType i = 0; i < sliceRowLengths.getSize(); i++ )
        this->sliceRowLengths.setElement( i, sliceRowLengths[ i ] );

    Containers::Vector< Index, Device, Index > permutationArray = matrix.getPermutationArray();
    this->permutationArray.setSize( permutationArray.getSize() );
    for( IndexType i = 0; i < permutationArray.getSize(); i++ )
        this->permutationArray.setElement( i, permutationArray[ i ] );

    Containers::Vector< Index, Device, Index > inversePermutation = matrix.getInversePermutationArray();
    this->inversePermutationArray.setSize( inversePermutation.getize() );
    for( IndexType i = 0; i < inversePermutation.getSize(); i++ )
        this->inversePermutationArray.setElement( i, inversePermutation[ i ] );

    for( IndexType i = 0; i < this->getRows(); i++ )
        for( IndexType j = 0; j <= i; j++ )
        {
            if( matrix.getElement( i, j ) != 0.0 )
                this->setElementFast( i, j, matrix.getElement( i, j ) );
        }

    colorPointers.reset();
    slicePointers.reset();
    sliceRowLengths.reset();
    permutationArray.reset();
    inversePermutation.reset();
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
template< typename InVector,
          typename OutVector >
void SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::vectorProductHost( const InVector& inVector,
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
__device__ void SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::computeMaximalRowLengthInSlicesCuda( const CompressedRowLengthsVector& rowLengths,
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
class SlicedEllpackSymmetricGraphDeviceDependentCode< Devices::Host >
{
   public:

      typedef Devices::Host Device;

      template< typename Real,
                typename Index,
                int SliceSize >
      static void initRowTraverse( const SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >& matrix,
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
      static void initRowTraverseFast( const SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >& matrix,
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
      static bool computeMaximalRowLengthInSlices( SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >& matrix,
                                                   const typename SlicedEllpackSymmetricGraph< Real, Device, Index >::RowLengthsVector& rowLengths,
                                                   Containers::Vector< Index, Device, Index >& sliceRowLengths,
                                                   Containers::Vector< Index, Device, Index >& slicePointers )
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
         Containers::Vector< Index, Device, Index > rowMapToSlice;
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
      static void vectorProduct( const SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >& matrix,
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
__global__ void SlicedEllpackSymmetricGraph_computeMaximalRowLengthInSlices_CudaKernel( SlicedEllpack< Real, Devices::Cuda, Index, SliceSize >* matrix,
                                                                                   const typename SlicedEllpackSymmetricGraph< Real, Devices::Cuda, Index, SliceSize >::RowLengthsVector* rowLengths,
                                                                                   int gridIdx )
{
   const Index sliceIdx = gridIdx * Devices::Cuda::getMaxGridSize() * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
   matrix->computeMaximalRowLengthInSlicesCuda( *rowLengths, sliceIdx );
}
#endif

#ifdef HAVE_CUDA
template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
template< typename InVector,
          typename OutVector >
void SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >::spmvCuda( const InVector& inVector,
                                                                              OutVector& outVector,
                                                                              const int globalIdx,
                                                                              const int color ) const
{
    /*const IndexType offset = this->colorPointers[ i ];
    const IndexType stop = this->colorPointers[ i + 1 ];
    if( globalIdx >= stop || globalIdx < offset )
        return;*/

    IndexType inSliceIdx = threadIdx.x % SliceSize;
    const IndexType sliceIdx = globalIdx / SliceSize;
    const IndexType sliceLength = this->sliceRowLengths[ sliceIdx ];
    const IndexType begin = this->slicePointers[ sliceIdx ] + inSliceIdx * sliceLength;
    const IndexType rowMapping = this->inversePermutationArray[ globalIdx ];
    for( IndexType elementPtr = begin; elementPtr < begin + sliceLength; elementPtr++ )
    {
        IndexType column = this->columnIndexes[ elementPtr ];
        if( column == this->getPaddingIndex() )
            break;

        outVector[ rowMapping ] += inVector[ column ] * this->values[ elementPtr ];
        if( rowMapping != column )
        {
            outVector[ column ] += inVector[ rowMapping ] * this->values[ elementPtr ];
        }
    }
}
#endif

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          int SliceSize,
          typename InVector,
          typename OutVector >
__global__
void SlicedEllpackSymmetricGraphVectorProductCuda( const SlicedEllpackSymmetricGraph< Real, Devices::Cuda, Index, SliceSize >& matrix,
                                                   const InVector* inVector,
                                                   OutVector* outVector,
                                                   const int gridIdx,
                                                   const int color,
                                                   const int sliceOffset )
{
    int globalIdx = ( gridIdx * Devices::Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x + sliceOffset;
    matrix->smvCuda( *inVector, *outVector, globalIdx, color );
}
#endif

template<>
class SlicedEllpackSymmetricGraphDeviceDependentCode< Devices::Cuda >
{
   public:

      typedef Devices::Cuda Device;

      template< typename Real,
                typename Index,
                int SliceSize >
      static void initRowTraverse( const SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >& matrix,
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
      static void initRowTraverseFast( const SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >& matrix,
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
      static bool computeMaximalRowLengthInSlices( SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >& matrix,
                                                   const typename SlicedEllpackSymmetricGraph< Real, Device, Index >::RowLengthsVector& rowLengths,
                                                   Containers::Vector< Index, Device, Index >& sliceRowLengths,
                                                   Containers::Vector< Index, Device, Index >& slicePointers )
      {
#ifdef HAVE_CUDA
         typedef SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize > Matrix;
         typedef typename Matrix::RowLengthsVector CompressedRowLengthsVector;
         Matrix* kernel_matrix = Devices::Cuda::passToDevice( matrix );
         CompressedRowLengthsVector* kernel_rowLengths = Devices::Cuda::passToDevice( rowLengths );
         const Index numberOfSlices = roundUpDivision( matrix.getRows(), SliceSize );
         dim3 cudaBlockSize( 256 ), cudaGridSize( Devices::Cuda::getMaxGridSize() );
         const Index cudaBlocks = roundUpDivision( numberOfSlices, cudaBlockSize.x );
         const Index cudaGrids = roundUpDivision( cudaBlocks, Devices::Cuda::getMaxGridSize() );
         for( int gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
         {
            if( gridIdx == cudaGrids - 1 )
               cudaGridSize.x = cudaBlocks % Devices::Cuda::getMaxGridSize();
            SlicedEllpackSymmetricGraph_computeMaximalRowLengthInSlices_CudaKernel< Real, Index, SliceSize ><<< cudaGridSize, cudaBlockSize >>>
                                                                             ( kernel_matrix,
                                                                               kernel_rowLengths,
                                                                               gridIdx );
         }
         Devices::Cuda::freeFromDevice( kernel_matrix );
         Devices::Cuda::freeFromDevice( kernel_rowLengths );
         TNL_CHECK_CUDA_DEVICE;
#endif
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector,
                int SliceSize >
      static void vectorProduct( const SlicedEllpackSymmetricGraph< Real, Device, Index, SliceSize >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         // TODO: tohle
#ifdef HAVE_CUDA
         typedef SlicedEllpackSymmetricGraph< Real, Devices::Cuda, Index, SliceSize > Matrix;
         typedef typename Matrix::IndexType IndexType;
         Matrix* kernel_this = Devices::Cuda::passToDevice( matrix );
         InVector* kernel_inVector = Devices::Cuda::passToDevice( inVector );
         OutVector* kernel_outVector = Devices::Cuda::passToDevice( outVector );
         dim3 cudaBlockSize( 256 ), cudaGridSize( Devices::Cuda::getMaxGridSize() );
         for( IndexType color = 0; color < matrix.getNumberOfColors(); color++ )
         {
            IndexType offset = matrix.colorPointers.getElement( color ); //can be computed in kernel
            // IndexType rowStop = matrix.colorPointers.getElement( color + 1 ); can be computed in kernel
            IndexType inSliceOffset = offset % SliceSize;
            // TODO: inSliceIdx is undefined
            //IndexType rows = matrix.colorPointers.getElement( color + 1 ) - matrix.colorPointers.getElement( color ) + inSliceIdx;
            // TODO: rows id undefined
            /*const IndexType cudaBlocks = roundUpDivision( rows, cudaBlockSize.x );
            const IndexType cudaGrids = rondUpDivision( cudaBlocks, Devices::Cuda::getMaxGridSize );
            for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
            {
               if( gridIdx == cudaGrids - 1 )
                  cudaGridSize.x = cudaBlocks % Devices::Cuda::getMaxGridSize();
               // TODO: this cannot be used here and i is undefined
               //IndexType offset = this->colorPointers[ i ];
               IndexType inSliceIdx = offset % SliceSize;
               IndexType sliceOffset = offset - inSliceIdx;
               SlicedEllpackSymmetricGraphVectorProductCuda< Real, Index, InVector, OutVector >
                                                           <<< cudaGridSize, cudaBlockSize >>>
                                                           ( kernel_this,
                                                             kernel_inVector,
                                                             kernel_outVector,
                                                             gridIdx,
                                                             color,
                                                             sliceOffset );
            }*/
         }
         Devices::Cuda::freeFromDevice( kernel_this );
         Devices::Cuda::freeFromDevice( kernel_inVector );
         Devices::Cuda::freeFromDevice( kernel_outVector );
         TNL_CHECK_CUDA_DEVICE;
#endif
      }

};

} // namespace Matrices
} // namespace TNL
