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
   return tnlSparseMatrix< Real, Device, Index >::setDimensions( rows, columns );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::setRowLengths( const RowLengthsVector& rowLengths )
{
   const IndexType slices = roundUpDivision( this->rows, SliceSize );
   if( ! this->sliceRowLengths.setSize( slices ) ||
       ! this->slicePointers.setSize( slices + 1 ) )
      return false;

   /****
    * Compute maximal row length in each slice
    */
   DeviceDependentCode::computeMaximalRowLengthInSlices( *this, rowLengths );

   /****
    * Compute the slice pointers using the exclusive prefix sum
    */
   this->slicePointers.setElement( slices, 0 );
   this->slicePointers.computeExclusivePrefixSum();

   return this->allocateMatrixElements( this->slicePointers[ slices ] );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
Index tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::getRowLength( const IndexType row ) const
{
   const IndexType slice = roundUpDivision( row, SliceSize );
   return this->sliceRowLengths[ slice ];
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
void tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::reset()
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
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::setElementFast( const IndexType row,
                                                                               const IndexType column,
                                                                               const Real& value )
{
   return this->addElement( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::setElement( const IndexType row,
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
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::addElementFast( const IndexType row,
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
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::addElement( const IndexType row,
                                                                           const IndexType column,
                                                                           const RealType& value,
                                                                           const RealType& thisElementMultiplicator )
{
   return this->addElementFast( row, column, value, thisElementMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize > :: setRowFast( const IndexType row,
                                                                             const IndexType* columnIndexes,
                                                                             const RealType* values,
                                                                             const IndexType elements )
{
   const IndexType sliceIdx = row / SliceSize;
   const IndexType rowLength = this->sliceRowLengths[ sliceIdx ];
   if( elements > rowLength )
      return false;

   IndexType elementPointer = this->slicePointers[ sliceIdx ] +
                              rowLength * ( row - sliceIdx * SliceSize );
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
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize > :: setRow( const IndexType row,
                                                                         const IndexType* columnIndexes,
                                                                         const RealType* values,
                                                                         const IndexType elements )
{
   return this->setRowFast( row, columnIndexes, values, elements );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize > :: addRowFast( const IndexType row,
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
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize > :: addRow( const IndexType row,
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
Real tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::getElementFast( const IndexType row,
                                                                               const IndexType column ) const
{
   const IndexType sliceIdx = row / SliceSize;
   const IndexType rowLength = this->sliceRowLengths[ sliceIdx ];
   IndexType elementPtr = this->slicePointers[ sliceIdx ] +
                          rowLength * ( row - sliceIdx * SliceSize );
   const IndexType rowEnd = elementPtr + rowLength;
   while( elementPtr < rowEnd && this->columnIndexes[ elementPtr ] < column )
      elementPtr++;
   if( elementPtr < rowEnd && this->columnIndexes[ elementPtr ] == column )
      return this->values[ elementPtr ];
   return 0.0;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
Real tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::getElement( const IndexType row,
                                                                           const IndexType column ) const
{
   return this->getElementFast( row, column );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
void tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::getRowFast( const IndexType row,
                                                                           IndexType* columns,
                                                                           RealType* values ) const
{
   const IndexType sliceIdx = row / SliceSize;
   const IndexType rowLength = this->sliceRowLengths[ sliceIdx ];
   IndexType elementPtr = this->slicePointers[ sliceIdx ] +
                          rowLength * ( row - sliceIdx * SliceSize );
   for( IndexType i = 0; i < rowLength; i++ )
   {
      columns[ i ] = this->columnIndexes[ elementPtr ];
      values[ i ] = this->values[ elementPtr ];
      elementPtr++;
   }
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::getRow( const IndexType row,
                                                                       IndexType* columns,
                                                                       RealType* values ) const
{
   return this->getRowFast( row, columns, values );
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
   if( ! tnlSparseMatrix< Real, Device, Index >::save( file ) ||
       ! this->slicePointers.save( file ) ||
       ! this->sliceRowLengths.save( file ) )
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >::load( tnlFile& file )
{
   if( ! tnlSparseMatrix< Real, Device, Index >::load( file ) ||
       ! this->slicePointers.load( file ) ||
       ! this->sliceRowLengths.load( file ) )
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

template<>
class tnlSlicedEllpackMatrixDeviceDependentCode< tnlHost >
{
   public:

      typedef tnlHost Device;

      template< typename Real,
                typename Index,
                int SliceSize >
      static Index getRowBegin( const tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >& matrix,
                                const Index row )
      {
         return row * matrix.rowLengths;
      }

      template< typename Real,
                typename Index,
                int SliceSize >
      static Index getRowEnd( const tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >& matrix,
                                const Index row )
      {
         return ( row + 1 ) * matrix.rowLengths;
      }

      template< typename Real,
                typename Index,
                int SliceSize >
      static Index getElementStep( const tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >& matrix )
      {
         return 1;
      }

      template< typename Real,
                typename Index,
                int SliceSize >
      static bool computeMaximalRowLengthInSlices( tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >& matrix,
                                                   const typename tnlSlicedEllpackMatrix< Real, Device, Index >::RowLengthsVector& rowLengths )
      {
         Index row( 0 ), slice( 0 ), sliceRowLength( 0 );
         while( row < matrix.getRows() )
         {
            sliceRowLength = Max( rowLengths.getElement( row++ ), sliceRowLength );
            if( row % SliceSize == 0 )
            {
               matrix.sliceRowLengths.setElement( slice, sliceRowLength );
               matrix.slicePointers.setElement( slice++, sliceRowLength * SliceSize );
               sliceRowLength = 0;
            }
         }
         if( row % SliceSize != 0 )
         {
            matrix.sliceRowLengths.setElement( slice, sliceRowLength );
            matrix.slicePointers.setElement( slice++, sliceRowLength * SliceSize );
         }
      }
};

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          int SliceSize >
__global__ void tnlSlicedEllpackMatrix_computeMaximalRowLengthInSlices_CudaKernel( tnlSlicedEllpackMatrix< Real, tnlCuda, Index, SliceSize >* matrix,
                                                                                   const typename tnlSlicedEllpackMatrix< Real, tnlCuda, Index, SliceSize >::RowLengthsVector* rowLengths,
                                                                                   int gridIdx )
{

}
#endif

template<>
class tnlSlicedEllpackMatrixDeviceDependentCode< tnlCuda >
{
   public:

      typedef tnlCuda Device;

      template< typename Real,
                typename Index,
                int SliceSize >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      static Index getRowBegin( const tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >& matrix,
                                const Index row )
      {
         return row * matrix.rowLengths;
      }

      template< typename Real,
                typename Index,
                int SliceSize >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      static Index getRowEnd( const tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >& matrix,
                                const Index row )
      {
         return ( row + 1 ) * matrix.rowLengths;
      }

      template< typename Real,
                typename Index,
                int SliceSize >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      static Index getElementStep( const tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >& matrix )
      {
         return 1;
      }

      template< typename Real,
                typename Index,
                int SliceSize >
      static bool computeMaximalRowLengthInSlices( tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize >& matrix,
                                                   const typename tnlSlicedEllpackMatrix< Real, Device, Index >::RowLengthsVector& rowLengths )
      {
#ifdef HAVE_CUDA
         typedef tnlSlicedEllpackMatrix< Real, Device, Index, SliceSize > Matrix;
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
            tnlSlicedEllpackMatrix_computeMaximalRowLengthInSlices_CudaKernel< Real, Index, SliceSize ><<< cudaGridSize, cudaBlockSize >>>
                                                                             ( kernel_matrix,
                                                                               kernel_rowLengths,
                                                                               gridIdx );
         }
         tnlCuda::freeFromDevice( kernel_matrix );
         tnlCuda::freeFromDevice( kernel_rowLengths );
         checkCudaDevice;
#endif
      }
};



#endif /* TNLSLICEDELLPACKMATRIX_IMPL_H_ */
