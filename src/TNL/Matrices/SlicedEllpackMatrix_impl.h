/***************************************************************************
                          tnlSlicedSlicedEllpackMatrix_impl.h  -  description
                             -------------------
    begin                : Dec 8, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/SlicedEllpackMatrix.h>
#include <TNL/Vectors/Vector.h>
#include <TNL/core/mfuncs.h>

namespace TNL {
namespace Matrices {   

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
SlicedEllpackMatrix< Real, Device, Index, SliceSize >::SlicedEllpackMatrix()
{
};

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
String SlicedEllpackMatrix< Real, Device, Index, SliceSize >::getType()
{
   return String( "SlicedEllpackMatrix< ") +
          String( TNL::getType< Real >() ) +
          String( ", " ) +
          Device :: getDeviceType() +
          String( " >" );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
String SlicedEllpackMatrix< Real, Device, Index, SliceSize >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool SlicedEllpackMatrix< Real, Device, Index, SliceSize >::setDimensions( const IndexType rows,
                                                                              const IndexType columns )
{
   Assert( rows > 0 && columns > 0,
              std::cerr << "rows = " << rows
                   << " columns = " << columns << std::endl );
   return SparseMatrix< Real, Device, Index >::setDimensions( rows, columns );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool SlicedEllpackMatrix< Real, Device, Index, SliceSize >::setCompressedRowsLengths( const CompressedRowsLengthsVector& rowLengths )
{
   Assert( this->getRows() > 0, );
   Assert( this->getColumns() > 0, );
   const IndexType slices = roundUpDivision( this->rows, SliceSize );
   if( ! this->sliceCompressedRowsLengths.setSize( slices ) ||
       ! this->slicePointers.setSize( slices + 1 ) )
      return false;

   DeviceDependentCode::computeMaximalRowLengthInSlices( *this, rowLengths );

   this->maxRowLength = rowLengths.max();

   this->slicePointers.computeExclusivePrefixSum();
   return this->allocateMatrixElements( this->slicePointers.getElement( slices ) );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
Index SlicedEllpackMatrix< Real, Device, Index, SliceSize >::getRowLength( const IndexType row ) const
{
   const IndexType slice = row / SliceSize;
   return this->sliceCompressedRowsLengths.getElement( slice );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool SlicedEllpackMatrix< Real, Device, Index, SliceSize >::setLike( const SlicedEllpackMatrix< Real2, Device2, Index2, SliceSize >& matrix )
{
   if( !SparseMatrix< Real, Device, Index >::setLike( matrix ) ||
       ! this->slicePointers.setLike( matrix.slicePointers ) ||
       ! this->sliceCompressedRowsLengths.setLike( matrix.sliceCompressedRowsLengths ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpackMatrix< Real, Device, Index, SliceSize >::reset()
{
   SparseMatrix< Real, Device, Index >::reset();
   this->slicePointers.reset();
   this->sliceCompressedRowsLengths.reset();
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool SlicedEllpackMatrix< Real, Device, Index, SliceSize >::operator == ( const SlicedEllpackMatrix< Real2, Device2, Index2 >& matrix ) const
{
   Assert( this->getRows() == matrix.getRows() &&
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
bool SlicedEllpackMatrix< Real, Device, Index, SliceSize >::operator != ( const SlicedEllpackMatrix< Real2, Device2, Index2 >& matrix ) const
{
   return ! ( ( *this ) == matrix );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
__cuda_callable__
bool SlicedEllpackMatrix< Real, Device, Index, SliceSize >::setElementFast( const IndexType row,
                                                                               const IndexType column,
                                                                               const Real& value )
{
   return this->addElementFast( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool SlicedEllpackMatrix< Real, Device, Index, SliceSize >::setElement( const IndexType row,
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
bool SlicedEllpackMatrix< Real, Device, Index, SliceSize >::addElementFast( const IndexType row,
                                                                               const IndexType column,
                                                                               const RealType& value,
                                                                               const RealType& thisElementMultiplicator )
{
   Assert( row >= 0 && row < this->rows &&
              column >= 0 && column <= this->rows,
              std::cerr << " row = " << row
                   << " column = " << column
                   << " this->rows = " << this->rows
                   << " this->columns = " << this-> columns );

   Index elementPtr, rowEnd, step;
   DeviceDependentCode::initRowTraverseFast( *this, row, elementPtr, rowEnd, step );

   IndexType col;
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
bool SlicedEllpackMatrix< Real, Device, Index, SliceSize >::addElement( const IndexType row,
                                                                           const IndexType column,
                                                                           const RealType& value,
                                                                           const RealType& thisElementMultiplicator )
{
   Assert( row >= 0 && row < this->rows &&
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
__cuda_callable__
bool SlicedEllpackMatrix< Real, Device, Index, SliceSize > :: setRowFast( const IndexType row,
                                                                             const IndexType* columnIndexes,
                                                                             const RealType* values,
                                                                             const IndexType elements )
{
   const IndexType sliceIdx = row / SliceSize;
   const IndexType rowLength = this->sliceCompressedRowsLengths[ sliceIdx ];
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
bool SlicedEllpackMatrix< Real, Device, Index, SliceSize > :: setRow( const IndexType row,
                                                                         const IndexType* columnIndexes,
                                                                         const RealType* values,
                                                                         const IndexType elements )
{
   const IndexType sliceIdx = row / SliceSize;
   const IndexType rowLength = this->sliceCompressedRowsLengths.getElement( sliceIdx );
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
bool SlicedEllpackMatrix< Real, Device, Index, SliceSize > :: addRowFast( const IndexType row,
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
bool SlicedEllpackMatrix< Real, Device, Index, SliceSize > :: addRow( const IndexType row,
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
Real SlicedEllpackMatrix< Real, Device, Index, SliceSize >::getElementFast( const IndexType row,
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
Real SlicedEllpackMatrix< Real, Device, Index, SliceSize >::getElement( const IndexType row,
                                                                           const IndexType column ) const
{

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
__cuda_callable__
void SlicedEllpackMatrix< Real, Device, Index, SliceSize >::getRowFast( const IndexType row,
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
typename SlicedEllpackMatrix< Real, Device, Index, SliceSize >::MatrixRow
SlicedEllpackMatrix< Real, Device, Index, SliceSize >::
getRow( const IndexType rowIndex )
{
   Index rowBegin, rowEnd, step;
   DeviceDependentCode::initRowTraverseFast( *this, rowIndex, rowBegin, rowEnd, step );
   const IndexType slice = rowIndex / SliceSize;
   return MatrixRow( &this->columnIndexes[ rowBegin ],
                     &this->values[ rowBegin ],
                     this->sliceCompressedRowsLengths[ slice ],
                     step );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
__cuda_callable__
const typename SlicedEllpackMatrix< Real, Device, Index, SliceSize >::MatrixRow
SlicedEllpackMatrix< Real, Device, Index, SliceSize >::
getRow( const IndexType rowIndex ) const
{
   Index rowBegin, rowEnd, step;
   DeviceDependentCode::initRowTraverseFast( *this, rowIndex, rowBegin, rowEnd, step );
   const IndexType slice = rowIndex / SliceSize;
   return MatrixRow( &this->columnIndexes[ rowBegin ],
                     &this->values[ rowBegin ],
                     this->sliceCompressedRowsLengths[ slice ],
                     step );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
  template< typename Vector >
__cuda_callable__
typename Vector::RealType SlicedEllpackMatrix< Real, Device, Index, SliceSize >::rowVectorProduct( const IndexType row,
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
void SlicedEllpackMatrix< Real, Device, Index, SliceSize >::vectorProduct( const InVector& inVector,
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
void SlicedEllpackMatrix< Real, Device, Index, SliceSize >::addMatrix( const SlicedEllpackMatrix< Real2, Device, Index2 >& matrix,
                                                                          const RealType& matrixMultiplicator,
                                                                          const RealType& thisMatrixMultiplicator )
{
   Assert( false, std::cerr << "TODO: implement" );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Real2,
             typename Index2 >
void SlicedEllpackMatrix< Real, Device, Index, SliceSize >::getTransposition( const SlicedEllpackMatrix< Real2, Device, Index2 >& matrix,
                                                                      const RealType& matrixMultiplicator )
{
   Assert( false, std::cerr << "TODO: implement" );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Vector >
bool SlicedEllpackMatrix< Real, Device, Index, SliceSize >::performSORIteration( const Vector& b,
                                                                                    const IndexType row,
                                                                                    Vector& x,
                                                                                    const RealType& omega ) const
{
   Assert( row >=0 && row < this->getRows(),
              std::cerr << "row = " << row
                   << " this->getRows() = " << this->getRows() << std::endl );

   RealType diagonalValue( 0.0 );
   RealType sum( 0.0 );

   /*const IndexType sliceIdx = row / SliceSize;
   const IndexType rowLength = this->sliceCompressedRowsLengths[ sliceIdx ];
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


template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool SlicedEllpackMatrix< Real, Device, Index, SliceSize >::save( File& file ) const
{
   if( ! SparseMatrix< Real, Device, Index >::save( file ) ||
       ! this->slicePointers.save( file ) ||
       ! this->sliceCompressedRowsLengths.save( file ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool SlicedEllpackMatrix< Real, Device, Index, SliceSize >::load( File& file )
{
   if( ! SparseMatrix< Real, Device, Index >::load( file ) ||
       ! this->slicePointers.load( file ) ||
       ! this->sliceCompressedRowsLengths.load( file ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool SlicedEllpackMatrix< Real, Device, Index, SliceSize >::save( const String& fileName ) const
{
   return Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool SlicedEllpackMatrix< Real, Device, Index, SliceSize >::load( const String& fileName )
{
   return Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpackMatrix< Real, Device, Index, SliceSize >::print( std::ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      const IndexType sliceIdx = row / SliceSize;
      const IndexType rowLength = this->sliceCompressedRowsLengths.getElement( sliceIdx );
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

#ifdef HAVE_CUDA
template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
__device__ void SlicedEllpackMatrix< Real, Device, Index, SliceSize >::computeMaximalRowLengthInSlicesCuda( const CompressedRowsLengthsVector& rowLengths,
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
   this->sliceCompressedRowsLengths[ sliceIdx ] = maxRowLength;
   this->slicePointers[ sliceIdx ] = maxRowLength * SliceSize;
   if( threadIdx.x == 0 )
      this->slicePointers[ this->slicePointers.getSize() - 1 ] = 0;

}
#endif

template<>
class SlicedEllpackMatrixDeviceDependentCode< Devices::Host >
{
   public:

      typedef Devices::Host Device;

      template< typename Real,
                typename Index,
                int SliceSize >
      static void initRowTraverse( const SlicedEllpackMatrix< Real, Device, Index, SliceSize >& matrix,
                                   const Index row,
                                   Index& rowBegin,
                                   Index& rowEnd,
                                   Index& step )
      {
         const Index sliceIdx = row / SliceSize;
         const Index slicePointer = matrix.slicePointers.getElement( sliceIdx );
         const Index rowLength = matrix.sliceCompressedRowsLengths.getElement( sliceIdx );

         rowBegin = slicePointer + rowLength * ( row - sliceIdx * SliceSize );
         rowEnd = rowBegin + rowLength;
         step = 1;
      }

      template< typename Real,
                typename Index,
                int SliceSize >
      __cuda_callable__
      static void initRowTraverseFast( const SlicedEllpackMatrix< Real, Device, Index, SliceSize >& matrix,
                                       const Index row,
                                       Index& rowBegin,
                                       Index& rowEnd,
                                       Index& step )
      {
         const Index sliceIdx = row / SliceSize;
         const Index slicePointer = matrix.slicePointers[ sliceIdx ];
         const Index rowLength = matrix.sliceCompressedRowsLengths[ sliceIdx ];

         rowBegin = slicePointer + rowLength * ( row - sliceIdx * SliceSize );
         rowEnd = rowBegin + rowLength;
         step = 1;
      }


      template< typename Real,
                typename Index,
                int SliceSize >
      static bool computeMaximalRowLengthInSlices( SlicedEllpackMatrix< Real, Device, Index, SliceSize >& matrix,
                                                   const typename SlicedEllpackMatrix< Real, Device, Index >::CompressedRowsLengthsVector& rowLengths )
      {
         Index row( 0 ), slice( 0 ), sliceRowLength( 0 );
         while( row < matrix.getRows() )
         {
            sliceRowLength = max( rowLengths.getElement( row++ ), sliceRowLength );
            if( row % SliceSize == 0 )
            {
               matrix.sliceCompressedRowsLengths.setElement( slice, sliceRowLength );
               matrix.slicePointers.setElement( slice++, sliceRowLength * SliceSize );
               sliceRowLength = 0;
            }
         }
         if( row % SliceSize != 0 )
         {
            matrix.sliceCompressedRowsLengths.setElement( slice, sliceRowLength );
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
      static void vectorProduct( const SlicedEllpackMatrix< Real, Device, Index, SliceSize >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
         for( Index row = 0; row < matrix.getRows(); row ++ )
            outVector[ row ] = matrix.rowVectorProduct( row, inVector );
      }

};

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          int SliceSize >
__global__ void SlicedEllpackMatrix_computeMaximalRowLengthInSlices_CudaKernel( SlicedEllpackMatrix< Real, Devices::Cuda, Index, SliceSize >* matrix,
                                                                                   const typename SlicedEllpackMatrix< Real, Devices::Cuda, Index, SliceSize >::CompressedRowsLengthsVector* rowLengths,
                                                                                   int gridIdx )
{
   const Index sliceIdx = gridIdx * Devices::Cuda::getMaxGridSize() * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
   matrix->computeMaximalRowLengthInSlicesCuda( *rowLengths, sliceIdx );
}
#endif

#ifdef HAVE_CUDA
template<
   typename Real,
   typename Index,
   int SliceSize >
__global__ void SlicedEllpackMatrixVectorProductCudaKernel(
   const Index rows,
   const Index columns,
   const Index* slicePointers,
   const Index* sliceCompressedRowsLengths,
   const Index paddingIndex,
   const Index* columnIndexes,
   const Real* values,
   const Real* inVector,
   Real* outVector,
   const Index gridIdx )
{
   const Index rowIdx = ( gridIdx * Devices::Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( rowIdx >= rows )
      return;
   const Index sliceIdx = rowIdx / SliceSize;
   const Index slicePointer = slicePointers[ sliceIdx ];
   const Index rowLength = sliceCompressedRowsLengths[ sliceIdx ];
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
   outVector[ rowIdx ] = result;
}
#endif


template<>
class SlicedEllpackMatrixDeviceDependentCode< Devices::Cuda >
{
   public:

      typedef Devices::Cuda Device;

      template< typename Real,
                typename Index,
                int SliceSize >
      static void initRowTraverse( const SlicedEllpackMatrix< Real, Device, Index, SliceSize >& matrix,
                                   const Index row,
                                   Index& rowBegin,
                                   Index& rowEnd,
                                   Index& step )
      {
         const Index sliceIdx = row / SliceSize;
         const Index slicePointer = matrix.slicePointers.getElement( sliceIdx );
         const Index rowLength = matrix.sliceCompressedRowsLengths.getElement( sliceIdx );

         rowBegin = slicePointer + row - sliceIdx * SliceSize;
         rowEnd = rowBegin + rowLength * SliceSize;
         step = SliceSize;
      }

      template< typename Real,
                typename Index,
                int SliceSize >
      __cuda_callable__
      static void initRowTraverseFast( const SlicedEllpackMatrix< Real, Device, Index, SliceSize >& matrix,
                                       const Index row,
                                       Index& rowBegin,
                                       Index& rowEnd,
                                       Index& step )
      {
         const Index sliceIdx = row / SliceSize;
         const Index slicePointer = matrix.slicePointers[ sliceIdx ];
         const Index rowLength = matrix.sliceCompressedRowsLengths[ sliceIdx ];

         rowBegin = slicePointer + row - sliceIdx * SliceSize;
         rowEnd = rowBegin + rowLength * SliceSize;
         step = SliceSize;

      }

      template< typename Real,
                typename Index,
                int SliceSize >
      static bool computeMaximalRowLengthInSlices( SlicedEllpackMatrix< Real, Device, Index, SliceSize >& matrix,
                                                   const typename SlicedEllpackMatrix< Real, Device, Index >::CompressedRowsLengthsVector& rowLengths )
      {
#ifdef HAVE_CUDA
         typedef SlicedEllpackMatrix< Real, Device, Index, SliceSize > Matrix;
         typedef typename Matrix::CompressedRowsLengthsVector CompressedRowsLengthsVector;
         Matrix* kernel_matrix = Devices::Cuda::passToDevice( matrix );
         CompressedRowsLengthsVector* kernel_rowLengths = Devices::Cuda::passToDevice( rowLengths );
         const Index numberOfSlices = roundUpDivision( matrix.getRows(), SliceSize );
         dim3 cudaBlockSize( 256 ), cudaGridSize( Devices::Cuda::getMaxGridSize() );
         const Index cudaBlocks = roundUpDivision( numberOfSlices, cudaBlockSize.x );
         const Index cudaGrids = roundUpDivision( cudaBlocks, Devices::Cuda::getMaxGridSize() );
         for( int gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
         {
            if( gridIdx == cudaGrids - 1 )
               cudaGridSize.x = cudaBlocks % Devices::Cuda::getMaxGridSize();
            SlicedEllpackMatrix_computeMaximalRowLengthInSlices_CudaKernel< Real, Index, SliceSize ><<< cudaGridSize, cudaBlockSize >>>
                                                                             ( kernel_matrix,
                                                                               kernel_rowLengths,
                                                                               gridIdx );
         }
         Devices::Cuda::freeFromDevice( kernel_matrix );
         Devices::Cuda::freeFromDevice( kernel_rowLengths );
         checkCudaDevice;
#endif
         return true;
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector,
                int SliceSize >
      static void vectorProduct( const SlicedEllpackMatrix< Real, Device, Index, SliceSize >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         //MatrixVectorProductCuda( matrix, inVector, outVector );
         #ifdef HAVE_CUDA
            typedef SlicedEllpackMatrix< Real, Device, Index, SliceSize > Matrix;
            typedef typename Matrix::IndexType IndexType;
            //Matrix* kernel_this = Devices::Cuda::passToDevice( matrix );
            //InVector* kernel_inVector = Devices::Cuda::passToDevice( inVector );
            //OutVector* kernel_outVector = Devices::Cuda::passToDevice( outVector );
            dim3 cudaBlockSize( 256 ), cudaGridSize( Devices::Cuda::getMaxGridSize() );
            const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
            const IndexType cudaGrids = roundUpDivision( cudaBlocks, Devices::Cuda::getMaxGridSize() );
            for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
            {
               if( gridIdx == cudaGrids - 1 )
                  cudaGridSize.x = cudaBlocks % Devices::Cuda::getMaxGridSize();
               SlicedEllpackMatrixVectorProductCudaKernel
               < Real, Index, SliceSize >
                <<< cudaGridSize, cudaBlockSize >>>
                ( matrix.getRows(),
                  matrix.getColumns(),
                  matrix.slicePointers.getData(),
                  matrix.sliceCompressedRowsLengths.getData(),
                  matrix.getPaddingIndex(),
                  matrix.columnIndexes.getData(),
                  matrix.values.getData(),
                  inVector.getData(),
                  outVector.getData(),
                  gridIdx );
               checkCudaDevice;
            }
            //Devices::Cuda::freeFromDevice( kernel_this );
            //Devices::Cuda::freeFromDevice( kernel_inVector );
            //Devices::Cuda::freeFromDevice( kernel_outVector );
            checkCudaDevice;
            cudaThreadSynchronize();
         #endif
      }

};

} // namespace Matrices
} // namespace TNL
