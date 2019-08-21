/***************************************************************************
                          SlocedEllpackSymmetric_impl.h  -  description
                             -------------------
    begin                : Aug 30, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/SlicedEllpackSymmetric.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Math.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Matrices {

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::SlicedEllpackSymmetric()
{
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::setDimensions( const IndexType rows,
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
void SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::setCompressedRowLengths( ConstCompressedRowLengthsVectorView rowLengths )
{
   TNL_ASSERT( this->getRows() > 0, );
   TNL_ASSERT( this->getColumns() > 0, );
   const IndexType slices = roundUpDivision( this->rows, SliceSize );
   this->sliceRowLengths.setSize( slices );
   this->slicePointers.setSize( slices + 1 );

   // TODO: Uncomment the next line and fix the compilation
   //DeviceDependentCode::computeMaximalRowLengthInSlices( *this, rowLengths );

   throw std::runtime_error("code fix required");

   this->maxRowLength = max( rowLengths );

   this->slicePointers.template prefixSum< Containers::Algorithms::ScanType::Exclusive >();
   this->allocateMatrixElements( this->slicePointers.getElement( slices ) );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
Index SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::getRowLength( const IndexType row ) const
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
bool SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::setLike( const SlicedEllpackSymmetric< Real2, Device2, Index2, SliceSize >& matrix )
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
void SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::reset()
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
bool SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::operator == ( const SlicedEllpackSymmetric< Real2, Device2, Index2 >& matrix ) const
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
   throw Exceptions::NotImplementedError( "SlicedEllpackSymmetric::operator== is not implemented." );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::operator != ( const SlicedEllpackSymmetric< Real2, Device2, Index2 >& matrix ) const
{
   return ! ( ( *this ) == matrix );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
__cuda_callable__
bool SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::setElementFast( const IndexType row,
                                                                                  const IndexType column,
                                                                                  const Real& value )
{
   return this->addElementFast( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
bool SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::setElement( const IndexType row,
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
bool SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::addElementFast( const IndexType row,
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
bool SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::addElement( const IndexType row,
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
__cuda_callable__
bool SlicedEllpackSymmetric< Real, Device, Index, SliceSize > :: setRowFast( const IndexType row,
                                                                                const IndexType* columnIndexes,
                                                                                const RealType* values,
                                                                                const IndexType elements )
{
   const IndexType sliceIdx = row / SliceSize;
   const IndexType rowLength = this->sliceRowLengths[ sliceIdx ];
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
bool SlicedEllpackSymmetric< Real, Device, Index, SliceSize > :: setRow( const IndexType row,
                                                                            const IndexType* columnIndexes,
                                                                            const RealType* values,
                                                                            const IndexType elements )
{
   const IndexType sliceIdx = row / SliceSize;
   const IndexType rowLength = this->sliceRowLengths.getElement( sliceIdx );
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
bool SlicedEllpackSymmetric< Real, Device, Index, SliceSize > :: addRowFast( const IndexType row,
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
bool SlicedEllpackSymmetric< Real, Device, Index, SliceSize > :: addRow( const IndexType row,
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
Real SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::getElementFast( const IndexType row,
                                                                                  const IndexType column ) const
{
   if( row < column )
      return this->getElementFast( column, row );

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
Real SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::getElement( const IndexType row,
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
__cuda_callable__
void SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::getRowFast( const IndexType row,
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
void SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::getRow( const IndexType row,
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
template< typename InVector,
          typename OutVector >
__cuda_callable__
void SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::rowVectorProduct( const IndexType row,
                                                                                    const InVector& inVector,
                                                                                    OutVector& outVector ) const
{
   Real result = 0.0;
   Index elementPtr, rowEnd, step;
   DeviceDependentCode::initRowTraverseFast( *this, row, elementPtr, rowEnd, step );

   IndexType column;
   while( elementPtr < rowEnd &&
          ( column = this->columnIndexes[ elementPtr ] ) < this->columns &&
          column != this->getPaddingIndex() )
   {
      result += this->values[ elementPtr ] * inVector[ column ];
      if( row != column )
         outVector[ column ] += this->values[ elementPtr ] * inVector[ row ];
      elementPtr += step;
   }
   outVector[ row ] += result;
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
template< typename InVector,
          typename OutVector >
__device__
void SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::spmvCuda( const InVector& inVector,
                                                                            OutVector& outVector,
                                                                            int rowIdx ) const
{
    if( rowIdx >= this->getRows() )
        return;

    Real result = 0.0;
    Index elementPtr, rowEnd, step;
    DeviceDependentCode::initRowTraverseFast( *this, rowIdx, elementPtr, rowEnd, step );
    IndexType column;
    while( elementPtr < rowEnd &&
           ( column = this->columnIndexes[ elementPtr ] ) < this->columns &&
           column != this->getPaddingIndex() )
    {
        result += this->values[ elementPtr ] * inVector[ column ];
        if( rowIdx != column )
            outVector[ column ] += this->values[ elementPtr ] * inVector[ rowIdx ];
        elementPtr += step;
    }
    outVector[ rowIdx ] += result;
}
#endif

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          int SliceSize,
          typename InVector,
          typename OutVector >
__global__ 
void SlicedEllpackSymmetricVectorProductCudaKernel( 
const SlicedEllpackSymmetric< Real, Devices::Cuda, Index, SliceSize >* matrix,
                                                       const InVector* inVector,
                                                       OutVector* outVector,
                                                       int gridIdx )
{
   int rowIdx = ( gridIdx * Devices::Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   matrix->spmvCuda( *inVector, *outVector, rowIdx );
}
#endif

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename InVector,
             typename OutVector >
void SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::vectorProduct( const InVector& inVector,
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
void SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::addMatrix( const SlicedEllpackSymmetric< Real2, Device, Index2 >& matrix,
                                                                             const RealType& matrixMultiplicator,
                                                                             const RealType& thisMatrixMultiplicator )
{
   throw Exceptions::NotImplementedError( "SlicedEllpackSymmetric::addMatrix is not implemented." );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Real2,
             typename Index2 >
void SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::getTransposition( const SlicedEllpackSymmetric< Real2, Device, Index2 >& matrix,
                                                                                    const RealType& matrixMultiplicator )
{
   throw Exceptions::NotImplementedError( "SlicedEllpackSymmetric::getTransposition is not implemented." );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
   template< typename Vector >
bool SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::performSORIteration( const Vector& b,
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
     std::cerr << "There is zero on the diagonal in " << row << "-th row of thge matrix " << this->getName() << ". I cannot perform SOR iteration." <<std::endl;
      return false;
   }
   x. setElement( row, x[ row ] + omega / diagonalValue * ( b[ row ] - sum ) );
   return true;
}


template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::save( File& file ) const
{
   Sparse< Real, Device, Index >::save( file );
   file << this->slicePointers << this->sliceRowLengths;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::load( File& file )
{
   Sparse< Real, Device, Index >::load( file );
   file >> this->slicePointers >> this->sliceRowLengths;
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
void SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::print( std::ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      const IndexType sliceIdx = row / SliceSize;
      const IndexType rowLength = this->sliceRowLengths.getElement( sliceIdx );
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
      str <<std::endl;
   }
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
__device__ void SlicedEllpackSymmetric< Real, Device, Index, SliceSize >::computeMaximalRowLengthInSlicesCuda( ConstCompressedRowLengthsVectorView rowLengths,
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
class SlicedEllpackSymmetricDeviceDependentCode< Devices::Host >
{
   public:

      typedef Devices::Host Device;

      template< typename Real,
                typename Index,
                int SliceSize >
      static void initRowTraverse( const SlicedEllpackSymmetric< Real, Device, Index, SliceSize >& matrix,
                                   const Index row,
                                   Index& rowBegin,
                                   Index& rowEnd,
                                   Index& step )
      {
         const Index sliceIdx = row / SliceSize;
         const Index slicePointer = matrix.slicePointers.getElement( sliceIdx );
         const Index rowLength = matrix.sliceRowLengths.getElement( sliceIdx );

         rowBegin = slicePointer + rowLength * ( row - sliceIdx * SliceSize );
         rowEnd = rowBegin + rowLength;
         step = 1;
      }

      template< typename Real,
                typename Index,
                int SliceSize >
      __cuda_callable__
      static void initRowTraverseFast( const SlicedEllpackSymmetric< Real, Device, Index, SliceSize >& matrix,
                                       const Index row,
                                       Index& rowBegin,
                                       Index& rowEnd,
                                       Index& step )
      {
         const Index sliceIdx = row / SliceSize;
         const Index slicePointer = matrix.slicePointers[ sliceIdx ];
         const Index rowLength = matrix.sliceRowLengths[ sliceIdx ];

         rowBegin = slicePointer + rowLength * ( row - sliceIdx * SliceSize );
         rowEnd = rowBegin + rowLength;
         step = 1;
      }


      template< typename Real,
                typename Index,
                int SliceSize >
      static void computeMaximalRowLengthInSlices( SlicedEllpackSymmetric< Real, Device, Index, SliceSize >& matrix,
                                                   typename SlicedEllpackSymmetric< Real, Device, Index >::ConstCompressedRowLengthsVectorView rowLengths )
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
         matrix.slicePointers.setElement( matrix.slicePointers.getSize() - 1, 0 );
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector,
                int SliceSize >
      static void vectorProduct( const SlicedEllpackSymmetric< Real, Device, Index, SliceSize >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         for( Index row = 0; row < matrix.getRows(); row++ )
         {
             matrix.rowVectorProduct( row, inVector, outVector );
         }
      }

};

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          int SliceSize >
__global__ void SlicedEllpackSymmetric_computeMaximalRowLengthInSlices_CudaKernel( SlicedEllpackSymmetric< Real, Devices::Cuda, Index, SliceSize >* matrix,
                                                                                   typename SlicedEllpackSymmetric< Real, Devices::Cuda, Index, SliceSize >::ConstCompressedRowLengthsVectorView rowLengths,
                                                                                   int gridIdx )
{
   const Index sliceIdx = gridIdx * Devices::Cuda::getMaxGridSize() * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
   matrix->computeMaximalRowLengthInSlicesCuda( rowLengths, sliceIdx );
}
#endif

template<>
class SlicedEllpackSymmetricDeviceDependentCode< Devices::Cuda >
{
   public:

      typedef Devices::Cuda Device;

      template< typename Real,
                typename Index,
                int SliceSize >
      static void initRowTraverse( const SlicedEllpackSymmetric< Real, Device, Index, SliceSize >& matrix,
                                   const Index row,
                                   Index& rowBegin,
                                   Index& rowEnd,
                                   Index& step )
      {
         const Index sliceIdx = row / SliceSize;
         const Index slicePointer = matrix.slicePointers.getElement( sliceIdx );
         const Index rowLength = matrix.sliceRowLengths.getElement( sliceIdx );

         rowBegin = slicePointer + row - sliceIdx * SliceSize;
         rowEnd = rowBegin + rowLength * SliceSize;
         step = SliceSize;
      }

      template< typename Real,
                typename Index,
                int SliceSize >
      __cuda_callable__
      static void initRowTraverseFast( const SlicedEllpackSymmetric< Real, Device, Index, SliceSize >& matrix,
                                       const Index row,
                                       Index& rowBegin,
                                       Index& rowEnd,
                                       Index& step )
      {
         const Index sliceIdx = row / SliceSize;
         const Index slicePointer = matrix.slicePointers[ sliceIdx ];
         const Index rowLength = matrix.sliceRowLengths[ sliceIdx ];

         rowBegin = slicePointer + row - sliceIdx * SliceSize;
         rowEnd = rowBegin + rowLength * SliceSize;
         step = SliceSize;

      }

      template< typename Real,
                typename Index,
                int SliceSize >
      static void computeMaximalRowLengthInSlices( SlicedEllpackSymmetric< Real, Device, Index, SliceSize >& matrix,
                                                   typename SlicedEllpackSymmetric< Real, Device, Index >::ConstCompressedRowLengthsVectorView rowLengths )
      {
#ifdef HAVE_CUDA
         typedef SlicedEllpackSymmetric< Real, Device, Index, SliceSize > Matrix;
         typedef typename Matrix::RowLengthsVector CompressedRowLengthsVector;
         Matrix* kernel_matrix = Devices::Cuda::passToDevice( matrix );
         const Index numberOfSlices = roundUpDivision( matrix.getRows(), SliceSize );
         dim3 cudaBlockSize( 256 ), cudaGridSize( Devices::Cuda::getMaxGridSize() );
         const Index cudaBlocks = roundUpDivision( numberOfSlices, cudaBlockSize.x );
         const Index cudaGrids = roundUpDivision( cudaBlocks, Devices::Cuda::getMaxGridSize() );
         for( int gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
         {
            if( gridIdx == cudaGrids - 1 )
               cudaGridSize.x = cudaBlocks % Devices::Cuda::getMaxGridSize();
            SlicedEllpackSymmetric_computeMaximalRowLengthInSlices_CudaKernel< Real, Index, SliceSize ><<< cudaGridSize, cudaBlockSize >>>
                                                                             ( kernel_matrix,
                                                                               rowLengths,
                                                                               gridIdx );
         }
         Devices::Cuda::freeFromDevice( kernel_matrix );
         TNL_CHECK_CUDA_DEVICE;
#endif
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector,
                int SliceSize >
      static void vectorProduct( const SlicedEllpackSymmetric< Real, Device, Index, SliceSize >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
#ifdef HAVE_CUDA
         typedef SlicedEllpackSymmetric< Real, Device, Index, SliceSize > Matrix;
         typedef typename Matrix::IndexType IndexType;
         Matrix* kernel_this = Devices::Cuda::passToDevice( matrix );
         InVector* kernel_inVector = Devices::Cuda::passToDevice( inVector );
         OutVector* kernel_outVector = Devices::Cuda::passToDevice( outVector );
         dim3 cudaBlockSize( 256 ), cudaGridSize( Devices::Cuda::getMaxGridSize() );
         const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
         const IndexType cudaGrids = roundUpDivision( cudaBlocks, Devices::Cuda::getMaxGridSize() );
         for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
         {
            if( gridIdx == cudaGrids - 1 )
               cudaGridSize.x = cudaBlocks % Devices::Cuda::getMaxGridSize();
            SlicedEllpackSymmetricVectorProductCudaKernel< Real, Index, SliceSize, InVector, OutVector >
                                                            <<< cudaGridSize, cudaBlockSize >>>
                                                              ( kernel_this,
                                                                kernel_inVector,
                                                                kernel_outVector,
                                                                gridIdx );
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
