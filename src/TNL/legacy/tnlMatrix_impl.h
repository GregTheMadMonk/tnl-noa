/***************************************************************************
                          tnlMatrix_impl.h  -  description
                             -------------------
    begin                : Dec 18, 2013
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

#ifndef TNLMATRIX_IMPL_H_
#define TNLMATRIX_IMPL_H_

#include <matrices/tnlMatrix.h>
#include <core/TNL_ASSERT.h>

template< typename Real,
          typename Device,
          typename Index >
tnlMatrix< Real, Device, Index >::tnlMatrix()
: rows( 0 ),
  columns( 0 ),
  numberOfColors( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index >
 bool tnlMatrix< Real, Device, Index >::setDimensions( const IndexType rows,
                                                       const IndexType columns )
{
   TNL_ASSERT( rows > 0 && columns > 0,
           std::cerr << " rows = " << rows << " columns = " << columns );
   this->rows = rows;
   this->columns = columns;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlMatrix< Real, Device, Index >::getRowLengths( Containers::Vector< IndexType, DeviceType, IndexType >& rowLengths ) const
{
   rowLengths.setSize( this->getRows() );
   for( IndexType row = 0; row < this->getRows(); row++ )
      rowLengths.setElement( row, this->getRowLength( row ) );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlMatrix< Real, Device, Index >::setLike( const tnlMatrix< Real2, Device2, Index2 >& matrix )
{
   return setDimensions( matrix.getRows(), matrix.getColumns() );
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Index tnlMatrix< Real, Device, Index >::getRows() const
{
   return this->rows;
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Index tnlMatrix< Real, Device, Index >::getColumns() const
{
   return this->columns;
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Index tnlMatrix< Real, Device, Index >::getNumberOfColors() const
{
    return this->numberOfColors;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlMatrix< Real, Device, Index >::reset()
{
   this->rows = 0;
   this->columns = 0;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Matrix >
bool tnlMatrix< Real, Device, Index >::copyFrom( const Matrix& matrix,
                                                 const CompressedRowLengthsVector& rowLengths )
{
   /*tnlStaticAssert( DeviceType::DeviceType == Devices::HostDevice, );
   tnlStaticAssert( DeviceType::DeviceType == Matrix:DeviceType::DeviceType, );*/

   this->setLike( matrix );
   if( ! this->setCompressedRowLengths( rowLengths ) )
      return false;
   Containers::Vector< RealType, Devices::Host, IndexType > values;
   Containers::Vector< IndexType, Devices::Host, IndexType > columns;
   if( ! values.setSize( this->getColumns() ) ||
       ! columns.setSize( this->getColumns() ) )
      return false;
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      matrix.getRow( row, columns.getData(), values.getData() );
      this->setRow( row, columns.getData(), values.getData(), rowLengths.getElement( row ) );
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
tnlMatrix< Real, Device, Index >& tnlMatrix< Real, Device, Index >::operator = ( const tnlMatrix< RealType, DeviceType, IndexType >& m )
{
   this->setLike( m );

   Containers::Vector< IndexType, DeviceType, IndexType > rowLengths;
   m.getRowLengths( rowLengths );
   this->setCompressedRowLengths( rowLengths );

   Containers::Vector< RealType, DeviceType, IndexType > rowValues;
   Containers::Vector< IndexType, DeviceType, IndexType > rowColumns;
   const IndexType maxRowLength = rowLengths.max();
   rowValues.setSize( maxRowLength );
   rowColumns.setSize( maxRowLength );
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      m.getRow( row,
                rowColumns.getData(),
                rowValues.getData() );
      this->setRow( row,
                    rowColumns.getData(),
                    rowValues.getData(),
                    m.getRowLength( row ) );
   }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Matrix >
bool tnlMatrix< Real, Device, Index >::operator == ( const Matrix& matrix ) const
{
   if( this->getRows() != matrix.getRows() ||
       this->getColumns() != matrix.getColumns() )
      return false;
   for( IndexType row = 0; row < this->getRows(); row++ )
      for( IndexType column = 0; column < this->getColumns(); column++ )
         if( this->getElement( row, column ) != matrix.getElement( row, column ) )
            return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Matrix >
bool tnlMatrix< Real, Device, Index >::operator != ( const Matrix& matrix ) const
{
   return ! operator == ( matrix );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMatrix< Real, Device, Index >::save( File& file ) const
{
   if( ! tnlObject::save( file ) ||
       ! file.write( &this->rows ) ||
       ! file.write( &this->columns ) ||
       ! this->values.save( file ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMatrix< Real, Device, Index >::load( File& file )
{
   if( ! tnlObject::load( file ) ||
       ! file.read( &this->rows ) ||
       ! file.read( &this->columns ) ||
       ! this->values.load( file ) )
      return false;
   return true;
}

/*
template< typename Real,
          typename Device,
          typename Index >
void tnlMatrix< Real, Device, Index >::computeColorsVector(Containers::Vector<Index, Device, Index> &colorsVector)
{
   this->numberOfColors = 0;

   for( IndexType i = this->getRows() - 1; i >= 0; i-- )
   {
      // init color array
      Containers::Vector< Index, Device, Index > usedColors;
      usedColors.setSize( this->numberOfColors );
      for( IndexType j = 0; j < this->numberOfColors; j++ )
         usedColors.setElement( j, 0 );

      // find all colors used in given row

   }
}
 */

template< typename Real,
          typename Device,
          typename Index >
void tnlMatrix< Real, Device, Index >::print( ostream& str ) const
{
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMatrix< Real, Device, Index >::help( bool verbose )
{
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
void tnlMatrix< Real, Device, Index >::copyFromHostToCuda( tnlMatrix< Real, Devices::Host, Index >& matrix )
{
    this->numberOfColors = matrix.getNumberOfColors();
    this->columns = matrix.getColumns();
    this->rows = matrix.getRows();

    this->values.setSize( matrix.getValuesSize() );
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlMatrix< Real, Device, Index >::getValuesSize() const
{
    return this->values.getSize();
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
  __device__ __host__
#endif
void tnlMatrix< Real, Device, Index >::computeColorsVector(Containers::Vector<Index, Device, Index> &colorsVector)
{
    for( IndexType i = this->getRows() - 1; i >= 0; i-- )
    {
        // init color array
        Containers::Vector< Index, Device, Index > usedColors;
        usedColors.setSize( this->numberOfColors );
        for( IndexType j = 0; j < this->numberOfColors; j++ )
            usedColors.setElement( j, 0 );

        // find all colors used in given row
        for( IndexType j = i + 1; j < this->getColumns(); j++ )
             if( this->getElement( i, j ) != 0.0 )
                 usedColors.setElement( colorsVector.getElement( j ), 1 );

        // find unused color
        bool found = false;
        for( IndexType j = 0; j < this->numberOfColors; j++ )
            if( usedColors.getElement( j ) == 0 )
            {
                colorsVector.setElement( i, j );
                found = true;
                break;
            }
        if( !found )
        {
            colorsVector.setElement( i, this->numberOfColors );
            this->numberOfColors++;
        }
    }
}

#ifdef HAVE_CUDA
template< typename Matrix,
          typename InVector,
          typename OutVector >
__global__ void tnlMatrixVectorProductCudaKernel( const Matrix* matrix,
                                                  const InVector* inVector,
                                                  OutVector* outVector,
                                                  int gridIdx )
{
   tnlStaticAssert( Matrix::DeviceType::DeviceType == tnlCudaDevice, );
   const typename Matrix::IndexType rowIdx = ( gridIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( rowIdx < matrix->getRows() )
      ( *outVector )[ rowIdx ] = matrix->rowVectorProduct( rowIdx, *inVector );
}
#endif

template< typename Matrix,
          typename InVector,
          typename OutVector >
void tnlMatrixVectorProductCuda( const Matrix& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
{
#ifdef HAVE_CUDA
   typedef typename Matrix::IndexType IndexType;
   Matrix* kernel_this = tnlCuda::passToDevice( matrix );
   InVector* kernel_inVector = tnlCuda::passToDevice( inVector );
   OutVector* kernel_outVector = tnlCuda::passToDevice( outVector );
   dim3 cudaBlockSize( 256 ), cudaGridSize( tnlCuda::getMaxGridSize() );
   const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
   const IndexType cudaGrids = roundUpDivision( cudaBlocks, tnlCuda::getMaxGridSize() );
   for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
   {
      if( gridIdx == cudaGrids - 1 )
         cudaGridSize.x = cudaBlocks % tnlCuda::getMaxGridSize();
      tnlMatrixVectorProductCudaKernel<<< cudaGridSize, cudaBlockSize >>>
                                     ( kernel_this,
                                       kernel_inVector,
                                       kernel_outVector,
                                       gridIdx );
   }
   tnlCuda::freeFromDevice( kernel_this );
   tnlCuda::freeFromDevice( kernel_inVector );
   tnlCuda::freeFromDevice( kernel_outVector );
   TNL_CHECK_CUDA_DEVICE;
#endif
}

#endif /* TNLMATRIX_IMPL_H_ */
