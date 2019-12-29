/***************************************************************************
                          MatrixView.hpp  -  description
                             -------------------
    begin                : Dec 28, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/Matrix.h>
#include <TNL/Assert.h>
#include <TNL/Cuda/LaunchHelpers.h>
#include <TNL/Cuda/MemoryHelpers.h>
#include <TNL/Cuda/SharedMemory.h>

namespace TNL {
namespace Matrices {

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
MatrixView< Real, Device, Index >::
MatrixView()
: rows( 0 ),
  columns( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
MatrixView< Real, Device, Index >::
MatrixView( const IndexType rows_, 
            const IndexType columns_,
            const ValuesView& values_ )
 : rows( rows_ ), columns( columns_ ), values( values_ )
{
}

/*template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
auto
MatrixView< Real, Device, Index >::
getView() ->ViewType
{
   return ViewType( rows, columns, values.getView() );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
auto
MatrixView< Real, Device, Index >::
getConstView() const -> ConstViewType
{
   return ConstViewType( rows, columns, values.getConstView() );
}*/

template< typename Real,
          typename Device,
          typename Index >
void MatrixView< Real, Device, Index >::getCompressedRowLengths( CompressedRowLengthsVector& rowLengths ) const
{
   rowLengths.setSize( this->getRows() );
   getCompressedRowLengths( rowLengths.getView() );
}

template< typename Real,
          typename Device,
          typename Index >
void MatrixView< Real, Device, Index >::getCompressedRowLengths( CompressedRowLengthsVectorView rowLengths ) const
{
   TNL_ASSERT_EQ( rowLengths.getSize(), this->getRows(), "invalid size of the rowLengths vector" );
   for( IndexType row = 0; row < this->getRows(); row++ )
      rowLengths.setElement( row, this->getRowLength( row ) );
}

template< typename Real,
          typename Device,
          typename Index >
Index MatrixView< Real, Device, Index >::getNumberOfMatrixElements() const
{
   return this->values.getSize();
}

template< typename Real,
          typename Device,
          typename Index >
Index MatrixView< Real, Device, Index >::getNumberOfNonzeroMatrixElements() const
{
   const auto values_view = this->values.getConstView();
   auto fetch = [=] __cuda_callable__ ( const IndexType i ) -> IndexType {
      return ( values_view[ i ] != 0.0 );
   };
   return Algorithms::Reduction< DeviceType >::reduce( this->values.getSize(), std::plus<>{}, fetch, 0 );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index MatrixView< Real, Device, Index >::getRows() const
{
   return this->rows;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index MatrixView< Real, Device, Index >::getColumns() const
{
   return this->columns;
}

template< typename Real,
          typename Device,
          typename Index >
const typename MatrixView< Real, Device, Index >::ValuesView&
MatrixView< Real, Device, Index >::
getValues() const
{
   return this->values;
}
   
template< typename Real,
          typename Device,
          typename Index >
typename MatrixView< Real, Device, Index >::ValuesView& 
MatrixView< Real, Device, Index >::
getValues()
{
   return this->values;
}

template< typename Real,
          typename Device,
          typename Index >
void MatrixView< Real, Device, Index >::reset()
{
   this->rows = 0;
   this->columns = 0;
   this->values.reset();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename MatrixT >
bool MatrixView< Real, Device, Index >::operator == ( const MatrixT& matrix ) const
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
   template< typename MatrixT >
bool MatrixView< Real, Device, Index >::operator != ( const MatrixT& matrix ) const
{
   return ! operator == ( matrix );
}

template< typename Real,
          typename Device,
          typename Index >
void MatrixView< Real, Device, Index >::save( File& file ) const
{
   Object::save( file );
   file.save( &this->rows );
   file.save( &this->columns );
   file << this->values;
}

template< typename Real,
          typename Device,
          typename Index >
void MatrixView< Real, Device, Index >::load( File& file )
{
   Object::load( file );
   file.load( &this->rows );
   file.load( &this->columns );
   file >> this->values;
}

template< typename Real,
          typename Device,
          typename Index >
void MatrixView< Real, Device, Index >::print( std::ostream& str ) const
{
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Index&
MatrixView< Real, Device, Index >::
getNumberOfColors() const
{
   return this->numberOfColors;
}

template< typename Real,
          typename Device,
          typename Index >
void 
MatrixView< Real, Device, Index >::
computeColorsVector(Containers::Vector<Index, Device, Index> &colorsVector)
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

/*
#ifdef HAVE_CUDA
template< typename Matrix,
          typename InVector,
          typename OutVector >
__global__ void MatrixVectorProductCudaKernel( const Matrix* matrix,
                                               const InVector* inVector,
                                               OutVector* outVector,
                                               int gridIdx )
{
   static_assert( std::is_same< typename Matrix::DeviceType, Devices::Cuda >::value, "" );
   const typename Matrix::IndexType rowIdx = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( rowIdx < matrix->getRows() )
      ( *outVector )[ rowIdx ] = matrix->rowVectorProduct( rowIdx, *inVector );
}
#endif

template< typename Matrix,
          typename InVector,
          typename OutVector >
void MatrixVectorProductCuda( const Matrix& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
{
#ifdef HAVE_CUDA
   typedef typename Matrix::IndexType IndexType;
   Matrix* kernel_this = Cuda::passToDevice( matrix );
   InVector* kernel_inVector = Cuda::passToDevice( inVector );
   OutVector* kernel_outVector = Cuda::passToDevice( outVector );
   dim3 cudaBlockSize( 256 ), cudaGridSize( Cuda::getMaxGridSize() );
   const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
   const IndexType cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridSize() );
   for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
   {
      if( gridIdx == cudaGrids - 1 )
         cudaGridSize.x = cudaBlocks % Cuda::getMaxGridSize();
      MatrixVectorProductCudaKernel<<< cudaGridSize, cudaBlockSize >>>
                                     ( kernel_this,
                                       kernel_inVector,
                                       kernel_outVector,
                                       gridIdx );
      TNL_CHECK_CUDA_DEVICE;
   }
   Cuda::freeFromDevice( kernel_this );
   Cuda::freeFromDevice( kernel_inVector );
   Cuda::freeFromDevice( kernel_outVector );
   TNL_CHECK_CUDA_DEVICE;
#endif
}
*/

} // namespace Matrices
} // namespace TNL
