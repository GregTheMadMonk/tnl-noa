/***************************************************************************
                          Matrix_impl.h  -  description
                             -------------------
    begin                : Dec 18, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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
          typename Index,
          typename RealAllocator >
Matrix< Real, Device, Index, RealAllocator >::
Matrix( const RealAllocatorType& allocator )
: rows( 0 ),
  columns( 0 ),
  values( allocator )
{
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
Matrix< Real, Device, Index, RealAllocator >::
Matrix( const IndexType rows_, const IndexType columns_, const RealAllocatorType& allocator )
: rows( rows_ ),
  columns( columns_ ),
  values( allocator )
{
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
void Matrix< Real, Device, Index, RealAllocator >::setDimensions( const IndexType rows,
                                                   const IndexType columns )
{
   TNL_ASSERT( rows > 0 && columns > 0,
               std::cerr << " rows = " << rows << " columns = " << columns );
   this->rows = rows;
   this->columns = columns;
}

/*template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
void Matrix< Real, Device, Index, RealAllocator >::getCompressedRowLengths( CompressedRowLengthsVector& rowLengths ) const
{
   rowLengths.setSize( this->getRows() );
   getCompressedRowLengths( rowLengths.getView() );
}*/

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
void Matrix< Real, Device, Index, RealAllocator >::getCompressedRowLengths( CompressedRowLengthsVectorView rowLengths ) const
{
   TNL_ASSERT_EQ( rowLengths.getSize(), this->getRows(), "invalid size of the rowLengths vector" );
   for( IndexType row = 0; row < this->getRows(); row++ )
      rowLengths.setElement( row, this->getRowLength( row ) );
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
   template< typename Matrix_ >
void Matrix< Real, Device, Index, RealAllocator >::setLike( const Matrix_& matrix )
{
   setDimensions( matrix.getRows(), matrix.getColumns() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
Index Matrix< Real, Device, Index, RealAllocator >::getAllocatedElementsCount() const
{
   return this->values.getSize();
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
Index Matrix< Real, Device, Index, RealAllocator >::getNumberOfNonzeroMatrixElements() const
{
   const auto values_view = this->values.getConstView();
   auto fetch = [=] __cuda_callable__ ( const IndexType i ) -> IndexType {
      return ( values_view[ i ] != 0.0 );
   };
   return Algorithms::Reduction< DeviceType >::reduce( this->values.getSize(), std::plus<>{}, fetch, 0 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
__cuda_callable__
Index Matrix< Real, Device, Index, RealAllocator >::getRows() const
{
   return this->rows;
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
__cuda_callable__
Index Matrix< Real, Device, Index, RealAllocator >::getColumns() const
{
   return this->columns;
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
const typename Matrix< Real, Device, Index, RealAllocator >::ValuesVectorType&
Matrix< Real, Device, Index, RealAllocator >::
getValues() const
{
   return this->values;
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
typename Matrix< Real, Device, Index, RealAllocator >::ValuesVectorType&
Matrix< Real, Device, Index, RealAllocator >::
getValues()
{
   return this->values;
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
void Matrix< Real, Device, Index, RealAllocator >::reset()
{
   this->rows = 0;
   this->columns = 0;
   this->values.reset();
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
   template< typename MatrixT >
bool Matrix< Real, Device, Index, RealAllocator >::operator == ( const MatrixT& matrix ) const
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
          typename Index,
          typename RealAllocator >
   template< typename MatrixT >
bool Matrix< Real, Device, Index, RealAllocator >::operator != ( const MatrixT& matrix ) const
{
   return ! operator == ( matrix );
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
void Matrix< Real, Device, Index, RealAllocator >::save( File& file ) const
{
   Object::save( file );
   file.save( &this->rows );
   file.save( &this->columns );
   file << this->values;
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
void Matrix< Real, Device, Index, RealAllocator >::load( File& file )
{
   Object::load( file );
   file.load( &this->rows );
   file.load( &this->columns );
   file >> this->values;
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
void Matrix< Real, Device, Index, RealAllocator >::print( std::ostream& str ) const
{
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
__cuda_callable__
const Index&
Matrix< Real, Device, Index, RealAllocator >::
getNumberOfColors() const
{
   return this->numberOfColors;
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
void
Matrix< Real, Device, Index, RealAllocator >::
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

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
void
Matrix< Real, Device, Index, RealAllocator >::
copyFromHostToCuda( Matrix< Real, Devices::Host, Index >& matrix )
{
    this->numberOfColors = matrix.getNumberOfColors();
    this->columns = matrix.getColumns();
    this->rows = matrix.getRows();

    this->values.setSize( matrix.getValuesSize() );
}

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

} // namespace Matrices
} // namespace TNL
