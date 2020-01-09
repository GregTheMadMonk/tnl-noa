/***************************************************************************
                          Tridiagonal.hpp  -  description
                             -------------------
    begin                : Nov 30, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Assert.h>
#include <TNL/Matrices/Tridiagonal.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Matrices {

template< typename Device >
class TridiagonalDeviceDependentCode;

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
Tridiagonal()
{
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
Tridiagonal( const IndexType rows, const IndexType columns )
{
   this->setDimensions( rows, columns );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
auto
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
getView() -> ViewType
{
   return ViewType( this->values.getView(), indexer );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
auto
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
getConstView() const -> ConstViewType
{
   return ConstViewType( this->values.getConstView(), indexer );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
String
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
getSerializationType()
{
   return String( "Matrices::Tridiagonal< " ) +
          TNL::getSerializationType< RealType >() + ", [any_device], " +
          TNL::getSerializationType< IndexType >() + ", " +
          ( RowMajorOrder ? "true" : "false" ) + ", [any_allocator] >";
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
String
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
void
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
setDimensions( const IndexType rows, const IndexType columns )
{
   Matrix< Real, Device, Index >::setDimensions( rows, columns );
   this->indexer.setDimensions( rows, columns );
   this->values.setSize( this->indexer.getStorageSize() );
   this->values = 0.0;
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
 //  template< typename Vector >
void
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
setCompressedRowLengths( const ConstCompressedRowLengthsVectorView rowLengths )
{
   if( max( rowLengths ) > 3 )
      throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
   if( rowLengths.getElement( 0 ) > 2 )
      throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
   const IndexType diagonalLength = min( this->getRows(), this->getColumns() );
   if( this->getRows() > this->getColumns() )
      if( rowLengths.getElement( this->getRows()-1 ) > 1 )
         throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
   if( this->getRows() == this->getColumns() )
      if( rowLengths.getElement( this->getRows()-1 ) > 2 )
         throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
   if( this->getRows() < this->getColumns() )
      if( rowLengths.getElement( this->getRows()-1 ) > 3 )
         throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
Index
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
getRowLength( const IndexType row ) const
{
   const IndexType diagonalLength = min( this->getRows(), this->getColumns() );
   if( row == 0 )
      return 2;
   if( row > 0 && row < diagonalLength - 1 )
      return 3;
   if( this->getRows() > this->getColumns() )
      return 1;
   if( this->getRows() == this->getColumns() )
      return 2;
   return 3;
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
Index
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
getMaxRowLength() const
{
   return 3;
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
   template< typename Real_, typename Device_, typename Index_, bool RowMajorOrder_, typename RealAllocator_ >
void
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
setLike( const Tridiagonal< Real_, Device_, Index_, RowMajorOrder_, RealAllocator_ >& m )
{
   this->setDimensions( m.getRows(), m.getColumns() );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
Index
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
getNumberOfMatrixElements() const
{
   return 3 * min( this->getRows(), this->getColumns() );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
Index
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
getNumberOfNonzeroMatrixElements() const
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
          bool RowMajorOrder,
          typename RealAllocator >
Index
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
getMaxRowlength() const
{
   return 3;
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
void
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
reset()
{
   Matrix< Real, Device, Index >::reset();
   this->values.reset();
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
   template< typename Real_, typename Device_, typename Index_, bool RowMajorOrder_, typename RealAllocator_ >
bool
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
operator == ( const Tridiagonal< Real_, Device_, Index_, RowMajorOrder_, RealAllocator_ >& matrix ) const
{
   if( RowMajorOrder == RowMajorOrder_ )
      return this->values == matrix.values;
   else
   {
      TNL_ASSERT( false, "TODO" );
   }
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
   template< typename Real_, typename Device_, typename Index_, bool RowMajorOrder_, typename RealAllocator_ >
bool
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
operator != ( const Tridiagonal< Real_, Device_, Index_, RowMajorOrder_, RealAllocator_ >& matrix ) const
{
   return ! this->operator==( matrix );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
void
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
setValue( const RealType& v )
{
   this->values = v;
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
__cuda_callable__
auto
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
getRow( const IndexType& rowIdx ) const -> const RowView
{
   return RowView( this->values.getView(), this->indexer );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
__cuda_callable__
auto
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
getRow( const IndexType& rowIdx ) -> RowView
{
   return RowView( this->values.getView(), this->indexer );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
bool
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
setElement( const IndexType row, const IndexType column, const RealType& value )
{
   TNL_ASSERT_GE( row, 0, "" );
   TNL_ASSERT_LT( row, this->getRows(), "" );
   TNL_ASSERT_GE( column, 0, "" );
   TNL_ASSERT_LT( column, this->getColumns(), "" );
   if( abs( row - column ) > 1 )
      throw std::logic_error( "Wrong matrix element coordinates in tridiagonal matrix." );
   this->values.setElement( this->getElementIndex( row, column ), value );
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
bool
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
addElement( const IndexType row,
            const IndexType column,
            const RealType& value,
            const RealType& thisElementMultiplicator )
{
   TNL_ASSERT_GE( row, 0, "" );
   TNL_ASSERT_LT( row, this->getRows(), "" );
   TNL_ASSERT_GE( column, 0, "" );
   TNL_ASSERT_LT( column, this->getColumns(), "" );
   if( abs( row - column ) > 1 )
      throw std::logic_error( "Wrong matrix element coordinates in tridiagonal matrix." );
   const Index i = this->getElementIndex( row, column );
   this->values.setElement( i, thisElementMultiplicator * this->values.getElement( i ) + value );
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
Real
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
getElement( const IndexType row, const IndexType column ) const
{
   TNL_ASSERT_GE( row, 0, "" );
   TNL_ASSERT_LT( row, this->getRows(), "" );
   TNL_ASSERT_GE( column, 0, "" );
   TNL_ASSERT_LT( column, this->getColumns(), "" );

   if( abs( column - row ) > 1 )
      return 0.0;
   return this->values.getElement( this->getElementIndex( row, column ) );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
rowsReduction( IndexType first, IndexType last, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const
{
   const auto values_view = this->values.getConstView();
   const auto indexer_ = this->indexer;
   const auto rows = this->getRows();
   const auto columns = this->getColumns();
   const auto size = this->size;
   auto f = [=] __cuda_callable__ ( IndexType rowIdx ) mutable {
      //bool compute;
      if( rowIdx == 0 )
      {
         IndexType i_0 = indexer.getGlobalIndex( 0, 0 );
         IndexType i_1 = indexer.getGlobalIndex( 0, 1 );
         keep( 0, reduce( fetch( 0, 0, i_0, values_view[ i_0 ] ),
                          fetch( 0, 1, i_1, values_view[ i_1 ] ) ) );
         return;
      }
      if( rowIdx < size || columns > rows )
      {
         IndexType i_0 = indexer.getGlobalIndex( rowIdx, 0 );
         IndexType i_1 = indexer.getGlobalIndex( rowIdx, 1 );
         IndexType i_2 = indexer.getGlobalIndex( rowIdx, 2 );

         keep( rowIdx, reduce( reduce( fetch( rowIdx, rowIdx - 1, i_0, values_view[ i_0 ] ),
                                       fetch( rowIdx, rowIdx, i_1, values_view[ i_1 ] ) ),
                               fetch( rowIdx, rowIdx + 1, i_2, values_view[ i_2] ) ) );
         return;
      }
      if( rows == columns )
      {
         IndexType i_0 = indexer.getGlobalIndex( rowIdx, 0 );
         IndexType i_1 = indexer.getGlobalIndex( rowIdx, 1 );
         keep( rowIdx, reduce( fetch( rowIdx, rowIdx - 1, i_0, values_view[ i_0 ] ),
                               fetch( rowIdx, rowIdx, i_1, values_view[ i_1 ] ) ) );
      }
      else
      {
         IndexType i_0 = indexer.getGlobalIndex( rowIdx, 0 );
         keep( rowIdx, fetch( rowIdx, rowIdx, i_0, values_view[ i_0 ] ) );
      }
   };
   Algorithms::ParallelFor< DeviceType >::exec( first, last, f );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
allRowsReduction( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const
{
   this->rowsReduction( 0, this->getRows(), fetch, reduce, keep, zero );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
   template< typename Function >
void
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
forRows( IndexType first, IndexType last, Function& function ) const
{
   const auto values_view = this->values.getConstView();
   const auto indexer_ = this->indexer;
   const auto rows = this->getRows();
   const auto columns = this->getColumns();
   const auto size = this->size;
   auto f = [=] __cuda_callable__ ( IndexType rowIdx ) mutable {
      //bool compute;
      if( rowIdx == 0 )
      {
         IndexType i_0 = indexer.getGlobalIndex( 0, 0 );
         IndexType i_1 = indexer.getGlobalIndex( 0, 1 );
         function( 0, 1, rowIdx,     values_view[ i_0 ] );
         function( 0, 2, rowIdx + 1, values_view[ i_1 ] );
         return;
      }
      if( rowIdx < size || columns > rows )
      {
         IndexType i_0 = indexer.getGlobalIndex( rowIdx, 0 );
         IndexType i_1 = indexer.getGlobalIndex( rowIdx, 1 );
         IndexType i_2 = indexer.getGlobalIndex( rowIdx, 2 );
         function( rowIdx, 0, rowIdx - 1, values_view[ i_0 ] );
         function( rowIdx, 1, rowIdx,     values_view[ i_1 ] );
         function( rowIdx, 2, rowIdx + 1, values_view[ i_2 ] );
         return;
      }
      if( rows == columns )
      {
         IndexType i_0 = indexer.getGlobalIndex( rowIdx, 0 );
         IndexType i_1 = indexer.getGlobalIndex( rowIdx, 1 );
         function( rowIdx, 0, rowIdx - 1, values_view[ i_0 ] );
         function( rowIdx, 1, rowIdx,     values_view[ i_1 ] );
      }
      else
      {
         IndexType i_0 = indexer.getGlobalIndex( rowIdx, 0 );
         function( rowIdx, 0, rowIdx, values_view[ i_0 ] );
      }
   };
   Algorithms::ParallelFor< DeviceType >::exec( first, last, f );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
  template< typename Function >
void
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
forRows( IndexType first, IndexType last, Function& function )
{
   const auto values_view = this->values.getConstView();
   const auto indexer_ = this->indexer;
   const auto rows = this->getRows();
   const auto columns = this->getColumns();
   const auto size = this->size;
   auto f = [=] __cuda_callable__ ( IndexType rowIdx ) mutable {
      //bool compute;
      if( rowIdx == 0 )
      {
         IndexType i_0 = indexer.getGlobalIndex( 0, 0 );
         IndexType i_1 = indexer.getGlobalIndex( 0, 1 );
         function( 0, 1, rowIdx,     values_view[ i_0 ] );
         function( 0, 2, rowIdx + 1, values_view[ i_1 ] );
         return;
      }
      if( rowIdx < size || columns > rows )
      {
         IndexType i_0 = indexer.getGlobalIndex( rowIdx, 0 );
         IndexType i_1 = indexer.getGlobalIndex( rowIdx, 1 );
         IndexType i_2 = indexer.getGlobalIndex( rowIdx, 2 );
         function( rowIdx, 0, rowIdx - 1, values_view[ i_0 ] );
         function( rowIdx, 1, rowIdx,     values_view[ i_1 ] );
         function( rowIdx, 2, rowIdx + 1, values_view[ i_2 ] );
         return;
      }
      if( rows == columns )
      {
         IndexType i_0 = indexer.getGlobalIndex( rowIdx, 0 );
         IndexType i_1 = indexer.getGlobalIndex( rowIdx, 1 );
         function( rowIdx, 0, rowIdx - 1, values_view[ i_0 ] );
         function( rowIdx, 1, rowIdx,     values_view[ i_1 ] );
      }
      else
      {
         IndexType i_0 = indexer.getGlobalIndex( rowIdx, 0 );
         function( rowIdx, 0, rowIdx, values_view[ i_0 ] );
      }
   };
   Algorithms::ParallelFor< DeviceType >::exec( first, last, f );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
   template< typename Function >
void
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
forAllRows( Function& function ) const
{
   this->forRows( 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
   template< typename Function >
void
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
forAllRows( Function& function )
{
   this->forRows( 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
template< typename Vector >
__cuda_callable__
typename Vector::RealType 
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
rowVectorProduct( const IndexType row, const Vector& vector ) const
{
   return TridiagonalDeviceDependentCode< Device >::
             rowVectorProduct( this->rows,
                               this->values,
                               row,
                               vector );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
   template< typename InVector,
             typename OutVector >
void 
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
vectorProduct( const InVector& inVector, OutVector& outVector ) const
{
   TNL_ASSERT( this->getColumns() == inVector.getSize(),
            std::cerr << "Matrix columns: " << this->getColumns() << std::endl
                 << "Vector size: " << inVector.getSize() << std::endl );
   TNL_ASSERT( this->getRows() == outVector.getSize(),
               std::cerr << "Matrix rows: " << this->getRows() << std::endl
                    << "Vector size: " << outVector.getSize() << std::endl );

   //DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
   template< typename Real_, typename Device_, typename Index_, bool RowMajorOrder_, typename RealAllocator_ >
void
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
addMatrix( const Tridiagonal< Real_, Device_, Index_, RowMajorOrder_, RealAllocator_ >& matrix,
           const RealType& matrixMultiplicator,
           const RealType& thisMatrixMultiplicator )
{
   TNL_ASSERT( this->getRows() == matrix.getRows(),
            std::cerr << "This matrix columns: " << this->getColumns() << std::endl
                 << "This matrix rows: " << this->getRows() << std::endl );

   if( thisMatrixMultiplicator == 1.0 )
      this->values += matrixMultiplicator * matrix.values;
   else
      this->values = thisMatrixMultiplicator * this->values + matrixMultiplicator * matrix.values;
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Real2,
          typename Index,
          typename Index2 >
__global__ void TridiagonalTranspositionCudaKernel( const Tridiagonal< Real2, Devices::Cuda, Index2 >* inMatrix,
                                                             Tridiagonal< Real, Devices::Cuda, Index >* outMatrix,
                                                             const Real matrixMultiplicator,
                                                             const Index gridIdx )
{
   const Index rowIdx = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( rowIdx < inMatrix->getRows() )
   {
      if( rowIdx > 0 )
        outMatrix->setElementFast( rowIdx-1,
                                   rowIdx,
                                   matrixMultiplicator * inMatrix->getElementFast( rowIdx, rowIdx-1 ) );
      outMatrix->setElementFast( rowIdx,
                                 rowIdx,
                                 matrixMultiplicator * inMatrix->getElementFast( rowIdx, rowIdx ) );
      if( rowIdx < inMatrix->getRows()-1 )
         outMatrix->setElementFast( rowIdx+1,
                                    rowIdx,
                                    matrixMultiplicator * inMatrix->getElementFast( rowIdx, rowIdx+1 ) );
   }
}
#endif

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
   template< typename Real2, typename Index2 >
void Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::getTransposition( const Tridiagonal< Real2, Device, Index2 >& matrix,
                                                                    const RealType& matrixMultiplicator )
{
   TNL_ASSERT( this->getRows() == matrix.getRows(),
               std::cerr << "This matrix rows: " << this->getRows() << std::endl
                    << "That matrix rows: " << matrix.getRows() << std::endl );
   if( std::is_same< Device, Devices::Host >::value )
   {
      const IndexType& rows = matrix.getRows();
      for( IndexType i = 1; i < rows; i++ )
      {
         RealType aux = matrix. getElement( i, i - 1 );
         this->setElement( i, i - 1, matrix.getElement( i - 1, i ) );
         this->setElement( i, i, matrix.getElement( i, i ) );
         this->setElement( i - 1, i, aux );
      }
   }
   if( std::is_same< Device, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      Tridiagonal* kernel_this = Cuda::passToDevice( *this );
      typedef  Tridiagonal< Real2, Device, Index2 > InMatrixType;
      InMatrixType* kernel_inMatrix = Cuda::passToDevice( matrix );
      dim3 cudaBlockSize( 256 ), cudaGridSize( Cuda::getMaxGridSize() );
      const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
      const IndexType cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridSize() );
      for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
      {
         if( gridIdx == cudaGrids - 1 )
            cudaGridSize.x = cudaBlocks % Cuda::getMaxGridSize();
         TridiagonalTranspositionCudaKernel<<< cudaGridSize, cudaBlockSize >>>
                                                    ( kernel_inMatrix,
                                                      kernel_this,
                                                      matrixMultiplicator,
                                                      gridIdx );
      }
      Cuda::freeFromDevice( kernel_this );
      Cuda::freeFromDevice( kernel_inMatrix );
      TNL_CHECK_CUDA_DEVICE;
#endif
   }
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
   template< typename Vector1, typename Vector2 >
__cuda_callable__
void Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::performSORIteration( const Vector1& b,
                                                              const IndexType row,
                                                              Vector2& x,
                                                              const RealType& omega ) const
{
   RealType sum( 0.0 );
   if( row > 0 )
      sum += this->getElementFast( row, row - 1 ) * x[ row - 1 ];
   if( row < this->getColumns() - 1 )
      sum += this->getElementFast( row, row + 1 ) * x[ row + 1 ];
   x[ row ] = ( 1.0 - omega ) * x[ row ] + omega / this->getElementFast( row, row ) * ( b[ row ] - sum );
}


// copy assignment
template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >&
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::operator=( const Tridiagonal& matrix )
{
   this->setLike( matrix );
   this->values = matrix.values;
   return *this;
}

// cross-device copy assignment
template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
   template< typename Real_, typename Device_, typename Index_, bool RowMajorOrder_, typename RealAllocator_ >
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >&
Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
operator=( const Tridiagonal< Real_, Device_, Index_, RowMajorOrder_, RealAllocator_ >& matrix )
{
   static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value,
                  "unknown device" );
   static_assert( std::is_same< Device_, Devices::Host >::value || std::is_same< Device_, Devices::Cuda >::value,
                  "unknown device" );

   this->setLike( matrix );

   throw Exceptions::NotImplementedError("Cross-device assignment for the Tridiagonal format is not implemented yet.");
}


template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
void Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::save( File& file ) const
{
   Matrix< Real, Device, Index >::save( file );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
void Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::load( File& file )
{
   Matrix< Real, Device, Index >::load( file );
   this->indexer.setDimensions( this->getRows(), this->getColumns() );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
void Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
void Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
void Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::print( std::ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      for( IndexType column = row - 1; column < row + 2; column++ )
         if( column >= 0 && column < this->columns )
            str << " Col:" << column << "->" << this->getElement( row, column ) << "\t";
      str << std::endl;
   }
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
__cuda_callable__
Index Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >::
getElementIndex( const IndexType row, const IndexType localIdx ) const
{
   TNL_ASSERT_GE( row, 0, "" );
   TNL_ASSERT_LT( row, this->getRows(), "" );
   TNL_ASSERT_GE( localIdx, 0, "" );
   TNL_ASSERT_LT( localIdx, 3, "" );
   return this->indexer.getGlobalIndex( row, localIdx );
}

/*
template<>
class TridiagonalDeviceDependentCode< Devices::Host >
{
   public:

      typedef Devices::Host Device;

      template< typename Index >
      __cuda_callable__
      static Index getElementIndex( const Index rows,
                                    const Index row,
                                    const Index column )
      {
         return 2*row + column;
      }

      template< typename Vector,
                typename Index,
                typename ValuesType  >
      __cuda_callable__
      static typename Vector::RealType rowVectorProduct( const Index rows,
                                                         const ValuesType& values,
                                                         const Index row,
                                                         const Vector& vector )
      {
         if( row == 0 )
            return vector[ 0 ] * values[ 0 ] +
                   vector[ 1 ] * values[ 1 ];
         Index i = 3 * row;
         if( row == rows - 1 )
            return vector[ row - 1 ] * values[ i - 1 ] +
                   vector[ row ] * values[ i ];
         return vector[ row - 1 ] * values[ i - 1 ] +
                vector[ row ] * values[ i ] +
                vector[ row + 1 ] * values[ i + 1 ];
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >& matrix,
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

template<>
class TridiagonalDeviceDependentCode< Devices::Cuda >
{
   public:

      typedef Devices::Cuda Device;

      template< typename Index >
      __cuda_callable__
      static Index getElementIndex( const Index rows,
                                    const Index row,
                                    const Index column )
      {
         return ( column - row + 1 )*rows + row - 1;
      }

      template< typename Vector,
                typename Index,
                typename ValuesType >
      __cuda_callable__
      static typename Vector::RealType rowVectorProduct( const Index rows,
                                                         const ValuesType& values,
                                                         const Index row,
                                                         const Vector& vector )
      {
         if( row == 0 )
            return vector[ 0 ] * values[ 0 ] +
                   vector[ 1 ] * values[ rows - 1 ];
         Index i = row - 1;
         if( row == rows - 1 )
            return vector[ row - 1 ] * values[ i ] +
                   vector[ row ] * values[ i + rows ];
         return vector[ row - 1 ] * values[ i ] +
                vector[ row ] * values[ i + rows ] +
                vector[ row + 1 ] * values[ i + 2*rows ];
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const Tridiagonal< Real, Device, Index, RowMajorOrder, RealAllocator >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         MatrixVectorProductCuda( matrix, inVector, outVector );
      }
};
 */

} // namespace Matrices
} // namespace TNL
