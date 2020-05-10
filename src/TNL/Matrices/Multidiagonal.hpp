/***************************************************************************
                          Multidiagonal.hpp  -  description
                             -------------------
    begin                : Oct 13, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <sstream>
#include <TNL/Assert.h>
#include <TNL/Matrices/Multidiagonal.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Matrices {

template< typename Device >
class MultidiagonalDeviceDependentCode;

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
Multidiagonal()
{
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Vector >
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
Multidiagonal( const IndexType rows,
               const IndexType columns,
               const Vector& diagonalsShifts )
{
   TNL_ASSERT_GT( diagonalsShifts.getSize(), 0, "Cannot construct mutltidiagonal matrix with no diagonals shifts." );
   this->setDimensions( rows, columns, diagonalsShifts );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
auto
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getView() const -> ViewType
{
   // TODO: fix when getConstView works
   return ViewType( const_cast< Multidiagonal* >( this )->values.getView(),
                    const_cast< Multidiagonal* >( this )->diagonalsShifts.getView(),
                    const_cast< Multidiagonal* >( this )->hostDiagonalsShifts.getView(),
                    indexer );
}

/*template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
auto
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getConstView() const -> ConstViewType
{
   return ConstViewType( this->values.getConstView(), indexer );
}*/

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
String
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getSerializationType()
{
   return String( "Matrices::Multidiagonal< " ) +
          TNL::getSerializationType< RealType >() + ", [any_device], " +
          TNL::getSerializationType< IndexType >() + ", " +
          ( Organization ? "true" : "false" ) + ", [any_allocator], [any_allocator] >";
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
String
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Vector >
void
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
setDimensions( const IndexType rows,
               const IndexType columns,
               const Vector& diagonalsShifts )
{
   Matrix< Real, Device, Index >::setDimensions( rows, columns );
   this->diagonalsShifts = diagonalsShifts;
   this->hostDiagonalsShifts = diagonalsShifts;
   const IndexType minShift = min( diagonalsShifts );
   IndexType nonemptyRows = min( rows, columns );
   if( rows > columns && minShift < 0 )
      nonemptyRows = min( rows, nonemptyRows - minShift );
   this->indexer.set( rows, columns, diagonalsShifts.getSize(), nonemptyRows );
   this->values.setSize( this->indexer.getStorageSize() );
   this->values = 0.0;
   this->view = this->getView();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
 //  template< typename Vector >
void
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
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
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
const Index&
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getDiagonalsCount() const
{
   return this->view.getDiagonalsCount();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
auto
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getDiagonalsShifts() const -> const DiagonalsShiftsType&
{
   return this->diagonalsShifts;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Vector >
void
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getCompressedRowLengths( Vector& rowLengths ) const
{
   return this->view.getCompressedRowLengths( rowLengths );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
Index
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getNonemptyRowsCount() const
{
   return this->indexer.getNonemptyRowsCount();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
Index
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getRowLength( const IndexType row ) const
{
   return this->view.getRowLength( row );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
Index
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getMaxRowLength() const
{
   return this->view.getMaxRowLength();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
void
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
setLike( const Multidiagonal< Real_, Device_, Index_, Organization_, RealAllocator_ >& m )
{
   this->setDimensions( m.getRows(), m.getColumns(), m.getDiagonalsShifts() );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
Index
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getNumberOfNonzeroMatrixElements() const
{
   return this->view.getNumberOfNonzeroMatrixElements();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
void
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
reset()
{
   Matrix< Real, Device, Index >::reset();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
bool
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
operator == ( const Multidiagonal< Real_, Device_, Index_, Organization_, RealAllocator_ >& matrix ) const
{
   if( Organization == Organization_ )
      return this->values == matrix.values;
   else
   {
      TNL_ASSERT( false, "TODO" );
   }
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
bool
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
operator != ( const Multidiagonal< Real_, Device_, Index_, Organization_, RealAllocator_ >& matrix ) const
{
   return ! this->operator==( matrix );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
void
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
setValue( const RealType& v )
{
   this->view.setValue( v );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
auto
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getRow( const IndexType& rowIdx ) const -> const RowView
{
   return this->view.getRow( rowIdx );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
auto
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getRow( const IndexType& rowIdx ) -> RowView
{
   return this->view.getRow( rowIdx );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
void
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
setElement( const IndexType row, const IndexType column, const RealType& value )
{
   this->view.setElement( row, column, value );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
void
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
addElement( const IndexType row,
            const IndexType column,
            const RealType& value,
            const RealType& thisElementMultiplicator )
{
   this->view.addElement( row, column, value, thisElementMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
Real
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getElement( const IndexType row, const IndexType column ) const
{
   return this->view.getElement( row, column );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
rowsReduction( IndexType first, IndexType last, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const
{
   this->view.rowsReduction( first, last, fetch, reduce, keep, zero );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
allRowsReduction( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const
{
   this->view.rowsReduction( 0, this->getRows(), fetch, reduce, keep, zero );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
forRows( IndexType first, IndexType last, Function& function ) const
{
   this->view.forRows( first, last, function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
  template< typename Function >
void
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
forRows( IndexType first, IndexType last, Function& function )
{
   this->view.forRows( first, last, function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
forAllRows( Function& function ) const
{
   this->view.forRows( 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
forAllRows( Function& function )
{
   this->view.forRows( 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Vector >
__cuda_callable__
typename Vector::RealType
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
rowVectorProduct( const IndexType row, const Vector& vector ) const
{
   return this->view.rowVectorProduct();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename InVector,
             typename OutVector >
void
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
vectorProduct( const InVector& inVector, OutVector& outVector ) const
{
   this->view.vectorProduct( inVector, outVector );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
void
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
addMatrix( const Multidiagonal< Real_, Device_, Index_, Organization_, RealAllocator_ >& matrix,
           const RealType& matrixMultiplicator,
           const RealType& thisMatrixMultiplicator )
{
   this->view.addMatrix( matrix.getView(), matrixMultiplicator, thisMatrixMultiplicator );
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Real2,
          typename Index,
          typename Index2 >
__global__ void MultidiagonalTranspositionCudaKernel( const Multidiagonal< Real2, Devices::Cuda, Index2 >* inMatrix,
                                                             Multidiagonal< Real, Devices::Cuda, Index >* outMatrix,
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
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Real2, typename Index2 >
void Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::getTransposition( const Multidiagonal< Real2, Device, Index2 >& matrix,
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
      Multidiagonal* kernel_this = Cuda::passToDevice( *this );
      typedef  Multidiagonal< Real2, Device, Index2 > InMatrixType;
      InMatrixType* kernel_inMatrix = Cuda::passToDevice( matrix );
      dim3 cudaBlockSize( 256 ), cudaGridSize( Cuda::getMaxGridSize() );
      const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
      const IndexType cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridSize() );
      for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
      {
         if( gridIdx == cudaGrids - 1 )
            cudaGridSize.x = cudaBlocks % Cuda::getMaxGridSize();
         MultidiagonalTranspositionCudaKernel<<< cudaGridSize, cudaBlockSize >>>
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
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Vector1, typename Vector2 >
__cuda_callable__
void Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::performSORIteration( const Vector1& b,
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
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >&
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::operator=( const Multidiagonal& matrix )
{
   this->setLike( matrix );
   this->values = matrix.values;
   return *this;
}

// cross-device copy assignment
template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_, typename IndexAllocator_ >
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >&
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
operator=( const Multidiagonal< Real_, Device_, Index_, Organization_, RealAllocator_, IndexAllocator_ >& matrix )
{
   using RHSMatrix = Multidiagonal< Real_, Device_, Index_, Organization_, RealAllocator_, IndexAllocator_ >;
   using RHSIndexType = typename RHSMatrix::IndexType;
   using RHSRealType = typename RHSMatrix::RealType;
   using RHSDeviceType = typename RHSMatrix::DeviceType;
   using RHSRealAllocatorType = typename RHSMatrix::RealAllocatorType;
   using RHSIndexAllocatorType = typename RHSMatrix::IndexAllocatorType;

   this->setLike( matrix );
   if( Organization == Organization_ )
      this->values = matrix.getValues();
   else
   {
      if( std::is_same< Device, Device_ >::value )
      {
         const auto matrix_view = matrix.getView();
         auto f = [=] __cuda_callable__ ( const IndexType& rowIdx, const IndexType& localIdx, const IndexType& column, Real& value, bool& compute ) mutable {
            value = matrix_view.getValues()[ matrix_view.getIndexer().getGlobalIndex( rowIdx, localIdx ) ];
         };
         this->forAllRows( f );
      }
      else
      {
         const IndexType maxRowLength = this->diagonalsShifts.getSize();
         const IndexType bufferRowsCount( 128 );
         const size_t bufferSize = bufferRowsCount * maxRowLength;
         Containers::Vector< RHSRealType, RHSDeviceType, RHSIndexType, RHSRealAllocatorType > matrixValuesBuffer( bufferSize );
         Containers::Vector< RHSIndexType, RHSDeviceType, RHSIndexType, RHSIndexAllocatorType > matrixColumnsBuffer( bufferSize );
         Containers::Vector< RealType, DeviceType, IndexType, RealAllocatorType > thisValuesBuffer( bufferSize );
         Containers::Vector< IndexType, DeviceType, IndexType, IndexAllocatorType > thisColumnsBuffer( bufferSize );
         auto matrixValuesBuffer_view = matrixValuesBuffer.getView();
         auto thisValuesBuffer_view = thisValuesBuffer.getView();

         IndexType baseRow( 0 );
         const IndexType rowsCount = this->getRows();
         while( baseRow < rowsCount )
         {
            const IndexType lastRow = min( baseRow + bufferRowsCount, rowsCount );

            ////
            // Copy matrix elements into buffer
            auto f1 = [=] __cuda_callable__ ( RHSIndexType rowIdx, RHSIndexType localIdx, RHSIndexType columnIndex, const RHSRealType& value, bool& compute ) mutable {
                  const IndexType bufferIdx = ( rowIdx - baseRow ) * maxRowLength + localIdx;
                  matrixValuesBuffer_view[ bufferIdx ] = value;
            };
            matrix.forRows( baseRow, lastRow, f1 );

            ////
            // Copy the source matrix buffer to this matrix buffer
            thisValuesBuffer_view = matrixValuesBuffer_view;

            ////
            // Copy matrix elements from the buffer to the matrix
            auto f2 = [=] __cuda_callable__ ( const IndexType rowIdx, const IndexType localIdx, const IndexType columnIndex, RealType& value, bool& compute  ) mutable {
               const IndexType bufferIdx = ( rowIdx - baseRow ) * maxRowLength + localIdx;
                  value = thisValuesBuffer_view[ bufferIdx ];
            };
            this->forRows( baseRow, lastRow, f2 );
            baseRow += bufferRowsCount;
         }
      }
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
void Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::save( File& file ) const
{
   Matrix< Real, Device, Index >::save( file );
   file << diagonalsShifts;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
void Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::load( File& file )
{
   Matrix< Real, Device, Index >::load( file );
   file >> this->diagonalsShifts;
   this->hostDiagonalsShifts = this->diagonalsShifts;
   const IndexType minShift = min( diagonalsShifts );
   IndexType nonemptyRows = min( this->getRows(), this->getColumns() );
   if( this->getRows() > this->getColumns() && minShift < 0 )
      nonemptyRows = min( this->getRows(), nonemptyRows - minShift );
   this->indexer.set( this->getRows(), this->getColumns(), diagonalsShifts.getSize(), nonemptyRows );
   this->view = this->getView();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
void Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
void Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
void
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
print( std::ostream& str ) const
{
   this->view.print( str );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
auto
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getIndexer() const -> const IndexerType&
{
   return this->indexer;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
auto
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getIndexer() -> IndexerType&
{
   return this->indexer;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
Index Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getElementIndex( const IndexType row, const IndexType column ) const
{
   IndexType localIdx = column - row;
   if( row > 0 )
      localIdx++;

   TNL_ASSERT_GE( localIdx, 0, "" );
   TNL_ASSERT_LT( localIdx, 3, "" );

   return this->indexer.getGlobalIndex( row, localIdx );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
Index
Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getPaddingIndex() const
{
   return this->view.getPaddingIndex();
}

/*
template<>
class MultidiagonalDeviceDependentCode< Devices::Host >
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
      static void vectorProduct( const Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >& matrix,
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
class MultidiagonalDeviceDependentCode< Devices::Cuda >
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
      static void vectorProduct( const Multidiagonal< Real, Device, Index, Organization, RealAllocator, IndexAllocator >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         MatrixVectorProductCuda( matrix, inVector, outVector );
      }
};
 */

} // namespace Matrices
} // namespace TNL
