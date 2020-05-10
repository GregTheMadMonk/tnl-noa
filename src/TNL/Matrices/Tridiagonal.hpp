/***************************************************************************
                          Tridiagonal.hpp  -  description
                             -------------------
    begin                : Nov 30, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <sstream>
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
          ElementsOrganization Organization,
          typename RealAllocator >
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
Tridiagonal()
{
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
Tridiagonal( const IndexType rows, const IndexType columns )
{
   this->setDimensions( rows, columns );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
auto
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
getView() const -> ViewType
{
   // TODO: fix when getConstView works
   return ViewType( const_cast< Tridiagonal* >( this )->values.getView(), indexer );
}

/*template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
auto
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
getConstView() const -> ConstViewType
{
   return ConstViewType( this->values.getConstView(), indexer );
}*/

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
String
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
getSerializationType()
{
   return String( "Matrices::Tridiagonal< " ) +
          TNL::getSerializationType< RealType >() + ", [any_device], " +
          TNL::getSerializationType< IndexType >() + ", " +
          ( Organization ? "true" : "false" ) + ", [any_allocator] >";
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
String
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
setDimensions( const IndexType rows, const IndexType columns )
{
   Matrix< Real, Device, Index >::setDimensions( rows, columns );
   this->indexer.setDimensions( rows, columns );
   this->values.setSize( this->indexer.getStorageSize() );
   this->values = 0.0;
   this->view = this->getView();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
 //  template< typename Vector >
void
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
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
          typename RealAllocator >
   template< typename Vector >
void
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
getCompressedRowLengths( Vector& rowLengths ) const
{
   return this->view.getCompressedRowLengths( rowLengths );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
Index
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
getRowLength( const IndexType row ) const
{
   return this->view.getRowLength( row );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
Index
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
getMaxRowLength() const
{
   return this->view.getMaxRowLength();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
void
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
setLike( const Tridiagonal< Real_, Device_, Index_, Organization_, RealAllocator_ >& m )
{
   this->setDimensions( m.getRows(), m.getColumns() );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
Index
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
getNumberOfNonzeroMatrixElements() const
{
   return this->view.getNumberOfNonzeroMatrixElements();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
reset()
{
   Matrix< Real, Device, Index >::reset();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
bool
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
operator == ( const Tridiagonal< Real_, Device_, Index_, Organization_, RealAllocator_ >& matrix ) const
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
          typename RealAllocator >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
bool
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
operator != ( const Tridiagonal< Real_, Device_, Index_, Organization_, RealAllocator_ >& matrix ) const
{
   return ! this->operator==( matrix );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
setValue( const RealType& v )
{
   this->view.setValue( v );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
__cuda_callable__
auto
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
getRow( const IndexType& rowIdx ) const -> const RowView
{
   return this->view.getRow( rowIdx );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
__cuda_callable__
auto
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
getRow( const IndexType& rowIdx ) -> RowView
{
   return this->view.getRow( rowIdx );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
setElement( const IndexType row, const IndexType column, const RealType& value )
{
   this->view.setElement( row, column, value );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
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
          typename RealAllocator >
Real
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
getElement( const IndexType row, const IndexType column ) const
{
   return this->view.getElement( row, column );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
rowsReduction( IndexType first, IndexType last, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const
{
   this->view.rowsReduction( first, last, fetch, reduce, keep, zero );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
allRowsReduction( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const
{
   this->view.rowsReduction( 0, this->getRows(), fetch, reduce, keep, zero );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Function >
void
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
forRows( IndexType first, IndexType last, Function& function ) const
{
   this->view.forRows( first, last, function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
  template< typename Function >
void
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
forRows( IndexType first, IndexType last, Function& function )
{
   this->view.forRows( first, last, function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Function >
void
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
forAllRows( Function& function ) const
{
   this->view.forRows( 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Function >
void
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
forAllRows( Function& function )
{
   this->view.forRows( 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
template< typename Vector >
__cuda_callable__
typename Vector::RealType
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
rowVectorProduct( const IndexType row, const Vector& vector ) const
{
   return this->view.rowVectorProduct();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename InVector,
             typename OutVector >
void
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
vectorProduct( const InVector& inVector, OutVector& outVector ) const
{
   this->view.vectorProduct( inVector, outVector );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
void
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
addMatrix( const Tridiagonal< Real_, Device_, Index_, Organization_, RealAllocator_ >& matrix,
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
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Real2, typename Index2 >
void Tridiagonal< Real, Device, Index, Organization, RealAllocator >::getTransposition( const Tridiagonal< Real2, Device, Index2 >& matrix,
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
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Vector1, typename Vector2 >
__cuda_callable__
void Tridiagonal< Real, Device, Index, Organization, RealAllocator >::performSORIteration( const Vector1& b,
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
          typename RealAllocator >
Tridiagonal< Real, Device, Index, Organization, RealAllocator >&
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::operator=( const Tridiagonal& matrix )
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
          typename RealAllocator >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
Tridiagonal< Real, Device, Index, Organization, RealAllocator >&
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
operator=( const Tridiagonal< Real_, Device_, Index_, Organization_, RealAllocator_ >& matrix )
{
   static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value,
                  "unknown device" );
   static_assert( std::is_same< Device_, Devices::Host >::value || std::is_same< Device_, Devices::Cuda >::value,
                  "unknown device" );

   this->setLike( matrix );
   if( Organization == Organization_ )
      this->values = matrix.getValues();
   else
   {
      if( std::is_same< Device, Device_ >::value )
      {
         const auto matrix_view = matrix.getView();
         auto f = [=] __cuda_callable__ ( const IndexType& rowIdx, const IndexType& localIdx, const IndexType& column, Real& value ) mutable {
            value = matrix_view.getValues()[ matrix_view.getIndexer().getGlobalIndex( rowIdx, localIdx ) ];
         };
         this->forAllRows( f );
      }
      else
      {
         Tridiagonal< Real, Device, Index, Organization_ > auxMatrix;
         auxMatrix = matrix;
         const auto matrix_view = auxMatrix.getView();
         auto f = [=] __cuda_callable__ ( const IndexType& rowIdx, const IndexType& localIdx, const IndexType& column, Real& value ) mutable {
            value = matrix_view.getValues()[ matrix_view.getIndexer().getGlobalIndex( rowIdx, localIdx ) ];
         };
         this->forAllRows( f );
      }
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void Tridiagonal< Real, Device, Index, Organization, RealAllocator >::save( File& file ) const
{
   Matrix< Real, Device, Index >::save( file );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void Tridiagonal< Real, Device, Index, Organization, RealAllocator >::load( File& file )
{
   Matrix< Real, Device, Index >::load( file );
   this->indexer.setDimensions( this->getRows(), this->getColumns() );
   this->view = this->getView();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void Tridiagonal< Real, Device, Index, Organization, RealAllocator >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void Tridiagonal< Real, Device, Index, Organization, RealAllocator >::load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
print( std::ostream& str ) const
{
   this->view.print( str );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
auto
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
getIndexer() const -> const IndexerType&
{
   return this->indexer;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
auto
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
getIndexer() -> IndexerType&
{
   return this->indexer;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
__cuda_callable__
Index
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
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
          typename RealAllocator >
__cuda_callable__
Index
Tridiagonal< Real, Device, Index, Organization, RealAllocator >::
getPaddingIndex() const
{
   return this->view.getPaddingIndex();
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
      static void vectorProduct( const Tridiagonal< Real, Device, Index, Organization, RealAllocator >& matrix,
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
      static void vectorProduct( const Tridiagonal< Real, Device, Index, Organization, RealAllocator >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         MatrixVectorProductCuda( matrix, inVector, outVector );
      }
};
 */

} // namespace Matrices
} // namespace TNL
