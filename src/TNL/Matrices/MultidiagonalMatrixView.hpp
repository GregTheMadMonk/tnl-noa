/***************************************************************************
                          MultidiagonalMatrixView.hpp  -  description
                             -------------------
    begin                : Jan 11, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Assert.h>
#include <TNL/Matrices/MultidiagonalMatrixView.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Matrices {

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
MultidiagonalMatrixView< Real, Device, Index, Organization >::
MultidiagonalMatrixView()
{
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
MultidiagonalMatrixView< Real, Device, Index, Organization >::
MultidiagonalMatrixView( const ValuesViewType& values,
                         const DiagonalsOffsetsView& diagonalsOffsets,
                         const HostDiagonalsOffsetsView& hostDiagonalsOffsets,
                         const IndexerType& indexer )
: MatrixView< Real, Device, Index >( indexer.getRows(), indexer.getColumns(), values ),
  diagonalsOffsets( diagonalsOffsets ),
  hostDiagonalsOffsets( hostDiagonalsOffsets ),
  indexer( indexer )
{
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
auto
MultidiagonalMatrixView< Real, Device, Index, Organization >::
getView() -> ViewType
{
   return ViewType( const_cast< MultidiagonalMatrixView* >( this )->values.getView(),
                    const_cast< MultidiagonalMatrixView* >( this )->diagonalsOffsets.getView(),
                    const_cast< MultidiagonalMatrixView* >( this )->hostDiagonalsOffsets.getView(),
                    indexer );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
auto
MultidiagonalMatrixView< Real, Device, Index, Organization >::
getConstView() const -> ConstViewType
{
   return ConstViewType( this->values.getConstView(),
                         this->diagonalsOffsets.getConstView(),
                         this->hostDiagonalsOffsets.getConstView(),
                         indexer );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
String
MultidiagonalMatrixView< Real, Device, Index, Organization >::
getSerializationType()
{
   return String( "Matrices::Multidiagonal< " ) +
          TNL::getSerializationType< RealType >() + ", [any_device], " +
          TNL::getSerializationType< IndexType >() + ", " +
          TNL::getSerializationType( Organization ) + ", [any_allocator], [any_allocator] >";
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
String
MultidiagonalMatrixView< Real, Device, Index, Organization >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
const Index&
MultidiagonalMatrixView< Real, Device, Index, Organization >::
getDiagonalsCount() const
{
   return this->diagonalsOffsets.getSize();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Vector >
void
MultidiagonalMatrixView< Real, Device, Index, Organization >::
getCompressedRowLengths( Vector& rowLengths ) const
{
   rowLengths.setSize( this->getRows() );
   rowLengths = 0;
   auto rowLengths_view = rowLengths.getView();
   auto fetch = [] __cuda_callable__ ( IndexType row, IndexType column, const RealType& value ) -> IndexType {
      return ( value != 0.0 );
   };
   auto reduce = [] __cuda_callable__ ( IndexType& aux, const IndexType a ) {
      aux += a;
   };
   auto keep = [=] __cuda_callable__ ( const IndexType rowIdx, const IndexType value ) mutable {
      rowLengths_view[ rowIdx ] = value;
   };
   this->allRowsReduction( fetch, reduce, keep, 0 );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
Index
MultidiagonalMatrixView< Real, Device, Index, Organization >::
getNonemptyRowsCount() const
{
   return this->indexer.getNonemptyRowsCount();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
Index
MultidiagonalMatrixView< Real, Device, Index, Organization >::
getRowLength( const IndexType row ) const
{
   return this->diagonalsOffsets.getSize();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
Index
MultidiagonalMatrixView< Real, Device, Index, Organization >::
getMaxRowLength() const
{
   return this->diagonalsOffsets.getSize();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
Index
MultidiagonalMatrixView< Real, Device, Index, Organization >::
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
          ElementsOrganization Organization >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_ >
bool
MultidiagonalMatrixView< Real, Device, Index, Organization >::
operator == ( const MultidiagonalMatrixView< Real_, Device_, Index_, Organization_ >& matrix ) const
{
   if( Organization == Organization_ )
      return this->values == matrix.values;
   else
   {
      TNL_ASSERT_TRUE( false, "TODO" );
   }
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_ >
bool
MultidiagonalMatrixView< Real, Device, Index, Organization >::
operator != ( const MultidiagonalMatrixView< Real_, Device_, Index_, Organization_ >& matrix ) const
{
   return ! this->operator==( matrix );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
void
MultidiagonalMatrixView< Real, Device, Index, Organization >::
setValue( const RealType& v )
{
   // we dont do this->values = v here because it would set even elements 'outside' the matrix
   // method getNumberOfNonzeroElements would not well
   const RealType newValue = v;
   auto f = [=] __cuda_callable__ ( const IndexType& rowIdx, const IndexType& localIdx, const IndexType columnIdx, RealType& value, bool& compute ) mutable {
      value = newValue;
   };
   this->forAllRows( f );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
auto
MultidiagonalMatrixView< Real, Device, Index, Organization >::
getRow( const IndexType& rowIdx ) const -> const RowView
{
   return RowView( rowIdx, this->diagonalsOffsets.getView(), this->values.getView(), this->indexer );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
auto
MultidiagonalMatrixView< Real, Device, Index, Organization >::
getRow( const IndexType& rowIdx ) -> RowView
{
   return RowView( rowIdx, this->diagonalsOffsets.getView(), this->values.getView(), this->indexer );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
void
MultidiagonalMatrixView< Real, Device, Index, Organization >::
setElement( const IndexType row, const IndexType column, const RealType& value )
{
   TNL_ASSERT_GE( row, 0, "" );
   TNL_ASSERT_LT( row, this->getRows(), "" );
   TNL_ASSERT_GE( column, 0, "" );
   TNL_ASSERT_LT( column, this->getColumns(), "" );

   for( IndexType i = 0; i < hostDiagonalsOffsets.getSize(); i++ )
      if( row + hostDiagonalsOffsets[ i ] == column )
      {
         this->values.setElement( this->getElementIndex( row, i ), value );
         return;
      }
   if( value != 0.0 )
   {
      std::stringstream msg;
      msg << "Wrong matrix element coordinates ( "  << row << ", " << column << " ) in multidiagonal matrix.";
      throw std::logic_error( msg.str() );
   }
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
void
MultidiagonalMatrixView< Real, Device, Index, Organization >::
addElement( const IndexType row,
            const IndexType column,
            const RealType& value,
            const RealType& thisElementMultiplicator )
{
   TNL_ASSERT_GE( row, 0, "" );
   TNL_ASSERT_LT( row, this->getRows(), "" );
   TNL_ASSERT_GE( column, 0, "" );
   TNL_ASSERT_LT( column, this->getColumns(), "" );

   for( IndexType i = 0; i < hostDiagonalsOffsets.getSize(); i++ )
      if( row + hostDiagonalsOffsets[ i ] == column )
      {
         const Index idx = this->getElementIndex( row, i );
         this->values.setElement( idx, thisElementMultiplicator * this->values.getElement( idx ) + value );
         return;
      }
   if( value != 0.0 )
   {
      std::stringstream msg;
      msg << "Wrong matrix element coordinates ( "  << row << ", " << column << " ) in multidiagonal matrix.";
      throw std::logic_error( msg.str() );
   }
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
Real
MultidiagonalMatrixView< Real, Device, Index, Organization >::
getElement( const IndexType row, const IndexType column ) const
{
   TNL_ASSERT_GE( row, 0, "" );
   TNL_ASSERT_LT( row, this->getRows(), "" );
   TNL_ASSERT_GE( column, 0, "" );
   TNL_ASSERT_LT( column, this->getColumns(), "" );

   for( IndexType localIdx = 0; localIdx < hostDiagonalsOffsets.getSize(); localIdx++ )
      if( row + hostDiagonalsOffsets[ localIdx ] == column )
         return this->values.getElement( this->indexer.getGlobalIndex( row, localIdx ) );
   return 0.0;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
MultidiagonalMatrixView< Real, Device, Index, Organization >&
MultidiagonalMatrixView< Real, Device, Index, Organization >::
operator=( const MultidiagonalMatrixView& view )
{
   MatrixView< Real, Device, Index >::operator=( view );
   this->diagonalsOffsets.bind( view.diagonalsOffsets );
   this->hostDiagonalsOffsets.bind( view.hostDiagonalsOffsets );
   this->indexer = view.indexer;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
MultidiagonalMatrixView< Real, Device, Index, Organization >::
rowsReduction( IndexType first, IndexType last, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero_ ) const
{
   using Real_ = decltype( fetch( IndexType(), IndexType(), RealType() ) );
   const auto values_view = this->values.getConstView();
   const auto diagonalsOffsets_view = this->diagonalsOffsets.getConstView();
   const IndexType diagonalsCount = this->diagonalsOffsets.getSize();
   const IndexType columns = this->getColumns();
   const auto indexer = this->indexer;
   const auto zero = zero_;
   auto f = [=] __cuda_callable__ ( IndexType rowIdx ) mutable {
      Real_ sum( zero );
      for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ )
      {
         const IndexType columnIdx = rowIdx + diagonalsOffsets_view[ localIdx ];
         if( columnIdx >= 0 && columnIdx < columns )
            reduce( sum, fetch( rowIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] ) );
      }
      keep( rowIdx, sum );
   };
   Algorithms::ParallelFor< DeviceType >::exec( first, last, f );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
MultidiagonalMatrixView< Real, Device, Index, Organization >::
allRowsReduction( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const
{
   this->rowsReduction( 0, this->indexer.getNonemptyRowsCount(), fetch, reduce, keep, zero );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Function >
void
MultidiagonalMatrixView< Real, Device, Index, Organization >::
forRows( IndexType first, IndexType last, Function& function ) const
{
   const auto values_view = this->values.getConstView();
   const auto diagonalsOffsets_view = this->diagonalsOffsets.getConstView();
   const IndexType diagonalsCount = this->diagonalsOffsets.getSize();
   const IndexType columns = this->getColumns();
   const auto indexer = this->indexer;
   bool compute( true );
   auto f = [=] __cuda_callable__ ( IndexType rowIdx ) mutable {
      for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ )
      {
         const IndexType columnIdx = rowIdx + diagonalsOffsets_view[ localIdx ];
         if( columnIdx >= 0 && columnIdx < columns )
            function( rowIdx, localIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ], compute );
      }
   };
   Algorithms::ParallelFor< DeviceType >::exec( first, last, f );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
  template< typename Function >
void
MultidiagonalMatrixView< Real, Device, Index, Organization >::
forRows( IndexType first, IndexType last, Function& function )
{
   auto values_view = this->values.getView();
   const auto diagonalsOffsets_view = this->diagonalsOffsets.getConstView();
   const IndexType diagonalsCount = this->diagonalsOffsets.getSize();
   const IndexType columns = this->getColumns();
   const auto indexer = this->indexer;
   bool compute( true );
   auto f = [=] __cuda_callable__ ( IndexType rowIdx ) mutable {
      for( IndexType localIdx = 0; localIdx < diagonalsCount && compute; localIdx++ )
      {
         const IndexType columnIdx = rowIdx + diagonalsOffsets_view[ localIdx ];
         if( columnIdx >= 0 && columnIdx < columns )
            function( rowIdx, localIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ], compute );
      }
   };
   Algorithms::ParallelFor< DeviceType >::exec( first, last, f );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Function >
void
MultidiagonalMatrixView< Real, Device, Index, Organization >::
forAllRows( Function& function ) const
{
   this->forRows( 0, this->indxer.getNonEmptyRowsCount(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Function >
void
MultidiagonalMatrixView< Real, Device, Index, Organization >::
forAllRows( Function& function )
{
   this->forRows( 0, this->indexer.getNonemptyRowsCount(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
template< typename Vector >
__cuda_callable__
typename Vector::RealType 
MultidiagonalMatrixView< Real, Device, Index, Organization >::
rowVectorProduct( const IndexType row, const Vector& vector ) const
{
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename InVector,
             typename OutVector >
void 
MultidiagonalMatrixView< Real, Device, Index, Organization >::
vectorProduct( const InVector& inVector, OutVector& outVector ) const
{
   TNL_ASSERT_EQ( this->getColumns(), inVector.getSize(), "Matrix columns do not fit with input vector." );
   TNL_ASSERT_EQ( this->getRows(), outVector.getSize(), "Matrix rows do not fit with output vector." );

   const auto inVectorView = inVector.getConstView();
   auto outVectorView = outVector.getView();
   auto fetch = [=] __cuda_callable__ ( const IndexType& row, const IndexType& column, const RealType& value ) -> RealType {
      return value * inVectorView[ column ];
   };
   auto reduction = [] __cuda_callable__ ( RealType& sum, const RealType& value ) {
      sum += value;
   };
   auto keeper = [=] __cuda_callable__ ( IndexType row, const RealType& value ) mutable {
      outVectorView[ row ] = value;
   };
   this->allRowsReduction( fetch, reduction, keeper, ( RealType ) 0.0 );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_ >
void
MultidiagonalMatrixView< Real, Device, Index, Organization >::
addMatrix( const MultidiagonalMatrixView< Real_, Device_, Index_, Organization_ >& matrix,
           const RealType& matrixMultiplicator,
           const RealType& thisMatrixMultiplicator )
{
   TNL_ASSERT_EQ( this->getRows(), matrix.getRows(), "Matrices rows are not equal." );
   TNL_ASSERT_EQ( this->getColumns(), matrix.getColumns(), "Matrices columns are not equal." );

   /*if( Organization == Organization_ )
   {
      if( thisMatrixMultiplicator == 1.0 )
         this->values += matrixMultiplicator * matrix.getValues();
      else
         this->values = thisMatrixMultiplicator * this->values + matrixMultiplicator * matrix.getValues();
   }
   else
   {
      const auto matrix_view = matrix;
      const auto matrixMult = matrixMultiplicator;
      const auto thisMult = thisMatrixMultiplicator;
      auto add0 = [=] __cuda_callable__ ( const IndexType& rowIdx, const IndexType& localIdx, const IndexType& column, Real& value ) mutable {
         value = matrixMult * matrix.getValues()[ matrix.getIndexer().getGlobalIndex( rowIdx, localIdx ) ];
      };
      auto add1 = [=] __cuda_callable__ ( const IndexType& rowIdx, const IndexType& localIdx, const IndexType& column, Real& value ) mutable {
         value += matrixMult * matrix.getValues()[ matrix.getIndexer().getGlobalIndex( rowIdx, localIdx ) ];
      };
      auto addGen = [=] __cuda_callable__ ( const IndexType& rowIdx, const IndexType& localIdx, const IndexType& column, Real& value ) mutable {
         value = thisMult * value + matrixMult * matrix.getValues()[ matrix.getIndexer().getGlobalIndex( rowIdx, localIdx ) ];
      };
      if( thisMult == 0.0 )
         this->forAllRows( add0 );
      else if( thisMult == 1.0 )
         this->forAllRows( add1 );
      else
         this->forAllRows( addGen );
   }*/
}

#ifdef HAVE_CUDA
/*template< typename Real,
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
}*/
#endif

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Real2, typename Index2 >
void
MultidiagonalMatrixView< Real, Device, Index, Organization >::
getTransposition( const MultidiagonalMatrixView< Real2, Device, Index2 >& matrix,
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
      /*Multidiagonal* kernel_this = Cuda::passToDevice( *this );
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
      TNL_CHECK_CUDA_DEVICE;*/
#endif
   }
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Vector1, typename Vector2 >
__cuda_callable__
void
MultidiagonalMatrixView< Real, Device, Index, Organization >::
performSORIteration( const Vector1& b,
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


template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
void MultidiagonalMatrixView< Real, Device, Index, Organization >::save( File& file ) const
{
   MatrixView< Real, Device, Index >::save( file );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
void
MultidiagonalMatrixView< Real, Device, Index, Organization >::
save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
void MultidiagonalMatrixView< Real, Device, Index, Organization >::print( std::ostream& str ) const
{
   for( IndexType rowIdx = 0; rowIdx < this->getRows(); rowIdx++ )
   {
      str <<"Row: " << rowIdx << " -> ";
      for( IndexType localIdx = 0; localIdx < this->hostDiagonalsOffsets.getSize(); localIdx++ )
      {
         const IndexType columnIdx = rowIdx + this->hostDiagonalsOffsets[ localIdx ];
         if( columnIdx >= 0 && columnIdx < this->columns )
         {
            auto v = this->values.getElement( this->indexer.getGlobalIndex( rowIdx, localIdx ) );
            if( v )
               str << " Col:" << columnIdx << "->" << v  << "\t";
         }
      }
      str << std::endl;
   }
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
auto
MultidiagonalMatrixView< Real, Device, Index, Organization >::
getIndexer() const -> const IndexerType&
{
   return this->indexer;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
auto
MultidiagonalMatrixView< Real, Device, Index, Organization >::
getIndexer() -> IndexerType&
{
   return this->indexer;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
Index
MultidiagonalMatrixView< Real, Device, Index, Organization >::
getElementIndex( const IndexType row, const IndexType localIdx ) const
{
   return this->indexer.getGlobalIndex( row, localIdx );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
Index
MultidiagonalMatrixView< Real, Device, Index, Organization >::
getPaddingIndex() const
{
   return -1;
}


} // namespace Matrices
} // namespace TNL
