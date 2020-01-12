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
          bool RowMajorOrder >
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
MultidiagonalMatrixView()
{
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
MultidiagonalMatrixView( const ValuesViewType& values,
                         const DiagonalsShiftsView& diagonalsShifts,
                         const IndexerType& indexer )
: MatrixView< Real, Device, Index >( indexer.getRows(), indexer.getColumns(), values ),
  diagonalsShifts( diagonalsShifts ),
  indexer( indexer )
{
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
auto
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
getView() -> ViewType
{
   return ViewType( this->values.getView(), indexer );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
auto
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
getConstView() const -> ConstViewType
{
   return ConstViewType( this->values.getConstView(), indexer );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
String
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
getSerializationType()
{
   return String( "Matrices::Multidiagonal< " ) +
          TNL::getSerializationType< RealType >() + ", [any_device], " +
          TNL::getSerializationType< IndexType >() + ", " +
          ( RowMajorOrder ? "true" : "false" ) + ", [any_allocator] >";
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
String
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
__cuda_callable__
const Index&
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
getDiagonalsCount() const
{
   return this->diagonalsShifts.getSize();
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
   template< typename Vector >
void
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
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
          bool RowMajorOrder >
Index
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
getNonemptyRowsCount() const
{
   return this->indexer.getNonemptyRowsCount();
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
Index
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
getRowLength( const IndexType row ) const
{
   return this->diagonalsShifts.getSize();
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
Index
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
getMaxRowLength() const
{
   return this->diagonalsShifts.getSize();
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
Index
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
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
          bool RowMajorOrder >
   template< typename Real_, typename Device_, typename Index_, bool RowMajorOrder_ >
bool
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
operator == ( const MultidiagonalMatrixView< Real_, Device_, Index_, RowMajorOrder_ >& matrix ) const
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
          bool RowMajorOrder >
   template< typename Real_, typename Device_, typename Index_, bool RowMajorOrder_ >
bool
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
operator != ( const MultidiagonalMatrixView< Real_, Device_, Index_, RowMajorOrder_ >& matrix ) const
{
   return ! this->operator==( matrix );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
void
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
setValue( const RealType& v )
{
   this->values = v;
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
__cuda_callable__
auto
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
getRow( const IndexType& rowIdx ) const -> const RowView
{
   return RowView( rowIdx, this->values.getView(), this->indexer );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
__cuda_callable__
auto
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
getRow( const IndexType& rowIdx ) -> RowView
{
   return RowView( rowIdx, this->values.getView(), this->indexer );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
bool
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
setElement( const IndexType row, const IndexType column, const RealType& value )
{
   TNL_ASSERT_GE( row, 0, "" );
   TNL_ASSERT_LT( row, this->getRows(), "" );
   TNL_ASSERT_GE( column, 0, "" );
   TNL_ASSERT_LT( column, this->getColumns(), "" );
   if( abs( row - column ) > 1 )
   {
      std::stringstream msg;
      msg << "Wrong matrix element coordinates ( "  << row << ", " << column << " ) in tridiagonal matrix.";
      throw std::logic_error( msg.str() );
   }
   this->values.setElement( this->getElementIndex( row, column ), value );
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
bool
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
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
   {
      std::stringstream msg;
      msg << "Wrong matrix element coordinates ( "  << row << ", " << column << " ) in tridiagonal matrix.";
      throw std::logic_error( msg.str() );
   }
   const Index i = this->getElementIndex( row, column );
   this->values.setElement( i, thisElementMultiplicator * this->values.getElement( i ) + value );
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
Real
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
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
          bool RowMajorOrder >
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >&
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
operator=( const MultidiagonalMatrixView& view )
{
   MatrixView< Real, Device, Index >::operator=( view );
   this->diagonalsShifts.copy( view.diagonalsShifts );
   this->hostDiagonalsShifts.copy( view.hostDiagonalsShifts );
   this->indexer = view.indexer;
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
rowsReduction( IndexType first, IndexType last, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero_ ) const
{
   using Real_ = decltype( fetch( IndexType(), IndexType(), RealType() ) );
   const auto values_view = this->values.getConstView();
   const auto indexer = this->indexer;
   const auto zero = zero_;
   auto f = [=] __cuda_callable__ ( IndexType rowIdx ) mutable {
      Real_ sum( zero );
      if( rowIdx == 0 )
      {
         reduce( sum, fetch( 0, 0, values_view[ indexer.getGlobalIndex( 0, 0 ) ] ) );
         reduce( sum, fetch( 0, 1, values_view[ indexer.getGlobalIndex( 0, 1 ) ] ) );
         keep( 0, sum );
         return;
      }
      if( rowIdx + 1 < indexer.getColumns() )
      {
         reduce( sum, fetch( rowIdx, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] ) );
         reduce( sum, fetch( rowIdx, rowIdx,     values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] ) );
         reduce( sum, fetch( rowIdx, rowIdx + 1, values_view[ indexer.getGlobalIndex( rowIdx, 2 ) ] ) );
         keep( rowIdx, sum );
         return;
      }
      if( rowIdx < indexer.getColumns() )
      {
         reduce( sum, fetch( rowIdx, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] ) );
         reduce( sum, fetch( rowIdx, rowIdx,     values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] ) );
         keep( rowIdx, sum );
      }
      else
      {
         keep( rowIdx, fetch( rowIdx, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] ) );
      }
   };
   Algorithms::ParallelFor< DeviceType >::exec( first, last, f );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
allRowsReduction( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const
{
   this->rowsReduction( 0, this->indexer.getNonEmptyRowsCount(), fetch, reduce, keep, zero );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
   template< typename Function >
void
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
forRows( IndexType first, IndexType last, Function& function ) const
{
   const auto values_view = this->values.getConstView();
   const auto indexer = this->indexer;
   auto f = [=] __cuda_callable__ ( IndexType rowIdx ) mutable {
      if( rowIdx == 0 )
      {
         function( 0, 0, 0, values_view[ indexer.getGlobalIndex( 0, 0 ) ] );
         function( 0, 1, 1, values_view[ indexer.getGlobalIndex( 0, 1 ) ] );
      } 
      else if( rowIdx + 1 < indexer.getColumns() )
      {
         function( rowIdx, 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
         function( rowIdx, 1, rowIdx,     values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
         function( rowIdx, 2, rowIdx + 1, values_view[ indexer.getGlobalIndex( rowIdx, 2 ) ] );
      }
      else if( rowIdx < indexer.getColumns() )
      {
         function( rowIdx, 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
         function( rowIdx, 1, rowIdx,     values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
      }
      else
         function( rowIdx, 0, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
   };
   Algorithms::ParallelFor< DeviceType >::exec( first, last, f );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
  template< typename Function >
void
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
forRows( IndexType first, IndexType last, Function& function )
{
   auto values_view = this->values.getView();
   const auto indexer = this->indexer;
   auto f = [=] __cuda_callable__ ( IndexType rowIdx ) mutable {
      if( rowIdx == 0 )
      {
         function( 0, 0, 0, values_view[ indexer.getGlobalIndex( 0, 0 ) ] );
         function( 0, 1, 1, values_view[ indexer.getGlobalIndex( 0, 1 ) ] );
      } 
      else if( rowIdx + 1 < indexer.getColumns() )
      {
         function( rowIdx, 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
         function( rowIdx, 1, rowIdx,     values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
         function( rowIdx, 2, rowIdx + 1, values_view[ indexer.getGlobalIndex( rowIdx, 2 ) ] );
      }
      else if( rowIdx < indexer.getColumns() )
      {
         function( rowIdx, 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
         function( rowIdx, 1, rowIdx,     values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
      }
      else
         function( rowIdx, 0, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
   };
   Algorithms::ParallelFor< DeviceType >::exec( first, last, f );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
   template< typename Function >
void
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
forAllRows( Function& function ) const
{
   this->forRows( 0, this->indxer.getNonEmptyRowsCount(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
   template< typename Function >
void
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
forAllRows( Function& function )
{
   this->forRows( 0, this->indexer.getNonEmptyRowsCount(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
template< typename Vector >
__cuda_callable__
typename Vector::RealType 
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
rowVectorProduct( const IndexType row, const Vector& vector ) const
{
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
   template< typename InVector,
             typename OutVector >
void 
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
vectorProduct( const InVector& inVector, OutVector& outVector ) const
{
   TNL_ASSERT_EQ( this->getColumns(), inVector.getSize(), "Matrix columns do not fit with input vector." );
   TNL_ASSERT_EQ( this->getRows(), outVector.getSize(), "Matrix rows do not fit with output vector." );

   const auto inVectorView = inVector.getConstView();
   auto outVectorView = outVector.getView();
   const auto valuesView = this->values.getConstView();
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
          bool RowMajorOrder >
   template< typename Real_, typename Device_, typename Index_, bool RowMajorOrder_ >
void
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
addMatrix( const MultidiagonalMatrixView< Real_, Device_, Index_, RowMajorOrder_ >& matrix,
           const RealType& matrixMultiplicator,
           const RealType& thisMatrixMultiplicator )
{
   TNL_ASSERT_EQ( this->getRows(), matrix.getRows(), "Matrices rows are not equal." );
   TNL_ASSERT_EQ( this->getColumns(), matrix.getColumns(), "Matrices columns are not equal." );

   if( RowMajorOrder == RowMajorOrder_ )
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
   }
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
          bool RowMajorOrder >
   template< typename Real2, typename Index2 >
void
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
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
          bool RowMajorOrder >
   template< typename Vector1, typename Vector2 >
__cuda_callable__
void
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
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
          bool RowMajorOrder >
void MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::save( File& file ) const
{
   MatrixView< Real, Device, Index >::save( file );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
void
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
void MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::print( std::ostream& str ) const
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
          bool RowMajorOrder >
__cuda_callable__
auto
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
getIndexer() const -> const IndexerType&
{
   return this->indexer;
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
__cuda_callable__
auto
MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
getIndexer() -> IndexerType&
{
   return this->indexer;
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder >
__cuda_callable__
Index MultidiagonalMatrixView< Real, Device, Index, RowMajorOrder >::
getElementIndex( const IndexType row, const IndexType column ) const
{
   IndexType localIdx = column - row;
   if( row > 0 )
      localIdx++;

   TNL_ASSERT_GE( localIdx, 0, "" );
   TNL_ASSERT_LT( localIdx, 3, "" );

   return this->indexer.getGlobalIndex( row, localIdx );
}

} // namespace Matrices
} // namespace TNL
