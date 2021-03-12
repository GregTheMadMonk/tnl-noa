/***************************************************************************
                          DenseMatrixView.hpp  -  description
                             -------------------
    begin                : Nov 29, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iomanip>
#include <functional>
#include <TNL/Assert.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Matrices {

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
DenseMatrixView< Real, Device, Index, Organization >::
DenseMatrixView()
{
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
DenseMatrixView< Real, Device, Index, Organization >::
DenseMatrixView( const IndexType rows,
                 const IndexType columns,
                 const ValuesViewType& values )
 : MatrixView< Real, Device, Index >( rows, columns, values )
{
   SegmentsType a( rows, columns );
   segments = a.getView();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
auto
DenseMatrixView< Real, Device, Index, Organization >::
getView() -> ViewType
{
   return ViewType( this->getRows(),
                    this->getColumns(),
                    this->getValues().getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
auto
DenseMatrixView< Real, Device, Index, Organization >::
getConstView() const -> ConstViewType
{
   return ConstViewType( this->getRows(),
                         this->getColumns(),
                         this->getValues().getConstView(),
                         this->getColumnsIndexes().getConstView() );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
String
DenseMatrixView< Real, Device, Index, Organization >::
getSerializationType()
{
   return String( "Matrices::DenseMatrix< " ) +
      TNL::getSerializationType< RealType >() + ", [any_device], " +
      TNL::getSerializationType< IndexType >() + ", " +
      TNL::getSerializationType( Organization ) + " >";
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
String
DenseMatrixView< Real, Device, Index, Organization >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Vector >
void
DenseMatrixView< Real, Device, Index, Organization >::
getRowCapacities( Vector& rowCapacities ) const
{
   rowCapacities.setSize( this->getRows() );
   rowCapacities = this->getColumns();
}


template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Vector >
void
DenseMatrixView< Real, Device, Index, Organization >::
getCompressedRowLengths( Vector& rowLengths ) const
{
   rowLengths.setSize( this->getRows() );
   rowLengths = 0;
   auto rowLengths_view = rowLengths.getView();
   auto fetch = [] __cuda_callable__ ( IndexType row, IndexType column, const RealType& value ) -> IndexType {
      return ( value != 0.0 );
   };
   auto keep = [=] __cuda_callable__ ( const IndexType rowIdx, const IndexType value ) mutable {
      rowLengths_view[ rowIdx ] = value;
   };
   this->allRowsReduction( fetch, std::plus<>{}, keep, 0 );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
Index
DenseMatrixView< Real, Device, Index, Organization >::
getAllocatedElementsCount() const
{
   return this->getRows() * this->getColumns();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
Index
DenseMatrixView< Real, Device, Index, Organization >::
getNonzeroElementsCount() const
{
   const auto values_view = this->values.getConstView();
   auto fetch = [=] __cuda_callable__ ( const IndexType i ) -> IndexType {
      return ( values_view[ i ] != 0.0 );
   };
   return Algorithms::Reduction< DeviceType >::reduce( ( IndexType ) 0, this->values.getSize(), std::plus<>{}, fetch, 0 );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
void
DenseMatrixView< Real, Device, Index, Organization >::
setValue( const Real& value )
{
   this->values = value;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__ auto
DenseMatrixView< Real, Device, Index, Organization >::
getRow( const IndexType& rowIdx ) const -> const RowView
{
   TNL_ASSERT_LT( rowIdx, this->getRows(), "Row index is larger than number of matrix rows." );
   return RowView( this->segments.getSegmentView( rowIdx ), this->values.getConstView() );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__ auto
DenseMatrixView< Real, Device, Index, Organization >::
getRow( const IndexType& rowIdx ) -> RowView
{
   TNL_ASSERT_LT( rowIdx, this->getRows(), "Row index is larger than number of matrix rows." );
   return RowView( this->segments.getSegmentView( rowIdx ), this->values.getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
Real& DenseMatrixView< Real, Device, Index, Organization >::operator()( const IndexType row,
                                                const IndexType column )
{
   TNL_ASSERT_GE( row, 0, "Row index must be non-negative." );
   TNL_ASSERT_LT( row, this->getRows(), "Row index is out of bounds." );
   TNL_ASSERT_GE( column, 0, "Column index must be non-negative." );
   TNL_ASSERT_LT( column, this->getColumns(), "Column index is out of bounds." );

   return this->values.operator[]( this->getElementIndex( row, column ) );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
const Real& DenseMatrixView< Real, Device, Index, Organization >::operator()( const IndexType row,
                                                      const IndexType column ) const
{
   TNL_ASSERT_GE( row, 0, "Row index must be non-negative." );
   TNL_ASSERT_LT( row, this->getRows(), "Row index is out of bounds." );
   TNL_ASSERT_GE( column, 0, "Column index must be non-negative." );
   TNL_ASSERT_LT( column, this->getColumns(), "Column index is out of bounds." );

   return this->values.operator[]( this->getElementIndex( row, column ) );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__ void
DenseMatrixView< Real, Device, Index, Organization >::
setElement( const IndexType row,
            const IndexType column,
            const RealType& value )
{
   this->values.setElement( this->getElementIndex( row, column ), value );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__ void
DenseMatrixView< Real, Device, Index, Organization >::
addElement( const IndexType row,
            const IndexType column,
            const RealType& value,
            const RealType& thisElementMultiplicator )
{
   const IndexType elementIndex = this->getElementIndex( row, column );
   if( thisElementMultiplicator == 1.0 )
      this->values.setElement( elementIndex,
                               this->values.getElement( elementIndex ) + value );
   else
      this->values.setElement( elementIndex,
                               thisElementMultiplicator * this->values.getElement( elementIndex ) + value );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__ Real
DenseMatrixView< Real, Device, Index, Organization >::
getElement( const IndexType row,
            const IndexType column ) const
{
   return this->values.getElement( this->getElementIndex( row, column ) );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
void
DenseMatrixView< Real, Device, Index, Organization >::
rowsReduction( IndexType begin, IndexType end, Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchValue& zero )
{
   auto values_view = this->values.getView();
   auto fetch_ = [=] __cuda_callable__ ( IndexType rowIdx, IndexType columnIdx, IndexType globalIdx, bool& compute ) mutable -> decltype( fetch( IndexType(), IndexType(), RealType() ) ) {
         return fetch( rowIdx, columnIdx, values_view[ globalIdx ] );
      return zero;
   };
   this->segments.segmentsReduction( begin, end, fetch_, reduce, keep, zero );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
void
DenseMatrixView< Real, Device, Index, Organization >::
rowsReduction( IndexType begin, IndexType end, Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchValue& zero ) const
{
   const auto values_view = this->values.getConstView();
   auto fetch_ = [=] __cuda_callable__ ( IndexType rowIdx, IndexType columnIdx, IndexType globalIdx, bool& compute ) mutable -> decltype( fetch( IndexType(), IndexType(), RealType() ) ) {
         return fetch( rowIdx, columnIdx, values_view[ globalIdx ] );
      return zero;
   };
   this->segments.segmentsReduction( begin, end, fetch_, reduce, keep, zero );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
DenseMatrixView< Real, Device, Index, Organization >::
allRowsReduction( Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& zero )
{
   this->rowsReduction( 0, this->getRows(), fetch, reduce, keep, zero );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
DenseMatrixView< Real, Device, Index, Organization >::
allRowsReduction( Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& zero ) const
{
   this->rowsReduction( 0, this->getRows(), fetch, reduce, keep, zero );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::
forElements( IndexType begin, IndexType end, Function& function ) const
{
   const auto values_view = this->values.getConstView();
   auto f = [=] __cuda_callable__ ( IndexType rowIdx, IndexType columnIdx, IndexType globalIdx, bool& compute ) mutable {
      function( rowIdx, columnIdx, columnIdx, values_view[ globalIdx ], compute );
   };
   this->segments.forElements( begin, end, f );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::
forElements( IndexType begin, IndexType end, Function& function )
{
   auto values_view = this->values.getView();
   auto f = [=] __cuda_callable__ ( IndexType rowIdx, IndexType columnIdx, IndexType globalIdx, bool& compute ) mutable {
      function( rowIdx, columnIdx, globalIdx, values_view[ globalIdx ], compute );
   };
   this->segments.forElements( begin, end, f );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::
forEachElement( Function& function ) const
{
   this->forElements( 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::
forEachElement( Function& function )
{
   this->forElements( 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::
sequentialForRows( IndexType begin, IndexType end, Function& function ) const
{
   for( IndexType row = begin; row < end; row ++ )
      this->forElements( row, row + 1, function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::
sequentialForRows( IndexType begin, IndexType end, Function& function )
{
   for( IndexType row = begin; row < end; row ++ )
      this->forElements( row, row + 1, function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::
sequentialForAllRows( Function& function ) const
{
   this->sequentialForRows( 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::
sequentialForAllRows( Function& function )
{
   this->sequentialForRows( 0, this->getRows(), function );
}


template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename InVector,
             typename OutVector >
void
DenseMatrixView< Real, Device, Index, Organization >::
vectorProduct( const InVector& inVector,
               OutVector& outVector,
               const RealType& matrixMultiplicator,
               const RealType& outVectorMultiplicator,
               const IndexType begin,
               IndexType end ) const
{
   TNL_ASSERT_EQ( this->getColumns(), inVector.getSize(), "Matrix columns count differs with input vector size." );
   TNL_ASSERT_EQ( this->getRows(), outVector.getSize(), "Matrix rows count differs with output vector size." );

   const auto inVectorView = inVector.getConstView();
   auto outVectorView = outVector.getView();
   const auto valuesView = this->values.getConstView();
   if( end == 0 )
      end = this->getRows();
   auto fetch = [=] __cuda_callable__ ( IndexType row, IndexType column, IndexType offset, bool& compute ) -> RealType {
      return valuesView[ offset ] * inVectorView[ column ];
   };
   auto keeper = [=] __cuda_callable__ ( IndexType row, const RealType& value ) mutable {
      outVectorView[ row ] = matrixMultiplicator * value + outVectorMultiplicator * outVectorView[ row ];
   };
   this->segments.segmentsReduction( begin, end, fetch, std::plus<>{}, keeper, ( RealType ) 0.0 );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Matrix >
void
DenseMatrixView< Real, Device, Index, Organization >::
addMatrix( const Matrix& matrix,
           const RealType& matrixMultiplicator,
           const RealType& thisMatrixMultiplicator )
{
   TNL_ASSERT( this->getColumns() == matrix.getColumns() &&
              this->getRows() == matrix.getRows(),
            std::cerr << "This matrix columns: " << this->getColumns() << std::endl
                 << "This matrix rows: " << this->getRows() << std::endl
                 << "That matrix columns: " << matrix.getColumns() << std::endl
                 << "That matrix rows: " << matrix.getRows() << std::endl );

   if( thisMatrixMultiplicator == 1.0 )
      this->values += matrixMultiplicator * matrix.values;
   else
      this->values = thisMatrixMultiplicator * this->values + matrixMultiplicator * matrix.values;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Matrix1, typename Matrix2, int tileDim >
void DenseMatrixView< Real, Device, Index, Organization >::getMatrixProduct( const Matrix1& matrix1,
                                                              const Matrix2& matrix2,
                                                              const RealType& matrix1Multiplicator,
                                                              const RealType& matrix2Multiplicator )
{
   TNL_ASSERT( matrix1.getColumns() == matrix2.getRows() &&
              this->getRows() == matrix1.getRows() &&
              this->getColumns() == matrix2.getColumns(),
            std::cerr << "This matrix columns: " << this->getColumns() << std::endl
                 << "This matrix rows: " << this->getRows() << std::endl
                 << "Matrix1 columns: " << matrix1.getColumns() << std::endl
                 << "Matrix1 rows: " << matrix1.getRows() << std::endl
                 << "Matrix2 columns: " << matrix2.getColumns() << std::endl
                 << "Matrix2 rows: " << matrix2.getRows() << std::endl );

   if( std::is_same< Device, Devices::Host >::value )
      for( IndexType i = 0; i < this->getRows(); i += tileDim )
         for( IndexType j = 0; j < this->getColumns(); j += tileDim )
         {
            const IndexType tileRows = min( tileDim, this->getRows() - i );
            const IndexType tileColumns = min( tileDim, this->getColumns() - j );
            for( IndexType i1 = i; i1 < i + tileRows; i1++ )
               for( IndexType j1 = j; j1 < j + tileColumns; j1++ )
                  this->setElementFast( i1, j1, 0.0 );

            for( IndexType k = 0; k < matrix1.getColumns(); k += tileDim )
            {
               const IndexType lastK = min( k + tileDim, matrix1.getColumns() );
               for( IndexType i1 = 0; i1 < tileRows; i1++ )
                  for( IndexType j1 = 0; j1 < tileColumns; j1++ )
                     for( IndexType k1 = k; k1 < lastK; k1++ )
                        this->addElementFast( i + i1, j + j1,
                            matrix1.getElementFast( i + i1, k1 ) * matrix2.getElementFast( k1, j + j1 ) );
            }
         }
   if( std::is_same< Device, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      /*dim3 cudaBlockSize( 0 ), cudaGridSize( 0 );
      const IndexType matrixProductCudaBlockSize( 256 );
      const IndexType rowTiles = roundUpDivision( this->getRows(), tileDim );
      const IndexType columnTiles = roundUpDivision( this->getColumns(), tileDim );
      const IndexType cudaBlockColumns( tileDim );
      const IndexType cudaBlockRows( matrixProductCudaBlockSize / tileDim );
      cudaBlockSize.x = cudaBlockColumns;
      cudaBlockSize.y = cudaBlockRows;
      const IndexType rowGrids = roundUpDivision( rowTiles, Cuda::getMaxGridSize() );
      const IndexType columnGrids = roundUpDivision( columnTiles, Cuda::getMaxGridSize() );

      for( IndexType gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ )
         for( IndexType gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ )
         {
            cudaGridSize.x = cudaGridSize.y = Cuda::getMaxGridSize();
            if( gridIdx_x == columnGrids - 1 )
               cudaGridSize.x = columnTiles % Cuda::getMaxGridSize();
            if( gridIdx_y == rowGrids - 1 )
               cudaGridSize.y = rowTiles % Cuda::getMaxGridSize();
            Dense* this_kernel = Cuda::passToDevice( *this );
            Matrix1* matrix1_kernel = Cuda::passToDevice( matrix1 );
            Matrix2* matrix2_kernel = Cuda::passToDevice( matrix2 );
            DenseMatrixProductKernel< Real,
                                               Index,
                                               Matrix1,
                                               Matrix2,
                                               tileDim,
                                               cudaBlockRows >
                                           <<< cudaGridSize,
                                               cudaBlockSize,
                                               3*tileDim*tileDim >>>
                                             ( this_kernel,
                                               matrix1_kernel,
                                               matrix2_kernel,
                                               matrix1Multiplicator,
                                               matrix2Multiplicator,
                                               gridIdx_x,
                                               gridIdx_y );
            Cuda::freeFromDevice( this_kernel );
            Cuda::freeFromDevice( matrix1_kernel );
            Cuda::freeFromDevice( matrix2_kernel );
         }*/
#endif
   }
}


template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Matrix, int tileDim >
void DenseMatrixView< Real, Device, Index, Organization >::getTransposition( const Matrix& matrix,
                                                              const RealType& matrixMultiplicator )
{
   TNL_ASSERT( this->getColumns() == matrix.getRows() &&
              this->getRows() == matrix.getColumns(),
               std::cerr << "This matrix columns: " << this->getColumns() << std::endl
                    << "This matrix rows: " << this->getRows() << std::endl
                    << "That matrix columns: " << matrix.getColumns() << std::endl
                    << "That matrix rows: " << matrix.getRows() << std::endl );

   if( std::is_same< Device, Devices::Host >::value )
   {
      const IndexType& rows = matrix.getRows();
      const IndexType& columns = matrix.getColumns();
      for( IndexType i = 0; i < rows; i += tileDim )
         for( IndexType j = 0; j < columns; j += tileDim )
            for( IndexType k = i; k < i + tileDim && k < rows; k++ )
               for( IndexType l = j; l < j + tileDim && l < columns; l++ )
                  this->setElement( l, k, matrixMultiplicator * matrix. getElement( k, l ) );
   }
   if( std::is_same< Device, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      /*dim3 cudaBlockSize( 0 ), cudaGridSize( 0 );
      const IndexType matrixProductCudaBlockSize( 256 );
      const IndexType rowTiles = roundUpDivision( this->getRows(), tileDim );
      const IndexType columnTiles = roundUpDivision( this->getColumns(), tileDim );
      const IndexType cudaBlockColumns( tileDim );
      const IndexType cudaBlockRows( matrixProductCudaBlockSize / tileDim );
      cudaBlockSize.x = cudaBlockColumns;
      cudaBlockSize.y = cudaBlockRows;
      const IndexType rowGrids = roundUpDivision( rowTiles, Cuda::getMaxGridSize() );
      const IndexType columnGrids = roundUpDivision( columnTiles, Cuda::getMaxGridSize() );
      const IndexType sharedMemorySize = tileDim*tileDim + tileDim*tileDim/Cuda::getNumberOfSharedMemoryBanks();

      Dense* this_device = Cuda::passToDevice( *this );
      Matrix* matrix_device = Cuda::passToDevice( matrix );

      for( IndexType gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ )
         for( IndexType gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ )
         {
            cudaGridSize.x = cudaGridSize.y = Cuda::getMaxGridSize();
            if( gridIdx_x == columnGrids - 1)
               cudaGridSize.x = columnTiles % Cuda::getMaxGridSize();
            if( gridIdx_y == rowGrids - 1 )
               cudaGridSize.y = rowTiles % Cuda::getMaxGridSize();
            if( ( gridIdx_x < columnGrids - 1 || matrix.getColumns() % tileDim == 0 ) &&
                ( gridIdx_y < rowGrids - 1 || matrix.getRows() % tileDim == 0 ) )
            {
               DenseTranspositionAlignedKernel< Real,
                                                         Index,
                                                         Matrix,
                                                         tileDim,
                                                         cudaBlockRows >
                                                     <<< cudaGridSize,
                                                         cudaBlockSize,
                                                         sharedMemorySize  >>>
                                                       ( this_device,
                                                         matrix_device,
                                                         matrixMultiplicator,
                                                         gridIdx_x,
                                                         gridIdx_y );
            }
            else
            {
               DenseTranspositionNonAlignedKernel< Real,
                                                         Index,
                                                         Matrix,
                                                         tileDim,
                                                         cudaBlockRows >
                                                     <<< cudaGridSize,
                                                         cudaBlockSize,
                                                         sharedMemorySize  >>>
                                                       ( this_device,
                                                         matrix_device,
                                                         matrixMultiplicator,
                                                         gridIdx_x,
                                                         gridIdx_y );
            }
            TNL_CHECK_CUDA_DEVICE;
         }
      Cuda::freeFromDevice( this_device );
      Cuda::freeFromDevice( matrix_device );*/
#endif
   }
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Vector1, typename Vector2 >
void DenseMatrixView< Real, Device, Index, Organization >::performSORIteration( const Vector1& b,
                                                        const IndexType row,
                                                        Vector2& x,
                                                        const RealType& omega ) const
{
   RealType sum( 0.0 ), diagonalValue;
   for( IndexType i = 0; i < this->getColumns(); i++ )
   {
      if( i == row )
         diagonalValue = this->getElement( row, row );
      else
         sum += this->getElement( row, i ) * x[ i ];
   }
   x[ row ] = ( 1.0 - omega ) * x[ row ] + omega / diagonalValue * ( b[ row ] - sum );
}


template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
DenseMatrixView< Real, Device, Index, Organization >&
DenseMatrixView< Real, Device, Index, Organization >::
operator=( const DenseMatrixView& matrix )
{
   MatrixView< Real, Device, Index >::operator=( matrix );
   this->segments = matrix.segments;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
void DenseMatrixView< Real, Device, Index, Organization >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
void DenseMatrixView< Real, Device, Index, Organization >::save( File& file ) const
{
   MatrixView< Real, Device, Index >::save( file );
   this->segments.save( file );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
void DenseMatrixView< Real, Device, Index, Organization >::print( std::ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      for( IndexType column = 0; column < this->getColumns(); column++ )
      {
         std::stringstream str_;
         str_ << std::setw( 4 ) << std::right << column << ":" << std::setw( 4 ) << std::left << this->getElement( row, column );
         str << std::setw( 10 ) << str_.str();
      }
      if( row < this->getRows() - 1 )
         str << std::endl;
   }
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
Index DenseMatrixView< Real, Device, Index, Organization >::getElementIndex( const IndexType row,
                                                              const IndexType column ) const
{
   return this->segments.getGlobalIndex( row, column );
}

} // namespace Matrices
} // namespace TNL
