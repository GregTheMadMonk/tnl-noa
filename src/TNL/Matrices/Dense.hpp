/***************************************************************************
                          Dense_impl.h  -  description
                             -------------------
    begin                : Nov 29, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Assert.h>
#include <TNL/Matrices/Dense.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Matrices {   

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::Dense()
{
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
String Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::getSerializationType()
{
   return String( "Matrices::Dense< " ) +
          getType< RealType >() + ", " +
          getType< Device >() + ", " +
          getType< IndexType >() + " >";
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
String Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
void Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::setDimensions( const IndexType rows,
                                                  const IndexType columns )
{
   Matrix< Real, Device, Index >::setDimensions( rows, columns );
   this->values.setSize( rows * columns );
   this->values.setValue( 0.0 );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
   template< typename Matrix_ >
void Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::setLike( const Matrix_& matrix )
{
   Matrix< Real, Device, Index, RealAllocator >::setLike( matrix );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
void Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::setCompressedRowLengths( ConstCompressedRowLengthsVectorView rowLengths )
{
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
Index Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::getRowLength( const IndexType row ) const
{
   return this->getColumns();
}

/*template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
__cuda_callable__
Index Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::getRowLengthFast( const IndexType row ) const
{
   return this->getColumns();
}*/

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
Index Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::getMaxRowLength() const
{
   return this->getColumns();
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
Index Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::getNumberOfMatrixElements() const
{
   return this->getRows() * this->getColumns();
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
Index Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::getNumberOfNonzeroMatrixElements() const
{
   IndexType nonzeroElements( 0 );
   for( IndexType row = 0; row < this->getRows(); row++ )
      for( IndexType column = 0; column < this->getColumns(); column++ )
         if( this->getElement( row, column ) != 0 )
            nonzeroElements++;
   return nonzeroElements;
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
void Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::reset()
{
   Matrix< Real, Device, Index >::reset();
   this->values.reset();
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
void Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::setValue( const Real& value )
{
   this->values.setValue( value );
}


template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
__cuda_callable__
Real& Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::operator()( const IndexType row,
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
          bool RowMajorOrder,
          typename RealAllocator >
__cuda_callable__
const Real& Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::operator()( const IndexType row,
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
          bool RowMajorOrder,
          typename RealAllocator >
__cuda_callable__
bool Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::setElementFast( const IndexType row,
                                                            const IndexType column,
                                                            const RealType& value )
{
   TNL_ASSERT_GE( row, 0, "Row index must be non-negative." );
   TNL_ASSERT_LT( row, this->getRows(), "Row index is out of bounds." );
   TNL_ASSERT_GE( column, 0, "Column index must be non-negative." );
   TNL_ASSERT_LT( column, this->getColumns(), "Column index is out of bounds." );

   this->values.operator[]( this->getElementIndex( row, column ) ) = value;
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
bool Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::setElement( const IndexType row,
                                               const IndexType column,
                                               const RealType& value )
{
   this->values.setElement( this->getElementIndex( row, column ), value );
   return true;
}


template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
__cuda_callable__
bool Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::addElementFast( const IndexType row,
                                                   const IndexType column,
                                                   const RealType& value,
                                                   const RealType& thisElementMultiplicator )
{
   TNL_ASSERT_GE( row, 0, "Row index must be non-negative." );
   TNL_ASSERT_LT( row, this->getRows(), "Row index is out of bounds." );
   TNL_ASSERT_GE( column, 0, "Column index must be non-negative." );
   TNL_ASSERT_LT( column, this->getColumns(), "Column index is out of bounds." );

   const IndexType elementIndex = this->getElementIndex( row, column );
   if( thisElementMultiplicator == 1.0 )
      this->values.operator[]( elementIndex ) += value;
   else
      this->values.operator[]( elementIndex ) =
         thisElementMultiplicator * this->values.operator[]( elementIndex ) + value;
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
bool Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::addElement( const IndexType row,
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
   return true;
}


template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
__cuda_callable__
bool Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::setRowFast( const IndexType row,
                                                        const IndexType* columns,
                                                        const RealType* values,
                                                        const IndexType elements )
{
   TNL_ASSERT( elements <= this->getColumns(),
            std::cerr << " elements = " << elements
                 << " this->columns = " << this->getColumns() );
   for( IndexType i = 0; i < elements; i++ )
      this->setElementFast( row, columns[ i ], values[ i ] );
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
bool Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::setRow( const IndexType row,
                                                    const IndexType* columns,
                                                    const RealType* values,
                                                    const IndexType elements )
{
   TNL_ASSERT( elements <= this->getColumns(),
            std::cerr << " elements = " << elements
                 << " this->columns = " << this->getColumns() );
   for( IndexType i = 0; i < elements; i++ )
      this->setElement( row, columns[ i ], values[ i ] );
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
__cuda_callable__
bool Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::addRowFast( const IndexType row,
                                                        const IndexType* columns,
                                                        const RealType* values,
                                                        const IndexType elements,
                                                        const RealType& thisRowMultiplicator )
{
   TNL_ASSERT( elements <= this->columns,
            std::cerr << " elements = " << elements
                 << " this->columns = " << this->columns );
   for( IndexType i = 0; i < elements; i++ )
      this->setElementFast( row, columns[ i ],
                            thisRowMultiplicator * this->getElementFast( row, columns[ i ] ) + values[ i ] );
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
bool Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::addRow( const IndexType row,
                                                    const IndexType* columns,
                                                    const RealType* values,
                                                    const IndexType elements,
                                                    const RealType& thisRowMultiplicator )
{
   TNL_ASSERT( elements <= this->columns,
            std::cerr << " elements = " << elements
                 << " this->columns = " << this->columns );
   for( IndexType i = 0; i < elements; i++ )
      this->setElement( row, columns[ i ],
                        thisRowMultiplicator * this->getElement( row, columns[ i ] ) + values[ i ] );
   return true;
}


template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
__cuda_callable__
const Real& Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::getElementFast( const IndexType row,
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
          bool RowMajorOrder,
          typename RealAllocator >
Real Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::getElement( const IndexType row,
                                                        const IndexType column ) const
{
   return this->values.getElement( this->getElementIndex( row, column ) );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
__cuda_callable__
void Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::getRowFast( const IndexType row,
                                                        IndexType* columns,
                                                        RealType* values ) const
{
   for( IndexType i = 0; i < this->getColumns(); i++ )
   {
      columns[ i ] = i;
      values[ i ] = this->getElementFast( row, i );
   }
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
__cuda_callable__
typename Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::MatrixRow
Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::
getRow( const IndexType rowIndex )
{
   if( std::is_same< Device, Devices::Host >::value )
      return MatrixRow( &this->values.getData()[ this->getElementIndex( rowIndex, 0 ) ],
                        this->columns,
                        1 );
   if( std::is_same< Device, Devices::Cuda >::value )
      return MatrixRow( &this->values.getData()[ this->getElementIndex( rowIndex, 0 ) ],
                        this->columns,
                        this->rows );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
__cuda_callable__
const typename Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::MatrixRow
Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::
getRow( const IndexType rowIndex ) const
{
   if( std::is_same< Device, Devices::Host >::value )
      return MatrixRow( &this->values.getData()[ this->getElementIndex( rowIndex, 0 ) ],
                        this->columns,
                        1 );
   if( std::is_same< Device, Devices::Cuda >::value )
      return MatrixRow( &this->values.getData()[ this->getElementIndex( rowIndex, 0 ) ],
                        this->columns,
                        this->rows );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
   template< typename Vector >
__cuda_callable__
typename Vector::RealType Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::rowVectorProduct( const IndexType row,
                                                                                   const Vector& vector ) const
{
   RealType sum( 0.0 );
   for( IndexType column = 0; column < this->getColumns(); column++ )
      sum += this->getElementFast( row, column ) * vector[ column ];
   return sum;
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
   template< typename InVector,
             typename OutVector >
void Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::vectorProduct( const InVector& inVector,
                                                           OutVector& outVector ) const
{
   TNL_ASSERT( this->getColumns() == inVector.getSize(),
            std::cerr << "Matrix columns: " << this->getColumns() << std::endl
                 << "Vector size: " << inVector.getSize() << std::endl );
   TNL_ASSERT( this->getRows() == outVector.getSize(),
               std::cerr << "Matrix rows: " << this->getRows() << std::endl
                    << "Vector size: " << outVector.getSize() << std::endl );

   DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
   template< typename Matrix >
void Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::addMatrix( const Matrix& matrix,
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

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator,
          typename Matrix1,
          typename Matrix2,
          int tileDim,
          int tileRowBlockSize >
__global__ void DenseMatrixProductKernel( Dense< Real, Devices::Cuda, Index >* resultMatrix,
                                                   const Matrix1* matrixA,
                                                   const Matrix2* matrixB,
                                                   const Real matrixAMultiplicator,
                                                   const Real matrixBMultiplicator,
                                                   const Index gridIdx_x,
                                                   const Index gridIdx_y )
{
   /****
    * Here we compute product C = A * B. To profit from the fast
    * shared memory we do it by tiles.
    */

   typedef Index IndexType;
   typedef Real RealType;
   __shared__ Real tileA[ tileDim*tileDim ];
   __shared__ Real tileB[ tileDim*tileDim ];
   __shared__ Real tileC[ tileDim*tileDim ];

   const IndexType& matrixARows = matrixA->getRows();
   const IndexType& matrixAColumns = matrixA->getColumns();
   const IndexType& matrixBRows = matrixB->getRows();
   const IndexType& matrixBColumns = matrixB->getColumns();

   /****
    * Reset the tile C
    */
   for( IndexType row = 0; row < tileDim; row += tileRowBlockSize )
      tileC[ ( row + threadIdx.y )*tileDim + threadIdx.x ] = 0.0;

   /****
    * Compute the result tile coordinates
    */
   const IndexType resultTileRow = ( gridIdx_y*gridDim.y + blockIdx.y )*tileDim;
   const IndexType resultTileColumn = ( gridIdx_x*gridDim.x + blockIdx.x )*tileDim;

   /****
    * Sum over the matrix tiles
    */
   for( IndexType i = 0; i < matrixAColumns; i += tileDim )
   {
      for( IndexType row = 0; row < tileDim; row += tileRowBlockSize )
      {
         const IndexType matrixARow = resultTileRow + threadIdx.y + row;
         const IndexType matrixAColumn = i + threadIdx.x;
         if( matrixARow < matrixARows && matrixAColumn < matrixAColumns )
            tileA[ (threadIdx.y + row)*tileDim + threadIdx.x ] =
               matrixAMultiplicator * matrixA->getElementFast( matrixARow,  matrixAColumn );

         const IndexType matrixBRow = i + threadIdx.y + row;
         const IndexType matrixBColumn = resultTileColumn + threadIdx.x;
         if( matrixBRow < matrixBRows && matrixBColumn < matrixBColumns )
            tileB[ (threadIdx.y + row)*tileDim + threadIdx.x ] =
               matrixBMultiplicator * matrixB->getElementFast( matrixBRow, matrixBColumn );
      }
      __syncthreads();

      const IndexType tileALastRow = tnlCudaMin( tileDim, matrixARows - resultTileRow );
      const IndexType tileALastColumn = tnlCudaMin( tileDim, matrixAColumns - i );
      const IndexType tileBLastRow = tnlCudaMin( tileDim, matrixBRows - i );
      const IndexType tileBLastColumn =
         tnlCudaMin( tileDim, matrixBColumns - resultTileColumn );

      for( IndexType row = 0; row < tileALastRow; row += tileRowBlockSize )
      {
         RealType sum( 0.0 );
         for( IndexType j = 0; j < tileALastColumn; j++ )
            sum += tileA[ ( threadIdx.y + row )*tileDim + j ]*
                      tileB[ j*tileDim + threadIdx.x ];
         tileC[ ( row + threadIdx.y )*tileDim + threadIdx.x ] += sum;
      }
      __syncthreads();
   }

   /****
    * Write the result tile to the result matrix
    */
   const IndexType& matrixCRows = resultMatrix->getRows();
   const IndexType& matrixCColumns = resultMatrix->getColumns();
   for( IndexType row = 0; row < tileDim; row += tileRowBlockSize )
   {
      const IndexType matrixCRow = resultTileRow + row + threadIdx.y;
      const IndexType matrixCColumn = resultTileColumn + threadIdx.x;
      if( matrixCRow < matrixCRows && matrixCColumn < matrixCColumns )
         resultMatrix->setElementFast( matrixCRow,
                                       matrixCColumn,
                                       tileC[ ( row + threadIdx.y )*tileDim + threadIdx.x ] );
   }

}
#endif

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
   template< typename Matrix1, typename Matrix2, int tileDim >
void Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::getMatrixProduct( const Matrix1& matrix1,
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
      dim3 cudaBlockSize( 0 ), cudaGridSize( 0 );
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
         }
#endif
   }
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename Matrix,
          bool RowMajorOrder,
          typename RealAllocator,
          int tileDim,
          int tileRowBlockSize >
__global__ void DenseTranspositionAlignedKernel( Dense< Real, Devices::Cuda, Index >* resultMatrix,
                                                          const Matrix* inputMatrix,
                                                          const Real matrixMultiplicator,
                                                          const Index gridIdx_x,
                                                          const Index gridIdx_y )
{
   __shared__ Real tile[ tileDim*tileDim ];

   const Index columns = inputMatrix->getColumns();
   const Index rows = inputMatrix->getRows();


   /****
    * Diagonal mapping of the CUDA blocks
    */
   Index blockIdx_x, blockIdx_y;
   if( columns == rows )
   {
      blockIdx_y = blockIdx.x;
      blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
   }
   else
   {
      Index bID = blockIdx.x + gridDim.x*blockIdx.y;
      blockIdx_y = bID % gridDim.y;
      blockIdx_x = ( ( bID / gridDim.y ) + blockIdx_y ) % gridDim.x;
   }

   /****
    * Read the tile to the shared memory
    */
   const Index readRowPosition =
      ( gridIdx_y*gridDim.y + blockIdx_y )*tileDim + threadIdx.y;
   const Index readColumnPosition =
      ( gridIdx_x*gridDim.x + blockIdx_x )*tileDim + threadIdx.x;
   for( Index rowBlock = 0;
        rowBlock < tileDim;
        rowBlock += tileRowBlockSize )
   {
      tile[ Cuda::getInterleaving( threadIdx.x*tileDim +  threadIdx.y + rowBlock ) ] =
               inputMatrix->getElementFast( readColumnPosition,
                                            readRowPosition + rowBlock );
   }
   __syncthreads();

   /****
    * Write the tile to the global memory
    */
   const Index writeRowPosition =
      ( gridIdx_x*gridDim.x + blockIdx_x )*tileDim + threadIdx.y;
   const Index writeColumnPosition =
      ( gridIdx_y*gridDim.y + blockIdx_y )*tileDim + threadIdx.x;
   for( Index rowBlock = 0;
        rowBlock < tileDim;
        rowBlock += tileRowBlockSize )
   {
      resultMatrix->setElementFast( writeColumnPosition,
                                    writeRowPosition + rowBlock,
                                    matrixMultiplicator * tile[ Cuda::getInterleaving( ( threadIdx.y + rowBlock ) * tileDim + threadIdx.x ) ] );

   }

}

template< typename Real,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator,
          typename Matrix,
          int tileDim,
          int tileRowBlockSize >
__global__ void DenseTranspositionNonAlignedKernel( Dense< Real, Devices::Cuda, Index >* resultMatrix,
                                                             const Matrix* inputMatrix,
                                                             const Real matrixMultiplicator,
                                                             const Index gridIdx_x,
                                                             const Index gridIdx_y )
{
   __shared__ Real tile[ tileDim*tileDim ];

   const Index columns = inputMatrix->getColumns();
   const Index rows = inputMatrix->getRows();

   /****
    * Diagonal mapping of the CUDA blocks
    */
   Index blockIdx_x, blockIdx_y;
   if( columns == rows )
   {
      blockIdx_y = blockIdx.x;
      blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
   }
   else
   {
      Index bID = blockIdx.x + gridDim.x*blockIdx.y;
      blockIdx_y = bID % gridDim.y;
      blockIdx_x = ( ( bID / gridDim.y ) + blockIdx_y ) % gridDim.x;
   }

   /****
    * Read the tile to the shared memory
    */
   const Index readRowPosition =
      ( gridIdx_y*gridDim.y + blockIdx_y )*tileDim + threadIdx.y;
   const Index readColumnPosition =
      ( gridIdx_x*gridDim.x + blockIdx_x )*tileDim + threadIdx.x;
   if( readColumnPosition < columns )
   {
      const Index readOffset = readRowPosition * columns + readColumnPosition;
      for( Index rowBlock = 0;
           rowBlock < tileDim;
           rowBlock += tileRowBlockSize )
      {
         if( readRowPosition + rowBlock < rows )
            tile[ Cuda::getInterleaving( threadIdx.x*tileDim +  threadIdx.y + rowBlock ) ] =
               inputMatrix->getElementFast( readColumnPosition,
                                            readRowPosition + rowBlock );
      }
   }
   __syncthreads();

   /****
    * Write the tile to the global memory
    */
   const Index writeRowPosition =
      ( gridIdx_x*gridDim.x + blockIdx_x )*tileDim + threadIdx.y;
   const Index writeColumnPosition =
      ( gridIdx_y*gridDim.y + blockIdx_y )*tileDim + threadIdx.x;
   if( writeColumnPosition < rows )
   {
      const Index writeOffset = writeRowPosition * rows + writeColumnPosition;
      for( Index rowBlock = 0;
           rowBlock < tileDim;
           rowBlock += tileRowBlockSize )
      {
         if( writeRowPosition + rowBlock < columns )
            resultMatrix->setElementFast( writeColumnPosition,
                                          writeRowPosition + rowBlock,
                                          matrixMultiplicator * tile[ Cuda::getInterleaving( ( threadIdx.y + rowBlock ) * tileDim + threadIdx.x ) ] );
      }
   }

}


#endif

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
   template< typename Matrix, int tileDim >
void Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::getTransposition( const Matrix& matrix,
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
      dim3 cudaBlockSize( 0 ), cudaGridSize( 0 );
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
      Cuda::freeFromDevice( matrix_device );
#endif
   }
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
   template< typename Vector1, typename Vector2 >
void Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::performSORIteration( const Vector1& b,
                                                        const IndexType row,
                                                        Vector2& x,
                                                        const RealType& omega ) const
{
   RealType sum( 0.0 ), diagonalValue;
   for( IndexType i = 0; i < this->getColumns(); i++ )
   {
      if( i == row )
         diagonalValue = this->getElementFast( row, row );
      else
         sum += this->getElementFast( row, i ) * x[ i ];
   }
   x[ row ] = ( 1.0 - omega ) * x[ row ] + omega / diagonalValue * ( b[ row ] - sum );
}


// copy assignment
template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
Dense< Real, Device, Index, RowMajorOrder, RealAllocator >&
Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::operator=( const Dense& matrix )
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
   template< typename Real2, typename Device2, typename Index2, typename >
Dense< Real, Device, Index, RowMajorOrder, RealAllocator >&
Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::operator=( const Dense< Real2, Device2, Index2 >& matrix )
{
   static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value,
                  "unknown device" );
   static_assert( std::is_same< Device2, Devices::Host >::value || std::is_same< Device2, Devices::Cuda >::value,
                  "unknown device" );

   this->setLike( matrix );

   throw Exceptions::NotImplementedError("Cross-device assignment for the Dense format is not implemented yet.");
}


template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
void Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
void Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
void Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::save( File& file ) const
{
   Matrix< Real, Device, Index >::save( file );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
void Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::load( File& file )
{
   Matrix< Real, Device, Index >::load( file );
}

template< typename Real,
          typename Device,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator >
void Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::print( std::ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      for( IndexType column = 0; column < this->getColumns(); column++ )
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
Index Dense< Real, Device, Index, RowMajorOrder, RealAllocator >::getElementIndex( const IndexType row,
                                                              const IndexType column ) const
{
   TNL_ASSERT( ( std::is_same< Device, Devices::Host >::value ||
          std::is_same< Device, Devices::Cuda >::value ), )
   if( std::is_same< Device, Devices::Host >::value )
      return row * this->columns + column;
   if( std::is_same< Device, Devices::Cuda >::value )
      return column * this->rows + row;
   return -1;
}

template<>
class DenseDeviceDependentCode< Devices::Host >
{
   public:

      typedef Devices::Host Device;

      template< typename Real,
                typename Index,
                bool RowMajorOrder,
                typename RealAllocator,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const Dense< Real, Device, Index, RowMajorOrder, RealAllocator >& matrix,
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
class DenseDeviceDependentCode< Devices::Cuda >
{
   public:

      typedef Devices::Cuda Device;

      template< typename Real,
                typename Index,
                bool RowMajorOrder,
                typename RealAllocator,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const Dense< Real, Device, Index, RowMajorOrder, RealAllocator >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         MatrixVectorProductCuda( matrix, inVector, outVector );
      }
};

} // namespace Matrices
} // namespace TNL
