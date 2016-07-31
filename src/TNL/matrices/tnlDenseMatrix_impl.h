/***************************************************************************
                          tnlDenseMatrix_impl.h  -  description
                             -------------------
    begin                : Nov 29, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Assert.h>
#include <TNL/matrices/tnlDenseMatrix.h>

#ifdef HAVE_CUDA
#include <TNL/core/cuda/reduction-operations.h>
#endif

namespace TNL {

template< typename Real,
          typename Device,
          typename Index >
tnlDenseMatrix< Real, Device, Index >::tnlDenseMatrix()
{
}

template< typename Real,
          typename Device,
          typename Index >
String tnlDenseMatrix< Real, Device, Index >::getType()
{
   return String( "tnlDenseMatrix< " ) +
          String( TNL::getType< RealType >() ) + ", " +
          String( Device :: getDeviceType() ) + ", " +
          String( TNL::getType< IndexType >() ) + " >";
}

template< typename Real,
          typename Device,
          typename Index >
String tnlDenseMatrix< Real, Device, Index >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlDenseMatrix< Real, Device, Index >::setDimensions( const IndexType rows,
                                                           const IndexType columns )
{
   if( ! tnlMatrix< Real, Device, Index >::setDimensions( rows, columns ) ||
       ! this->values.setSize( rows * columns ) )
     return false;
   this->values.setValue( 0.0 );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlDenseMatrix< Real, Device, Index >::setLike( const tnlDenseMatrix< Real2, Device2, Index2 >& matrix )
{
   return this->setDimensions( matrix.getRows(), matrix.getColumns() );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlDenseMatrix< Real, Device, Index >::setCompressedRowsLengths( const CompressedRowsLengthsVector& rowLengths )
{
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlDenseMatrix< Real, Device, Index >::getRowLength( const IndexType row ) const
{
   return this->getColumns();
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlDenseMatrix< Real, Device, Index >::getMaxRowLength() const
{
   return this->getColumns();
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlDenseMatrix< Real, Device, Index >::getNumberOfMatrixElements() const
{
   return this->getRows() * this->getColumns();
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlDenseMatrix< Real, Device, Index >::getNumberOfNonzeroMatrixElements() const
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
          typename Index >
void tnlDenseMatrix< Real, Device, Index >::reset()
{
   tnlMatrix< Real, Device, Index >::reset();
   this->values.reset();
}

template< typename Real,
          typename Device,
          typename Index >
void tnlDenseMatrix< Real, Device, Index >::setValue( const Real& value )
{
   this->values.setValue( value );
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool tnlDenseMatrix< Real, Device, Index >::setElementFast( const IndexType row,
                                                            const IndexType column,
                                                            const RealType& value )
{
   Assert( row >= 0 && row < this->getRows() &&
              column >= 0 && column < this->getColumns(),
              std::cerr << " row = " << row << " column = " << column << " this->getRows() = " << this->getRows()
                   << " this->getColumns() = " << this->getColumns() );
   this->values.operator[]( this->getElementIndex( row, column ) ) = value;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlDenseMatrix< Real, Device, Index >::setElement( const IndexType row,
                                                        const IndexType column,
                                                        const RealType& value )
{
   this->values.setElement( this->getElementIndex( row, column ), value );
   return true;
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool tnlDenseMatrix< Real, Device, Index >::addElementFast( const IndexType row,
                                                            const IndexType column,
                                                            const RealType& value,
                                                            const RealType& thisElementMultiplicator )
{
   Assert( row >= 0 && row < this->getRows() &&
              column >= 0 && column < this->getColumns(),
              printf( " row = %d, column = %d, this->getRows = %d, this->getColumns() = %d \n", row, column, this->getRows(), this->getColumns() ) );
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
          typename Index >
bool tnlDenseMatrix< Real, Device, Index >::addElement( const IndexType row,
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
          typename Index >
__cuda_callable__
bool tnlDenseMatrix< Real, Device, Index >::setRowFast( const IndexType row,
                                                        const IndexType* columns,
                                                        const RealType* values,
                                                        const IndexType elements )
{
   Assert( elements <= this->getColumns(),
            std::cerr << " elements = " << elements
                 << " this->columns = " << this->getColumns() );
   for( IndexType i = 0; i < elements; i++ )
      this->setElementFast( row, columns[ i ], values[ i ] );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlDenseMatrix< Real, Device, Index >::setRow( const IndexType row,
                                                    const IndexType* columns,
                                                    const RealType* values,
                                                    const IndexType elements )
{
   Assert( elements <= this->getColumns(),
            std::cerr << " elements = " << elements
                 << " this->columns = " << this->getColumns() );
   for( IndexType i = 0; i < elements; i++ )
      this->setElement( row, columns[ i ], values[ i ] );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool tnlDenseMatrix< Real, Device, Index >::addRowFast( const IndexType row,
                                                        const IndexType* columns,
                                                        const RealType* values,
                                                        const IndexType elements,
                                                        const RealType& thisRowMultiplicator )
{
   Assert( elements <= this->columns,
            std::cerr << " elements = " << elements
                 << " this->columns = " << this->columns );
   for( IndexType i = 0; i < elements; i++ )
      this->setElementFast( row, columns[ i ],
                            thisRowMultiplicator * this->getElementFast( row, columns[ i ] ) + values[ i ] );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlDenseMatrix< Real, Device, Index >::addRow( const IndexType row,
                                                    const IndexType* columns,
                                                    const RealType* values,
                                                    const IndexType elements,
                                                    const RealType& thisRowMultiplicator )
{
   Assert( elements <= this->columns,
            std::cerr << " elements = " << elements
                 << " this->columns = " << this->columns );
   for( IndexType i = 0; i < elements; i++ )
      this->setElement( row, columns[ i ],
                        thisRowMultiplicator * this->getElement( row, columns[ i ] ) + values[ i ] );
   return true;
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlDenseMatrix< Real, Device, Index >::getElementFast( const IndexType row,
                                                            const IndexType column ) const
{
   Assert( row >= 0 && row < this->getRows() &&
              column >= 0 && column < this->getColumns(),
              printf( " row = %d, column = %d, this->getRows = %d, this->getColumns() = %d \n", row, column, this->getRows(), this->getColumns() ) );
   return this->values.operator[]( this->getElementIndex( row, column ) );
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlDenseMatrix< Real, Device, Index >::getElement( const IndexType row,
                                                        const IndexType column ) const
{
   return this->values.getElement( this->getElementIndex( row, column ) );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
void tnlDenseMatrix< Real, Device, Index >::getRowFast( const IndexType row,
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
          typename Index >
__cuda_callable__
typename tnlDenseMatrix< Real, Device, Index >::MatrixRow
tnlDenseMatrix< Real, Device, Index >::
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
          typename Index >
__cuda_callable__
const typename tnlDenseMatrix< Real, Device, Index >::MatrixRow
tnlDenseMatrix< Real, Device, Index >::
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
          typename Index >
   template< typename Vector >
__cuda_callable__
typename Vector::RealType tnlDenseMatrix< Real, Device, Index >::rowVectorProduct( const IndexType row,
                                                                                   const Vector& vector ) const
{
   RealType sum( 0.0 );
   for( IndexType column = 0; column < this->getColumns(); column++ )
      sum += this->getElementFast( row, column ) * vector[ column ];
   return sum;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename InVector,
             typename OutVector >
void tnlDenseMatrix< Real, Device, Index >::vectorProduct( const InVector& inVector,
                                                           OutVector& outVector ) const
{
   Assert( this->getColumns() == inVector.getSize(),
            std::cerr << "Matrix columns: " << this->getColumns() << std::endl
                 << "Vector size: " << inVector.getSize() << std::endl );
   Assert( this->getRows() == outVector.getSize(),
               std::cerr << "Matrix rows: " << this->getRows() << std::endl
                    << "Vector size: " << outVector.getSize() << std::endl );

   DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Matrix >
void tnlDenseMatrix< Real, Device, Index >::addMatrix( const Matrix& matrix,
                                                       const RealType& matrixMultiplicator,
                                                       const RealType& thisMatrixMultiplicator )
{
   Assert( this->getColumns() == matrix.getColumns() &&
              this->getRows() == matrix.getRows(),
            std::cerr << "This matrix columns: " << this->getColumns() << std::endl
                 << "This matrix rows: " << this->getRows() << std::endl
                 << "That matrix columns: " << matrix.getColumns() << std::endl
                 << "That matrix rows: " << matrix.getRows() << std::endl );

   if( thisMatrixMultiplicator == 1.0 )
      this->values.addVector( matrix.values, matrixMultiplicator );
   else
      this->values.addVector( matrix.values, matrixMultiplicator, thisMatrixMultiplicator );
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename Matrix1,
          typename Matrix2,
          int tileDim,
          int tileRowBlockSize >
__global__ void tnlDenseMatrixMatrixProductKernel( tnlDenseMatrix< Real, Devices::Cuda, Index >* resultMatrix,
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
          typename Index >
   template< typename Matrix1, typename Matrix2, int tileDim >
void tnlDenseMatrix< Real, Device, Index >::getMatrixProduct( const Matrix1& matrix1,
                                                              const Matrix2& matrix2,
                                                              const RealType& matrix1Multiplicator,
                                                              const RealType& matrix2Multiplicator )
{
   Assert( matrix1.getColumns() == matrix2.getRows() &&
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
      const IndexType rowGrids = roundUpDivision( rowTiles, Devices::Cuda::getMaxGridSize() );
      const IndexType columnGrids = roundUpDivision( columnTiles, Devices::Cuda::getMaxGridSize() );

      for( IndexType gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ )
         for( IndexType gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ )
         {
            cudaGridSize.x = cudaGridSize.y = Devices::Cuda::getMaxGridSize();
            if( gridIdx_x == columnGrids - 1 )
               cudaGridSize.x = columnTiles % Devices::Cuda::getMaxGridSize();
            if( gridIdx_y == rowGrids - 1 )
               cudaGridSize.y = rowTiles % Devices::Cuda::getMaxGridSize();
            ThisType* this_kernel = Devices::Cuda::passToDevice( *this );
            Matrix1* matrix1_kernel = Devices::Cuda::passToDevice( matrix1 );
            Matrix2* matrix2_kernel = Devices::Cuda::passToDevice( matrix2 );
            tnlDenseMatrixMatrixProductKernel< Real,
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
            Devices::Cuda::freeFromDevice( this_kernel );
            Devices::Cuda::freeFromDevice( matrix1_kernel );
            Devices::Cuda::freeFromDevice( matrix2_kernel );
         }
#endif
   }
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename Matrix,
          int tileDim,
          int tileRowBlockSize >
__global__ void tnlDenseMatrixTranspositionAlignedKernel( tnlDenseMatrix< Real, Devices::Cuda, Index >* resultMatrix,
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
      tile[ Devices::Cuda::getInterleaving( threadIdx.x*tileDim +  threadIdx.y + rowBlock ) ] =
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
                                    matrixMultiplicator * tile[ Devices::Cuda::getInterleaving( ( threadIdx.y + rowBlock ) * tileDim + threadIdx.x ) ] );

   }

}

template< typename Real,
          typename Index,
          typename Matrix,
          int tileDim,
          int tileRowBlockSize >
__global__ void tnlDenseMatrixTranspositionNonAlignedKernel( tnlDenseMatrix< Real, Devices::Cuda, Index >* resultMatrix,
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
            tile[ Devices::Cuda::getInterleaving( threadIdx.x*tileDim +  threadIdx.y + rowBlock ) ] =
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
                                          matrixMultiplicator * tile[ Devices::Cuda::getInterleaving( ( threadIdx.y + rowBlock ) * tileDim + threadIdx.x ) ] );
      }
   }

}


#endif

template< typename Real,
          typename Device,
          typename Index >
   template< typename Matrix, int tileDim >
void tnlDenseMatrix< Real, Device, Index >::getTransposition( const Matrix& matrix,
                                                              const RealType& matrixMultiplicator )
{
   Assert( this->getColumns() == matrix.getRows() &&
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
      const IndexType rowGrids = roundUpDivision( rowTiles, Devices::Cuda::getMaxGridSize() );
      const IndexType columnGrids = roundUpDivision( columnTiles, Devices::Cuda::getMaxGridSize() );
      const IndexType sharedMemorySize = tileDim*tileDim + tileDim*tileDim/Devices::Cuda::getNumberOfSharedMemoryBanks();

      ThisType* this_device = Devices::Cuda::passToDevice( *this );
      Matrix* matrix_device = Devices::Cuda::passToDevice( matrix );

      for( IndexType gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ )
         for( IndexType gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ )
         {
            cudaGridSize.x = cudaGridSize.y = Devices::Cuda::getMaxGridSize();
            if( gridIdx_x == columnGrids - 1)
               cudaGridSize.x = columnTiles % Devices::Cuda::getMaxGridSize();
            if( gridIdx_y == rowGrids - 1 )
               cudaGridSize.y = rowTiles % Devices::Cuda::getMaxGridSize();
            if( ( gridIdx_x < columnGrids - 1 || matrix.getColumns() % tileDim == 0 ) &&
                ( gridIdx_y < rowGrids - 1 || matrix.getRows() % tileDim == 0 ) )
            {
               tnlDenseMatrixTranspositionAlignedKernel< Real,
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
               tnlDenseMatrixTranspositionNonAlignedKernel< Real,
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
            checkCudaDevice;
         }
      Devices::Cuda::freeFromDevice( this_device );
      Devices::Cuda::freeFromDevice( matrix_device );
#endif
   }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
void tnlDenseMatrix< Real, Device, Index >::performSORIteration( const Vector& b,
                                                                 const IndexType row,
                                                                 Vector& x,
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

template< typename Real,
          typename Device,
          typename Index >
bool tnlDenseMatrix< Real, Device, Index >::save( const String& fileName ) const
{
   return Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlDenseMatrix< Real, Device, Index >::load( const String& fileName )
{
   return Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlDenseMatrix< Real, Device, Index >::save( File& file ) const
{
   if( ! tnlMatrix< Real, Device, Index >::save( file )  )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlDenseMatrix< Real, Device, Index >::load( File& file )
{
   if( ! tnlMatrix< Real, Device, Index >::load( file ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlDenseMatrix< Real, Device, Index >::print( std::ostream& str ) const
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
          typename Index >
__cuda_callable__
Index tnlDenseMatrix< Real, Device, Index >::getElementIndex( const IndexType row,
                                                              const IndexType column ) const
{
   Assert( ( std::is_same< Device, Devices::Host >::value ||
          std::is_same< Device, Devices::Cuda >::value ), )
   if( std::is_same< Device, Devices::Host >::value )
      return row * this->columns + column;
   if( std::is_same< Device, Devices::Cuda >::value )
      return column * this->rows + row;
   return -1;
}

template<>
class tnlDenseMatrixDeviceDependentCode< Devices::Host >
{
   public:

      typedef Devices::Host Device;

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const tnlDenseMatrix< Real, Device, Index >& matrix,
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
class tnlDenseMatrixDeviceDependentCode< Devices::Cuda >
{
   public:

      typedef Devices::Cuda Device;

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const tnlDenseMatrix< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         tnlMatrixVectorProductCuda( matrix, inVector, outVector );
      }
};

} // namespace TNL
