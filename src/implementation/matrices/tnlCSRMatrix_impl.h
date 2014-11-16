/***************************************************************************
                          tnlCSRMatrix_impl.h  -  description
                             -------------------
    begin                : Dec 10, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLCSRMATRIX_IMPL_H_
#define TNLCSRMATRIX_IMPL_H_

#include <matrices/tnlCSRMatrix.h>
#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlSharedVector.h>
#include <core/mfuncs.h>

template< typename Real,
          typename Device,
          typename Index >
tnlCSRMatrix< Real, Device, Index >::tnlCSRMatrix()
: spmvCudaKernel( hybrid ),
  cudaWarpSize( 32 ), //tnlCuda::getWarpSize() )
  hybridModeSplit( 4 )
{
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlCSRMatrix< Real, Device, Index >::getType()
{
   return tnlString( "tnlCSRMatrix< ") +
          tnlString( ::getType< Real>() ) +
          tnlString( ", " ) +
          Device :: getDeviceType() +
          tnlString( " >" );
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlCSRMatrix< Real, Device, Index >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlCSRMatrix< Real, Device, Index >::setDimensions( const IndexType rows,
                                                         const IndexType columns )
{
   if( ! tnlSparseMatrix< Real, Device, Index >::setDimensions( rows, columns ) ||
       ! this->rowPointers.setSize( this->rows + 1 ) )
      return false;
   this->rowPointers.setValue( 0 );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlCSRMatrix< Real, Device, Index >::setRowLengths( const RowLengthsVector& rowLengths )
{
   /****
    * Compute the rows pointers. The last one is
    * the end of the last row and so it says the
    * necessary length of the vectors this->values
    * and this->columnIndexes.
    */
   tnlAssert( this->getRows() > 0, );
   tnlAssert( this->getColumns() > 0, );
   tnlSharedVector< IndexType, DeviceType, IndexType > rowPtrs;
   rowPtrs.bind( this->rowPointers.getData(), this->getRows() );
   rowPtrs = rowLengths;
   this->rowPointers.setElement( this->rows, 0 );
   this->rowPointers.computeExclusivePrefixSum();
   this->maxRowLength = rowLengths.max();

   /****
    * Allocate values and column indexes
    */
   if( ! this->values.setSize( this->rowPointers.getElement( this->rows ) ) ||
       ! this->columnIndexes.setSize( this->rowPointers.getElement( this->rows ) ) )
      return false;
   this->columnIndexes.setValue( this->columns );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlCSRMatrix< Real, Device, Index >::getRowLength( const IndexType row ) const
{
   return this->rowPointers[ row + 1 ] - this->rowPointers[ row ];
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlCSRMatrix< Real, Device, Index >::setLike( const tnlCSRMatrix< Real2, Device2, Index2 >& matrix )
{
   if( ! tnlSparseMatrix< Real, Device, Index >::setLike( matrix ) ||
       ! this->rowPointers.setLike( matrix.rowPointers ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlCSRMatrix< Real, Device, Index >::reset()
{
   tnlSparseMatrix< Real, Device, Index >::reset();
   this->rowPointers.reset();
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlCSRMatrix< Real, Device, Index >::setElementFast( const IndexType row,
                                                          const IndexType column,
                                                          const Real& value )
{
   return this->addElementFast( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlCSRMatrix< Real, Device, Index >::setElement( const IndexType row,
                                                      const IndexType column,
                                                      const Real& value )
{
   return this->addElement( row, column, value, 0.0 );
}


template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlCSRMatrix< Real, Device, Index >::addElementFast( const IndexType row,
                                                          const IndexType column,
                                                          const RealType& value,
                                                          const RealType& thisElementMultiplicator )
{
   /*tnlAssert( row >= 0 && row < this->rows &&
              column >= 0 && column <= this->rows,
              cerr << " row = " << row
                   << " column = " << column
                   << " this->rows = " << this->rows
                   << " this->columns = " << this-> columns );*/

   IndexType elementPtr = this->rowPointers[ row ];
   const IndexType rowEnd = this->rowPointers[ row + 1 ];
   IndexType col;
   while( elementPtr < rowEnd &&
          ( col = this->columnIndexes[ elementPtr ] ) < column &&
          col != this->getPaddingIndex() ) elementPtr++;
   if( elementPtr == rowEnd )
      return false;
   if( col == column )
   {
      this->values[ elementPtr ] = thisElementMultiplicator * this->values[ elementPtr ] + value;
      return true;
   }
   else
      if( col == this->getPaddingIndex() )
      {
         this->columnIndexes[ elementPtr ] = column;
         this->values[ elementPtr ] = value;
         return true;
      }
      else
      {
         IndexType j = rowEnd - 1;
         while( j > elementPtr )
         {
            this->columnIndexes[ j ] = this->columnIndexes[ j - 1 ];
            this->values[ j ] = this->values[ j - 1 ];
            j--;
         }
         this->columnIndexes[ elementPtr ] = column;
         this->values[ elementPtr ] = value;
         return true;
      }
   return false;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlCSRMatrix< Real, Device, Index >::addElement( const IndexType row,
                                                      const IndexType column,
                                                      const RealType& value,
                                                      const RealType& thisElementMultiplicator )
{
   tnlAssert( row >= 0 && row < this->rows &&
               column >= 0 && column < this->columns,
               cerr << " row = " << row
                    << " column = " << column
                    << " this->rows = " << this->rows
                    << " this->columns = " << this->columns );

    IndexType elementPtr = this->rowPointers.getElement( row );
    const IndexType rowEnd = this->rowPointers.getElement( row + 1 );
    IndexType col;
    while( elementPtr < rowEnd &&
           ( col = this->columnIndexes.getElement( elementPtr ) ) < column &&
           col != this->getPaddingIndex() ) elementPtr++;
    if( elementPtr == rowEnd )
       return false;
    if( col == column )
    {
       this->values.setElement( elementPtr, thisElementMultiplicator * this->values.getElement( elementPtr ) + value );
       return true;
    }
    else
       if( col == this->getPaddingIndex() )
       {
          this->columnIndexes.setElement( elementPtr, column );
          this->values.setElement( elementPtr, value );
          return true;
       }
       else
       {
          IndexType j = rowEnd - 1;
          while( j > elementPtr )
          {
             this->columnIndexes.setElement( j, this->columnIndexes.getElement( j - 1 ) );
             this->values.setElement( j, this->values.getElement( j - 1 ) );
             j--;
          }
          this->columnIndexes.setElement( elementPtr, column );
          this->values.setElement( elementPtr, value );
          return true;
       }
    return false;
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlCSRMatrix< Real, Device, Index > :: setRowFast( const IndexType row,
                                                        const IndexType* columnIndexes,
                                                        const RealType* values,
                                                        const IndexType elements )
{
   IndexType elementPointer = this->rowPointers[ row ];
   const IndexType rowLength = this->rowPointers[ row + 1 ] - elementPointer;
   if( elements > rowLength )
      return false;

   for( IndexType i = 0; i < elements; i++ )
   {
      this->columnIndexes[ elementPointer ] = columnIndexes[ i ];
      this->values[ elementPointer ] = values[ i ];
      elementPointer++;
   }
   for( IndexType i = elements; i < rowLength; i++ )
      this->columnIndexes[ elementPointer++ ] = this->getPaddingIndex();
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlCSRMatrix< Real, Device, Index > :: setRow( const IndexType row,
                                                    const IndexType* columnIndexes,
                                                    const RealType* values,
                                                    const IndexType elements )
{
   IndexType elementPointer = this->rowPointers.getElement( row );
   const IndexType rowLength = this->rowPointers.getElement( row + 1 ) - elementPointer;
   if( elements > rowLength )
      return false;

   for( IndexType i = 0; i < elements; i++ )
   {
      this->columnIndexes.setElement( elementPointer, columnIndexes[ i ] );
      this->values.setElement( elementPointer, values[ i ] );
      elementPointer++;
   }
   for( IndexType i = elements; i < rowLength; i++ )
      this->columnIndexes.setElement( elementPointer++, this->getPaddingIndex() );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlCSRMatrix< Real, Device, Index > :: addRowFast( const IndexType row,
                                                        const IndexType* columns,
                                                        const RealType* values,
                                                        const IndexType numberOfElements,
                                                        const RealType& thisElementMultiplicator )
{
   // TODO: implement
   return false;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlCSRMatrix< Real, Device, Index > :: addRow( const IndexType row,
                                                    const IndexType* columns,
                                                    const RealType* values,
                                                    const IndexType numberOfElements,
                                                    const RealType& thisElementMultiplicator )
{
   return this->addRowFast( row, columns, values, numberOfElements, thisElementMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Real tnlCSRMatrix< Real, Device, Index >::getElementFast( const IndexType row,
                                                          const IndexType column ) const
{
   IndexType elementPtr = this->rowPointers[ row ];
   const IndexType rowEnd = this->rowPointers[ row + 1 ];
   IndexType col;
   while( elementPtr < rowEnd &&
          ( col = this->columnIndexes[ elementPtr ] ) < column &&
          col != this->getPaddingIndex() )
      elementPtr++;
   if( elementPtr < rowEnd && col == column )
      return this->values[ elementPtr ];
   return 0.0;
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlCSRMatrix< Real, Device, Index >::getElement( const IndexType row,
                                                      const IndexType column ) const
{
   IndexType elementPtr = this->rowPointers.getElement( row );
   const IndexType rowEnd = this->rowPointers.getElement( row + 1 );
   IndexType col;
   while( elementPtr < rowEnd &&
          ( col = this->columnIndexes.getElement( elementPtr ) ) < column &&
          col != this->getPaddingIndex() )
      elementPtr++;
   if( elementPtr < rowEnd && col == column )
      return this->values.getElement( elementPtr );
   return 0.0;
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
void tnlCSRMatrix< Real, Device, Index >::getRowFast( const IndexType row,
                                                      IndexType* columns,
                                                      RealType* values ) const
{
   IndexType elementPointer = this->rowPointers[ row ];
   const IndexType rowLength = this->rowPointers[ row + 1 ] - elementPointer;
   for( IndexType i = 0; i < rowLength; i++ )
   {
      columns[ i ] = this->columnIndexes[ elementPointer ];
      values[ i ] = this->values[ elementPointer ];
      elementPointer++;
   }
}

template< typename Real,
          typename Device,
          typename Index >
void tnlCSRMatrix< Real, Device, Index >::getRow( const IndexType row,
                                                  IndexType* columns,
                                                  RealType* values ) const
{
   IndexType elementPointer = this->rowPointers.getElement( row );
   const IndexType rowLength = this->rowPointers.getElement( row + 1 ) - elementPointer;
   for( IndexType i = 0; i < rowLength; i++ )
   {
      columns[ i ] = this->columnIndexes.getElement( elementPointer );
      values[ i ] = this->values.getElement( elementPointer );
      elementPointer++;
   }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
typename Vector::RealType tnlCSRMatrix< Real, Device, Index >::rowVectorProduct( const IndexType row,
                                                                                 const Vector& vector ) const
{
   Real result = 0.0;
   IndexType elementPtr = this->rowPointers[ row ];
   const IndexType rowEnd = this->rowPointers[ row + 1 ];
   IndexType column;
   while( elementPtr < rowEnd &&
          ( column = this->columnIndexes[ elementPtr ] ) != this->getPaddingIndex() )
      result += this->values[ elementPtr++ ] * vector[ column ];
   return result;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename InVector, typename OutVector >
void tnlCSRMatrix< Real, Device, Index >::vectorProduct( const InVector& inVector,
                                                         OutVector& outVector ) const
{
   DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Index2 >
void tnlCSRMatrix< Real, Device, Index >::addMatrix( const tnlCSRMatrix< Real2, Device, Index2 >& matrix,
                                                                          const RealType& matrixMultiplicator,
                                                                          const RealType& thisMatrixMultiplicator )
{
   tnlAssert( false, cerr << "TODO: implement" );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Index2 >
void tnlCSRMatrix< Real, Device, Index >::getTransposition( const tnlCSRMatrix< Real2, Device, Index2 >& matrix,
                                                                      const RealType& matrixMultiplicator )
{
   tnlAssert( false, cerr << "TODO: implement" );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
bool tnlCSRMatrix< Real, Device, Index >::performSORIteration( const Vector& b,
                                                               const IndexType row,
                                                               Vector& x,
                                                               const RealType& omega ) const
{
   tnlAssert( row >=0 && row < this->getRows(),
              cerr << "row = " << row
                   << " this->getRows() = " << this->getRows()
                   << " this->getName() = " << this->getName() << endl );

   RealType diagonalValue( 0.0 );
   RealType sum( 0.0 );

   IndexType elementPtr = this->rowPointers[ row ];
   const IndexType rowEnd = this->rowPointers[ row + 1 ];
   IndexType column;
   while( elementPtr < rowEnd && ( column = this->columnIndexes[ elementPtr ] ) != this->getPaddingIndex() )
   {
      if( column == row )
         diagonalValue = this->values[ elementPtr ];
      else
         sum += this->values[ elementPtr ] * x[ column ];
      elementPtr++;
   }
   if( diagonalValue == ( Real ) 0.0 )
   {
      cerr << "There is zero on the diagonal in " << row << "-th row of the matrix " << this->getName() << ". I cannot perform SOR iteration." << endl;
      return false;
   }
   x[ row ] = ( 1.0 - omega ) * x[ row ] + omega / diagonalValue * ( b[ row ] - sum );
   return true;
}


template< typename Real,
          typename Device,
          typename Index >
bool tnlCSRMatrix< Real, Device, Index >::save( tnlFile& file ) const
{
   if( ! tnlSparseMatrix< Real, Device, Index >::save( file ) ||
       ! this->rowPointers.save( file ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlCSRMatrix< Real, Device, Index >::load( tnlFile& file )
{
   if( ! tnlSparseMatrix< Real, Device, Index >::load( file ) ||
       ! this->rowPointers.load( file ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlCSRMatrix< Real, Device, Index >::save( const tnlString& fileName ) const
{
   return tnlObject::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlCSRMatrix< Real, Device, Index >::load( const tnlString& fileName )
{
   return tnlObject::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlCSRMatrix< Real, Device, Index >::print( ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      IndexType elementPtr = this->rowPointers[ row ];
      const IndexType rowEnd = this->rowPointers[ row + 1 ];
      IndexType column;
      while( elementPtr < rowEnd &&
             ( column = this->columnIndexes.getElement( elementPtr ) ) < this->columns &&
             column != this->getPaddingIndex() )
         str << " Col:" << column << "->" << this->values.getElement( elementPtr++ ) << "\t";
      str << endl;
   }
}

template< typename Real,
          typename Device,
          typename Index >
void tnlCSRMatrix< Real, Device, Index >::setCudaKernelType( const SPMVCudaKernel kernel )
{
   this->spmvCudaKernel = kernel;
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
typename tnlCSRMatrix< Real, Device, Index >::SPMVCudaKernel tnlCSRMatrix< Real, Device, Index >::getCudaKernelType() const
{
   return this->spmvCudaKernel;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlCSRMatrix< Real, Device, Index >::setCudaWarpSize( const int warpSize )
{
   this->cudaWarpSize = warpSize;
}

template< typename Real,
          typename Device,
          typename Index >
int tnlCSRMatrix< Real, Device, Index >::getCudaWarpSize() const
{
   return this->cudaWarpSize;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlCSRMatrix< Real, Device, Index >::setHybridModeSplit( const IndexType hybridModeSplit )
{
   this->hybridModeSplit = hybridModeSplit;
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Index tnlCSRMatrix< Real, Device, Index >::getHybridModeSplit() const
{
   return this->hybridModeSplit;
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Device,
          typename Index >
   template< typename InVector,
             typename OutVector,
             int warpSize >
__device__
void tnlCSRMatrix< Real, Device, Index >::spmvCudaVectorized( const InVector& inVector,
                                                              OutVector& outVector,
                                                              const IndexType warpStart,
                                                              const IndexType warpEnd,
                                                              const IndexType inWarpIdx ) const
{
   Real* aux = getSharedMemory< Real >();
   for( IndexType row = warpStart; row < warpEnd; row++ )
   {
      aux[ threadIdx.x ] = 0.0;

      IndexType elementPtr = this->rowPointers[ row ] + inWarpIdx;
      const IndexType rowEnd = this->rowPointers[ row + 1 ];
      IndexType column;
      while( elementPtr < rowEnd &&
             ( column = this->columnIndexes[ elementPtr ] ) < this->getColumns() )
      {
         aux[ threadIdx.x ] += inVector[ column ] * this->values[ elementPtr ];
         elementPtr += warpSize;
      }
      if( warpSize == 32 )
         if( inWarpIdx < 16 ) aux[ threadIdx.x ] += aux[ threadIdx.x + 16 ];
      if( warpSize >= 16 )
         if( inWarpIdx < 8 ) aux[ threadIdx.x ] += aux[ threadIdx.x + 8 ];
      if( warpSize >= 8 )
         if( inWarpIdx < 4 ) aux[ threadIdx.x ] += aux[ threadIdx.x + 4 ];
      if( warpSize >= 4 )
         if( inWarpIdx < 2 ) aux[ threadIdx.x ] += aux[ threadIdx.x + 2 ];
      if( warpSize >= 2 )
         if( inWarpIdx < 1 ) aux[ threadIdx.x ] += aux[ threadIdx.x + 1 ];
      __syncthreads(); // TODO: I am not sure why

      if( inWarpIdx == 0 )
         outVector[ row ] = aux[ threadIdx.x ];
   }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename InVector,
             typename OutVector,
             int warpSize >
__device__
void tnlCSRMatrix< Real, Device, Index >::vectorProductCuda( const InVector& inVector,
                                                             OutVector& outVector,
                                                             int gridIdx ) const
{
   IndexType globalIdx = ( gridIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   const IndexType warpStart = warpSize * ( globalIdx / warpSize );
   const IndexType warpEnd = Min( warpStart + warpSize, this->getRows() );
   const IndexType inWarpIdx = globalIdx % warpSize;

   if( this->getCudaKernelType() == vector )
      spmvCudaVectorized< InVector, OutVector, warpSize >( inVector, outVector, warpStart, warpEnd, inWarpIdx );

   /****
    * Hybrid mode
    */
   const Index firstRow = ( gridIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x;
   const IndexType lastRow = Min( this->getRows(), firstRow + blockDim. x );
   const IndexType nonzerosPerRow = ( this->rowPointers[ lastRow ] - this->rowPointers[ firstRow ] ) /
                                    ( lastRow - firstRow );

   if( nonzerosPerRow < this->getHybridModeSplit() )
   {
      /****
       * Use the scalar mode
       */
      if( globalIdx < this->getRows() )
          outVector[ globalIdx ] = this->rowVectorProduct( globalIdx, inVector );
   }
   else
   {
      /****
       * Use the vector mode
       */
      spmvCudaVectorized< InVector, OutVector, warpSize >( inVector, outVector, warpStart, warpEnd, inWarpIdx );
   }
}
#endif

template<>
class tnlCSRMatrixDeviceDependentCode< tnlHost >
{
   public:

      typedef tnlHost Device;

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const tnlCSRMatrix< Real, Device, Index >& matrix,      
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         for( Index row = 0; row < matrix.getRows(); row ++ )
            outVector[ row ] = matrix.rowVectorProduct( row, inVector );
      }

};

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename InVector,
          typename OutVector,
          int warpSize >
__global__ void tnlCSRMatrixVectorProductCudaKernel( const tnlCSRMatrix< Real, tnlCuda, Index >* matrix,
                                                     const InVector* inVector,
                                                     OutVector* outVector,
                                                     int gridIdx )
{
   typedef tnlCSRMatrix< Real, tnlCuda, Index > Matrix;
   tnlStaticAssert( Matrix::DeviceType::DeviceType == tnlCudaDevice, );
   const typename Matrix::IndexType rowIdx = ( gridIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( matrix->getCudaKernelType() == Matrix::scalar )
   {
      if( rowIdx < matrix->getRows() )
         ( *outVector )[ rowIdx ] = matrix->rowVectorProduct( rowIdx, *inVector );
   }
   if( matrix->getCudaKernelType() == Matrix::vector ||
       matrix->getCudaKernelType() == Matrix::hybrid )
   {
      matrix->template vectorProductCuda< InVector, OutVector, warpSize >
                                        ( *inVector, *outVector, gridIdx );
   }
}
#endif

template< typename Real,
          typename Index,
          typename InVector,
          typename OutVector >
void tnlCSRMatrixVectorProductCuda( const tnlCSRMatrix< Real, tnlCuda, Index >& matrix,
                                    const InVector& inVector,
                                    OutVector& outVector )
{
#ifdef HAVE_CUDA
   typedef tnlCSRMatrix< Real, tnlCuda, Index > Matrix;
   typedef typename Matrix::IndexType IndexType;
   Matrix* kernel_this = tnlCuda::passToDevice( matrix );
   InVector* kernel_inVector = tnlCuda::passToDevice( inVector );
   OutVector* kernel_outVector = tnlCuda::passToDevice( outVector );
   dim3 cudaBlockSize( 256 ), cudaGridSize( tnlCuda::getMaxGridSize() );
   const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
   const IndexType cudaGrids = roundUpDivision( cudaBlocks, tnlCuda::getMaxGridSize() );
   for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
   {
      if( gridIdx == cudaGrids - 1 )
         cudaGridSize.x = cudaBlocks % tnlCuda::getMaxGridSize();
      const int sharedMemory = cudaBlockSize.x * sizeof( Real );
      if( matrix.getCudaWarpSize() == 32 )
         tnlCSRMatrixVectorProductCudaKernel< Real, Index, InVector, OutVector, 32 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx );
      if( matrix.getCudaWarpSize() == 16 )
         tnlCSRMatrixVectorProductCudaKernel< Real, Index, InVector, OutVector, 16 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx );
      if( matrix.getCudaWarpSize() == 8 )
         tnlCSRMatrixVectorProductCudaKernel< Real, Index, InVector, OutVector, 8 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx );
      if( matrix.getCudaWarpSize() == 4 )
         tnlCSRMatrixVectorProductCudaKernel< Real, Index, InVector, OutVector, 4 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx );
      if( matrix.getCudaWarpSize() == 2 )
         tnlCSRMatrixVectorProductCudaKernel< Real, Index, InVector, OutVector, 2 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx );
      if( matrix.getCudaWarpSize() == 1 )
         tnlCSRMatrixVectorProductCudaKernel< Real, Index, InVector, OutVector, 1 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx );

   }
   tnlCuda::freeFromDevice( kernel_this );
   tnlCuda::freeFromDevice( kernel_inVector );
   tnlCuda::freeFromDevice( kernel_outVector );
   checkCudaDevice;
#endif
}


template<>
class tnlCSRMatrixDeviceDependentCode< tnlCuda >
{
   public:

      typedef tnlCuda Device;

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const tnlCSRMatrix< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         tnlCSRMatrixVectorProductCuda( matrix, inVector, outVector );
      }

};


#endif /* TNLCSRMATRIX_IMPL_H_ */
