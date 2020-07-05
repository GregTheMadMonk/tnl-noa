/***************************************************************************
                          CSR_impl.h  -  description
                             -------------------
    begin                : Dec 10, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/Legacy/CSR.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Math.h>
#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Exceptions/NotImplementedError.h>
#include <TNL/Atomic.h>
#include <vector>

#ifdef HAVE_CUSPARSE
#include <cuda.h>
#include <cusparse.h>
#endif

template<typename Index>
struct Block {
   Block(Index row, Index index = 0) noexcept {
      this->index = index;
      this->row = row;
   }

   Index index;
   Index row;
};

/* Configuration */
constexpr size_t MAX_X_DIM = 2147483647;
constexpr int ELEMENTS_PER_WARP = 1024;
//-----------------------------------------------------------------

namespace TNL {
namespace Matrices {
   namespace Legacy {

#ifdef HAVE_CUSPARSE
template< typename Real, typename Index >
class tnlCusparseCSRWrapper {};
#endif


template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
CSR< Real, Device, Index, KernelType >::CSR()
: //spmvCudaKernel( hybrid ),
  cudaWarpSize( 32 ), //Cuda::getWarpSize() )
  hybridModeSplit( 4 )
{
};

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
String CSR< Real, Device, Index, KernelType >::getSerializationType()
{
   return String( "Matrices::CSR< ") +
          TNL::getType< Real>() +
          ", [any_device], " +
          String( TNL::getType< Index >() ) +
          String( " >" );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
String CSR< Real, Device, Index, KernelType >::getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::setDimensions( const IndexType rows,
                                                const IndexType columns )
{
   Sparse< Real, Device, Index >::setDimensions( rows, columns );
   this->rowPointers.setSize( this->rows + 1 );
   this->rowPointers.setValue( 0 );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::setCompressedRowLengths( ConstCompressedRowLengthsVectorView rowLengths )
{
   TNL_ASSERT_GT( this->getRows(), 0, "cannot set row lengths of an empty matrix" );
   TNL_ASSERT_GT( this->getColumns(), 0, "cannot set row lengths of an empty matrix" );
   TNL_ASSERT_EQ( this->getRows(), rowLengths.getSize(), "wrong size of the rowLengths vector" );

   /****
    * Compute the rows pointers. The last one is
    * the end of the last row and so it says the
    * necessary length of the vectors this->values
    * and this->columnIndexes.
    */
   Containers::VectorView< IndexType, DeviceType, IndexType > rowPtrs;
   rowPtrs.bind( this->rowPointers.getData(), this->getRows() );
   rowPtrs = rowLengths;
   this->rowPointers.setElement( this->rows, 0 );
   this->rowPointers.template scan< Algorithms::ScanType::Exclusive >();
   this->maxRowLength = max( rowLengths );

   /****
    * Allocate values and column indexes
    */
   this->values.setSize( this->rowPointers.getElement( this->rows ) );
   this->columnIndexes.setSize( this->rowPointers.getElement( this->rows ) );
   this->columnIndexes.setValue( this->columns );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::getCompressedRowLengths( CompressedRowLengthsVectorView rowLengths ) const
{
   TNL_ASSERT_EQ( rowLengths.getSize(), this->getRows(), "invalid size of the rowLengths vector" );
   for( IndexType row = 0; row < this->getRows(); row++ )
      rowLengths.setElement( row, this->getRowLength( row ) );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
Index CSR< Real, Device, Index, KernelType >::getRowLength( const IndexType row ) const
{
   return this->rowPointers.getElement( row + 1 ) - this->rowPointers.getElement( row );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
Index CSR< Real, Device, Index, KernelType >::getRowLengthFast( const IndexType row ) const
{
   return this->rowPointers[ row + 1 ] - this->rowPointers[ row ];
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
Index CSR< Real, Device, Index, KernelType >::getNonZeroRowLength( const IndexType row ) const
{
    // TODO: Fix/Implement
    TNL_ASSERT( false, std::cerr << "TODO: Fix/Implement" );
    return 0;
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
Index CSR< Real, Device, Index, KernelType >::getNonZeroRowLengthFast( const IndexType row ) const
{
   ConstMatrixRow matrixRow = this->getRow( row );
   return matrixRow.getNonZeroElementsCount();
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
   template< typename Real2,
             typename Device2,
             typename Index2,
             CSRKernel KernelType2 >
void CSR< Real, Device, Index, KernelType >::setLike( const CSR< Real2, Device2, Index2, KernelType2 >& matrix )
{
   Sparse< Real, Device, Index >::setLike( matrix );
   this->rowPointers.setLike( matrix.rowPointers );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::reset()
{
   Sparse< Real, Device, Index >::reset();
   this->rowPointers.reset();
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
bool CSR< Real, Device, Index, KernelType >::setElementFast( const IndexType row,
                                                          const IndexType column,
                                                          const Real& value )
{
   return this->addElementFast( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
bool CSR< Real, Device, Index, KernelType >::setElement( const IndexType row,
                                                      const IndexType column,
                                                      const Real& value )
{
   return this->addElement( row, column, value, 0.0 );
}


template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
bool CSR< Real, Device, Index, KernelType >::addElementFast( const IndexType row,
                                                          const IndexType column,
                                                          const RealType& value,
                                                          const RealType& thisElementMultiplicator )
{
   IndexType elementPtr = this->rowPointers[ row ];
   const IndexType rowEnd = this->rowPointers[ row + 1 ];
   IndexType col = 0;
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
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
bool CSR< Real, Device, Index, KernelType >::addElement( const IndexType row,
                                                      const IndexType column,
                                                      const RealType& value,
                                                      const RealType& thisElementMultiplicator )
{
   TNL_ASSERT( row >= 0 && row < this->rows &&
               column >= 0 && column < this->columns,
               std::cerr << " row = " << row
                    << " column = " << column
                    << " this->rows = " << this->rows
                    << " this->columns = " << this->columns );

    IndexType elementPtr = this->rowPointers.getElement( row );
    const IndexType rowEnd = this->rowPointers.getElement( row + 1 );
    IndexType col = 0;
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
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
bool CSR< Real, Device, Index, KernelType > :: setRowFast( const IndexType row,
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
      //printf( "Setting element row: %d column: %d value: %f \n", row, columnIndexes[ i ], values[ i ] );
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
          typename Index,
          CSRKernel KernelType >
bool CSR< Real, Device, Index, KernelType > :: setRow( const IndexType row,
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
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
bool CSR< Real, Device, Index, KernelType > :: addRowFast( const IndexType row,
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
          typename Index,
          CSRKernel KernelType >
bool CSR< Real, Device, Index, KernelType > :: addRow( const IndexType row,
                                                    const IndexType* columns,
                                                    const RealType* values,
                                                    const IndexType numberOfElements,
                                                    const RealType& thisElementMultiplicator )
{
   return this->addRowFast( row, columns, values, numberOfElements, thisElementMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
Real CSR< Real, Device, Index, KernelType >::getElementFast( const IndexType row,
                                                          const IndexType column ) const
{
   IndexType elementPtr = this->rowPointers[ row ];
   const IndexType rowEnd = this->rowPointers[ row + 1 ];
   IndexType col = 0;
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
          typename Index,
          CSRKernel KernelType >
Real CSR< Real, Device, Index, KernelType >::getElement( const IndexType row,
                                                      const IndexType column ) const
{
   IndexType elementPtr = this->rowPointers.getElement( row );
   const IndexType rowEnd = this->rowPointers.getElement( row + 1 );
   IndexType col = 0;
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
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
void CSR< Real, Device, Index, KernelType >::getRowFast( const IndexType row,
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
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
typename CSR< Real, Device, Index, KernelType >::MatrixRow
CSR< Real, Device, Index, KernelType >::
getRow( const IndexType rowIndex )
{
   const IndexType rowOffset = this->rowPointers[ rowIndex ];
   const IndexType rowLength = this->rowPointers[ rowIndex + 1 ] - rowOffset;
   return MatrixRow( &this->columnIndexes[ rowOffset ],
                     &this->values[ rowOffset ],
                     rowLength,
                     1 );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
typename CSR< Real, Device, Index, KernelType >::ConstMatrixRow
CSR< Real, Device, Index, KernelType >::
getRow( const IndexType rowIndex ) const
{
    const IndexType rowOffset = this->rowPointers[ rowIndex ];
    const IndexType rowLength = this->rowPointers[ rowIndex + 1 ] - rowOffset;
    return ConstMatrixRow( &this->columnIndexes[ rowOffset ],
                           &this->values[ rowOffset ],
                           rowLength,
                           1 );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
   template< typename Vector >
__cuda_callable__
typename Vector::RealType CSR< Real, Device, Index, KernelType >::rowVectorProduct( const IndexType row,
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
          typename Index,
          CSRKernel KernelType >
   template< typename InVector, typename OutVector >
void CSR< Real, Device, Index, KernelType >::vectorProduct( const InVector& inVector,
                                                OutVector& outVector ) const
{
   DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
   template< typename Real2,
             typename Index2,
             CSRKernel KernelType2 >
void CSR< Real, Device, Index, KernelType >::addMatrix( const CSR< Real2, Device, Index2, KernelType2 >& matrix,
                                            const RealType& matrixMultiplicator,
                                            const RealType& thisMatrixMultiplicator )
{
   throw Exceptions::NotImplementedError( "CSR::addMatrix is not implemented." );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
   template< typename Real2,
             typename Index2,
             CSRKernel KernelType2 >
void CSR< Real, Device, Index, KernelType >::getTransposition( const CSR< Real2, Device, Index2, KernelType2 >& matrix,
                                                                      const RealType& matrixMultiplicator )
{
   throw Exceptions::NotImplementedError( "CSR::getTransposition is not implemented." );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
   template< typename Vector1, typename Vector2 >
bool CSR< Real, Device, Index, KernelType >::performSORIteration( const Vector1& b,
                                                      const IndexType row,
                                                      Vector2& x,
                                                      const RealType& omega ) const
{
   TNL_ASSERT( row >=0 && row < this->getRows(),
              std::cerr << "row = " << row
                   << " this->getRows() = " << this->getRows() << std::endl );

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
      std::cerr << "There is zero on the diagonal in " << row << "-th row of the matrix. I cannot perform SOR iteration." << std::endl;
      return false;
   }
   x[ row ] = ( 1.0 - omega ) * x[ row ] + omega / diagonalValue * ( b[ row ] - sum );
   return true;
}


// copy assignment
template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
CSR< Real, Device, Index, KernelType >&
CSR< Real, Device, Index, KernelType >::operator=( const CSR& matrix )
{
   this->setLike( matrix );
   this->values = matrix.values;
   this->columnIndexes = matrix.columnIndexes;
   this->rowPointers = matrix.rowPointers;
   return *this;
}

// cross-device copy assignment
template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
   template< typename Real2, typename Device2, typename Index2, CSRKernel KernelType2, typename >
CSR< Real, Device, Index, KernelType >&
CSR< Real, Device, Index, KernelType >::operator=( const CSR< Real2, Device2, Index2, KernelType2 >& matrix )
{
   this->setLike( matrix );
   this->values = matrix.values;
   this->columnIndexes = matrix.columnIndexes;
   this->rowPointers = matrix.rowPointers;
   return *this;
}


template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::save( File& file ) const
{
   Sparse< Real, Device, Index >::save( file );
   file << this->rowPointers;
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::load( File& file )
{
   Sparse< Real, Device, Index >::load( file );
   file >> this->rowPointers;
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::print( std::ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      IndexType elementPtr = this->rowPointers.getElement( row );
      const IndexType rowEnd = this->rowPointers.getElement( row + 1 );
      IndexType column;
      while( elementPtr < rowEnd &&
             ( column = this->columnIndexes.getElement( elementPtr ) ) < this->columns &&
             column != this->getPaddingIndex() )
         str << " Col:" << column << "->" << this->values.getElement( elementPtr++ ) << "\t";
      str << std::endl;
   }
}

/*template< typename Real,
          typename Device,
          typename Index >
void CSR< Real, Device, Index, KernelType >::setCudaKernelType( const SPMVCudaKernel kernel )
{
   this->spmvCudaKernel = kernel;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename CSR< Real, Device, Index, KernelType >::SPMVCudaKernel CSR< Real, Device, Index, KernelType >::getCudaKernelType() const
{
   return this->spmvCudaKernel;
}*/

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::setCudaWarpSize( const int warpSize )
{
   this->cudaWarpSize = warpSize;
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
int CSR< Real, Device, Index, KernelType >::getCudaWarpSize() const
{
   return this->cudaWarpSize;
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::setHybridModeSplit( const IndexType hybridModeSplit )
{
   this->hybridModeSplit = hybridModeSplit;
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
Index CSR< Real, Device, Index, KernelType >::getHybridModeSplit() const
{
   return this->hybridModeSplit;
}

#ifdef HAVE_CUDA

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
   template< typename InVector,
             typename OutVector,
             int warpSize >
__device__
void CSR< Real, Device, Index, KernelType >::spmvCudaVectorized( const InVector& inVector,
                                                              OutVector& outVector,
                                                              const IndexType gridIdx ) const
{
   IndexType globalIdx = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   const IndexType warpStart = warpSize * ( globalIdx / warpSize );
   const IndexType warpEnd = min( warpStart + warpSize, this->getRows() );
   const IndexType inWarpIdx = globalIdx % warpSize;

   volatile Real* aux = Cuda::getSharedMemory< Real >();
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
      if( inWarpIdx == 0 )
         outVector[ row ] = aux[ threadIdx.x ];
   }
}

template< typename Real,
          typename Index,
          int warpSize >
__global__
void SpMVCSRAdaptive( const Real *inVector,
                      Real *outVector,
                      const Index* rowPointers,
                      const Index* columnIndexes,
                      const Real* values,
                      Block<Index> *blocks,
                      Index blocks_size,
                      Index getColumns,
                      Index gridID,
                      const Index sharedPerWarp) {
   __shared__ Real shared_res[49152/sizeof(Real)];
   const Index index = (gridID * MAX_X_DIM) + (blockIdx.x * blockDim.x) + threadIdx.x;
   const Index blockIdx = index / warpSize;
   if (blockIdx >= blocks_size)
      return;

   Real result = 0.0;
   const Index laneID = index % warpSize;
   const Index minRow = blocks[blockIdx].row;
   const Index maxRow = blocks[blockIdx + 1].row;
   const Index minID = rowPointers[minRow];
   Index maxID = rowPointers[maxRow];
   Index i, to, column, offset;
   Index elements = maxID - minID;
   /* rows per block more than 1 */
   if (elements == 0 || elements > ELEMENTS_PER_WARP) {
      /////////////////////////////////////* CSR VECTOR L */////////////
      const Index warpInRow = blocks[blockIdx].index;
      if (elements == 0) maxID = rowPointers[minRow + 1];
      
      offset = warpInRow * ELEMENTS_PER_WARP;
      to = minID + (warpInRow + 1) * ELEMENTS_PER_WARP;
      if (to > maxID) to = maxID;
      
      for (i = minID + offset + laneID; i < to; i += warpSize) {
         column = columnIndexes[i];
         if (column >= getColumns)
            break;

         result += values[i] * inVector[column];
      }

      /* Parallel reduction */
      result += __shfl_down_sync((unsigned)(warpSize - 1), result, 16);
      result += __shfl_down_sync((unsigned)(warpSize - 1), result, 8);
      result += __shfl_down_sync((unsigned)(warpSize - 1), result, 4);
      result += __shfl_down_sync((unsigned)(warpSize - 1), result, 2);
      result += __shfl_down_sync((unsigned)(warpSize - 1), result, 1);
      if (laneID == 0) atomicAdd(&outVector[minRow], result);
   } else if (elements <= sharedPerWarp) {
      /////////////////////////////////////* CSR STREAM *//////////////
      /* Copy and calculate elements from global to shared memory, coalesced */
      offset = threadIdx.x / warpSize * sharedPerWarp;
      Index elementID = laneID + minID;
      Index sharedID = laneID + offset; // index for shared memory
      for (; elementID < maxID; elementID += warpSize, sharedID += warpSize) {
         column = columnIndexes[elementID];
         if (column >= getColumns)
            continue; // can't be break
         shared_res[sharedID] = values[elementID] * inVector[column];
      }

      /* Calculate result */
      for (Index row = minRow + laneID; row < maxRow; row += warpSize) {
         to = rowPointers[row + 1] - minID + offset; // end of preprocessed data
         result = 0;
         /* Scalar reduction */
         for (sharedID = rowPointers[row] - minID + offset; sharedID < to; ++sharedID)
            result += shared_res[sharedID];

         outVector[row] = result; // Write result
      }
   } else {
      /////////////////////////////////////* CSR VECTOR *//////////////
      for (i = minID + laneID; i < maxID; i += warpSize) {
         column = columnIndexes[i];
         if (column >= getColumns)
            break;

         result += values[i] * inVector[column];
      }
      /* Parallel reduction */
      result += __shfl_down_sync((unsigned)(warpSize - 1), result, 16);
      result += __shfl_down_sync((unsigned)(warpSize - 1), result, 8);
      result += __shfl_down_sync((unsigned)(warpSize - 1), result, 4);
      result += __shfl_down_sync((unsigned)(warpSize - 1), result, 2);
      result += __shfl_down_sync((unsigned)(warpSize - 1), result, 1);
      if (laneID == 0) outVector[minRow] = result; // Write result
   }
}

template< typename Real,
          typename Index,
          int warpSize >
__global__
void SpMVCSRScalar( const Real *inVector,
                    Real* outVector,
                    const Index* rowPointers,
                    const Index* columnIndexes,
                    const Real* values,
                    const Index rows,
                    const Index getColumns,
                    const Index gridID) {
   const Index index = (gridID * MAX_X_DIM) + (blockIdx.x * blockDim.x) + threadIdx.x;
   if (index >= rows)
      return;

   Real result = 0.0;
   const Index endID = rowPointers[index + 1];

   for (Index i = rowPointers[index]; i < endID; ++i) {
      const Index column = columnIndexes[i];
      if (column >= getColumns)
         break;

      result += values[i] * inVector[column];
   }

   outVector[index] = result;
}

template< typename Real,
          typename Index,
          int warpSize >
__global__
void SpMVCSRMultiVector( const Real *inVector,
                         Real* outVector,
                         const Index* rowPointers,
                         const Index* columnIndexes,
                         const Real* values,
                         const Index rows,
                         const Index getColumns,
                         const Index perWarp,
                         const Index offset,
                         const Index gridID)
{
   const Index index = (gridID * MAX_X_DIM) + (blockIdx.x * blockDim.x) + threadIdx.x;
   const Index rowID = index / offset;
   if (rowID >= rows)
      return;

   const Index inRowID = index % offset;

   Real result = 0.0;
   Index endID = rowPointers[rowID + 1];

   /* Calculate result */
   for (Index i = rowPointers[rowID] + inRowID; i < endID; i += offset) {
      Index column = columnIndexes[i];
      if (column >= getColumns)
         break;

      result += values[i] * inVector[column];
   }

   /* Reduction */
   result += __shfl_down_sync((unsigned)(warpSize - 1), result, 16);
   result += __shfl_down_sync((unsigned)(warpSize - 1), result, 8);
   result += __shfl_down_sync((unsigned)(warpSize - 1), result, 4);
   result += __shfl_down_sync((unsigned)(warpSize - 1), result, 2);
   result += __shfl_down_sync((unsigned)(warpSize - 1), result, 1);
   /* Write result */
   if (index % warpSize == 0) atomicAdd(&outVector[rowID], result);
}

template< typename Real,
          typename Index,
          int warpSize >
__global__
void SpMVCSRVector( const Real *inVector,
                    Real* outVector,
                    const Index* rowPointers,
                    const Index* columnIndexes,
                    const Real* values,
                    const Index rows,
                    const Index getColumns,
                    const Index gridID)
{
   const Index index = (gridID * MAX_X_DIM) + (blockIdx.x * blockDim.x) + threadIdx.x;
   const Index warpID = index / warpSize;
   if (warpID >= rows)
      return;

   const Index laneID = index % warpSize;
   Real result = 0.0;
   Index endID = rowPointers[warpID + 1];

   /* Calculate result */
   for (Index i = rowPointers[warpID] + laneID; i < endID; i += warpSize) {
      Index column = columnIndexes[i];
      if (column >= getColumns)
         break;

      result += values[i] * inVector[column];
   }

   /* Reduction */
   result += __shfl_down_sync((unsigned)(warpSize - 1), result, 16);
   result += __shfl_down_sync((unsigned)(warpSize - 1), result, 8);
   result += __shfl_down_sync((unsigned)(warpSize - 1), result, 4);
   result += __shfl_down_sync((unsigned)(warpSize - 1), result, 2);
   result += __shfl_down_sync((unsigned)(warpSize - 1), result, 1);
   /* Write result */
   if (laneID == 0) outVector[warpID] = result;
}

template< typename Real,
          typename Index,
          int warpSize >
__global__
void SpMVCSRLight( const Real *inVector,
                   Real* outVector,
                   const Index* rowPointers,
                   const Index* columnIndexes,
                   const Real* values,
                   const Index rows,
                   const Index getColumns,
                   const Index groupSize,
                   const Index gridID,
                   unsigned *rowCnt) {
   const Index index = (gridID * MAX_X_DIM) + (blockIdx.x * blockDim.x) + threadIdx.x;
   const Index laneID = index % warpSize;
   const Index groupID = laneID / groupSize;
   const Index inGroupID = laneID % groupSize;
   Index row, minID, maxID, i;

   while (true) {

      /* Get row number */
      if (inGroupID == 0) row = atomicAdd(rowCnt, 1);

      /* Propagate row number in group */
      row = __shfl_sync((unsigned)(warpSize - 1), row, groupID * groupSize);
      if (row >= rows)
         return;

      minID = rowPointers[row];
      maxID = rowPointers[row + 1];

      Real result = 0.0;
      for (i = minID + inGroupID; i < maxID; i += groupSize) {
         const Index column = columnIndexes[i];
         if (column >= getColumns)
            break;

         result += values[i] * inVector[column];
      }

      /* Parallel reduction */
      for (i = groupSize / 2; i > 0; i /= 2)
         result += __shfl_down_sync((unsigned)(warpSize - 1), result, i);
      /* Write result */
      if (inGroupID == 0)
         outVector[row] = result;
   }
}

template< typename Real,
          typename Index,
          int warpSize >
__global__
void SpMVCSRLightWithoutAtomic( const Real *inVector,
                                Real* outVector,
                                const Index* rowPointers,
                                const Index* columnIndexes,
                                const Real* values,
                                const Index rows,
                                const Index getColumns,
                                const Index groupSize,
                                const Index gridID) {
   const Index index = (gridID * MAX_X_DIM) + (blockIdx.x * blockDim.x) + threadIdx.x;
   const Index row = index / groupSize;

   if (row >= rows)
      return;

   const Index inGroupID = index % groupSize;
   const Index minID = rowPointers[row];
   const Index maxID = rowPointers[row + 1];

   Real result = 0.0;
   for (Index i = minID + inGroupID; i < maxID; i += groupSize) {
      const Index column = columnIndexes[i];
      if (column >= getColumns)
         break;

      result += values[i] * inVector[column];
   }

   /* Parallel reduction */
   for (i = groupSize / 2; i > 0; i /= 2)
      result += __shfl_down_sync((unsigned)(warpSize - 1), result, i);

   /* Write result */
   if (inGroupID == 0) outVector[row] = result;
}

template< typename Real,
          typename Index,
          int warpSize >
void SpMVCSRScalarPrepare( const Real *inVector,
                           Real* outVector,
                           const Index* rowPointers,
                           const Index* columnIndexes,
                           const Real* values,
                           const Index rows,
                           const Index getColumns) {
   const Index threads = 1024; // block size
   size_t neededThreads = rows;
   Index blocks;
   /* Execute kernels on device */
   for (Index grid = 0; neededThreads != 0; ++grid) {
      if (MAX_X_DIM * threads >= neededThreads) {
         blocks = roundUpDivision(neededThreads, threads);
         neededThreads = 0;
      } else {
         blocks = MAX_X_DIM;
         neededThreads -= MAX_X_DIM * threads;
      }

      SpMVCSRScalar<Real, Index, warpSize><<<blocks, threads>>>(
               inVector,
               outVector,
               rowPointers,
               columnIndexes,
               values,
               rows,
               getColumns,
               grid
      );
   }
}

template< typename Real,
          typename Index,
          int warpSize >
void SpMVCSRVectorPrepare( const Real *inVector,
                           Real* outVector,
                           const Index* rowPointers,
                           const Index* columnIndexes,
                           const Real* values,
                           const Index rows,
                           const Index getColumns) {
   const Index threads = 1024; // block size
   size_t neededThreads = rows * warpSize;
   Index blocks;
   /* Execute kernels on device */
   for (Index grid = 0; neededThreads != 0; ++grid) {
      if (MAX_X_DIM * threads >= neededThreads) {
         blocks = roundUpDivision(neededThreads, threads);
         neededThreads = 0;
      } else {
         blocks = MAX_X_DIM;
         neededThreads -= MAX_X_DIM * threads;
      }

      SpMVCSRVector<Real, Index, warpSize><<<blocks, threads>>>(
               inVector,
               outVector,
               rowPointers,
               columnIndexes,
               values,
               rows,
               getColumns,
               grid
      );
   }
}

template< typename Real,
          typename Index,
          int warpSize >
void SpMVCSRLightPrepare( const Real *inVector,
                          Real* outVector,
                          const Index* rowPointers,
                          const Index* columnIndexes,
                          const Real* values,
                          const Index valuesSize,
                          const Index rows,
                          const Index getColumns) {
   const Index threads = 1024; // block size
   Index blocks, groupSize;
   /* Copy rowCnt to GPU */
   unsigned rowCnt = 0;
   unsigned *kernelRowCnt;
   cudaMalloc((void **)&kernelRowCnt, sizeof(*kernelRowCnt));
   cudaMemcpy(kernelRowCnt, &rowCnt, sizeof(*kernelRowCnt), cudaMemcpyHostToDevice);

   
   const Index nnz = roundUpDivision(valuesSize, rows); // non zeroes per row
   if (nnz <= 2)
      groupSize = 2;
   else if (nnz <= 4)
      groupSize = 4;
   else if (nnz <= 8)
      groupSize = 8;
   else if (nnz <= 16)
      groupSize = 16;
   else
      groupSize = 32;

   size_t neededThreads = groupSize * rows;
   /* Execute kernels on device */
   for (Index grid = 0; neededThreads != 0; ++grid) {
      if (MAX_X_DIM * threads >= neededThreads) {
         blocks = roundUpDivision(neededThreads, threads);
         neededThreads = 0;
      } else {
         blocks = MAX_X_DIM;
         neededThreads -= MAX_X_DIM * threads;
      }

      if (groupSize == 32) { // CSR SpMV Light with groupsize = 32 is CSR Vector
         SpMVCSRVector<Real, Index, warpSize><<<blocks, threads>>>(
               inVector,
               outVector,
               rowPointers,
               columnIndexes,
               values,
               rows,
               getColumns,
               grid
         );
      } else {
         SpMVCSRLight<Real, Index, warpSize><<<blocks, threads>>>(
               inVector,
               outVector,
               rowPointers,
               columnIndexes,
               values,
               rows,
               getColumns,
               groupSize,
               grid,
               kernelRowCnt
         );
      }
   }

   cudaFree(kernelRowCnt);
}

template< typename Real,
          typename Index,
          int warpSize >
void SpMVCSRLightWithoutAtomicPrepare( const Real *inVector,
                                       Real* outVector,
                                       const Index* rowPointers,
                                       const Index* columnIndexes,
                                       const Real* values,
                                       const Index valuesSize,
                                       const Index rows,
                                       const Index getColumns) {
   const Index threads = 1024; // block size
   size_t neededThreads = rows * warpSize;
   Index blocks, groupSize;
   
   const Index nnz = roundUpDivision(valuesSize, rows); // non zeroes per row
   if (nnz <= 2)
      groupSize = 2;
   else if (nnz <= 4)
      groupSize = 4;
   else if (nnz <= 8)
      groupSize = 8;
   else if (nnz <= 16)
      groupSize = 16;
   else
      groupSize = 32;

   neededThreads = groupSize * rows;
   /* Execute kernels on device */
   for (Index grid = 0; neededThreads != 0; ++grid) {
      if (MAX_X_DIM * threads >= neededThreads) {
         blocks = roundUpDivision(neededThreads, threads);
         neededThreads = 0;
      } else {
         blocks = MAX_X_DIM;
         neededThreads -= MAX_X_DIM * threads;
      }

      if (groupSize == 32) { // CSR SpMV Light with groupsize = 32 is CSR Vector
         SpMVCSRVector<Real, Index, warpSize><<<blocks, threads>>>(
               inVector,
               outVector,
               rowPointers,
               columnIndexes,
               values,
               rows,
               getColumns,
               grid
         );
      } else {
         SpMVCSRLightWithoutAtomic<Real, Index, warpSize><<<blocks, threads>>>(
                  inVector,
                  outVector,
                  rowPointers,
                  columnIndexes,
                  values,
                  rows,
                  getColumns,
                  groupSize,
                  grid
         );
      }
   }
}

template< typename Real,
          typename Index,
          int warpSize >
void SpMVCSRMultiVectorPrepare( const Real *inVector,
                                Real* outVector,
                                const Index* rowPointers,
                                const Index* columnIndexes,
                                const Real* values,
                                const Index valuesSize,
                                const Index rows,
                                const Index getColumns) {
   /* Configuration */
   //----------------------------------------------------------------------------------
   const Index threads = 1024; // block size
   Index blocks;

   const Index nnz = roundUpDivision(valuesSize, rows); // non zeroes per row
   const size_t neededWarps = roundUpDivision(nnz, ELEMENTS_PER_WARP); // warps per row
   const Index offset = neededWarps * ELEMENTS_PER_WARP;
   size_t neededThreads = offset * rows;
   /* Execute kernels on device */
   for (Index grid = 0; neededThreads != 0; ++grid) {
      if (MAX_X_DIM * threads >= neededThreads) {
         blocks = roundUpDivision(neededThreads, threads);
         neededThreads = 0;
      } else {
         blocks = MAX_X_DIM;
         neededThreads -= MAX_X_DIM * threads;
      }

      if (neededWarps == 1) { // one warp per row -> execute CSR Vector
         SpMVCSRVector<Real, Index, warpSize><<<blocks, threads>>>(
               inVector,
               outVector,
               rowPointers,
               columnIndexes,
               values,
               rows,
               getColumns,
               grid
         );
      } else {
         SpMVCSRMultiVector<Real, Index, warpSize><<<blocks, threads>>>(
                  inVector,
                  outVector,
                  rowPointers,
                  columnIndexes,
                  values,
                  rows,
                  getColumns,
                  ELEMENTS_PER_WARP,
                  offset,
                  grid
         );
      }
   }
}

/* Find limit of block */
template< typename Real,
          typename Index,
          typename Device,
          CSRKernel KernelType>
Index findLimit(const Index start, const Index max,
               const CSR< Real, Device, Index, KernelType >& matrix,
               const Index size) {
   Index sum = 0;
   for (Index current = start; current < size - 1; ++current) {
      Index elements = matrix.getRowPointers().getElement(current + 1) -
                       matrix.getRowPointers().getElement(current);
      sum += elements;
      if (sum > max) {
         if (current - start > 1) // extra row
            return current;
         else                     // one long row
            return current + 1;
      }
   }

   return size - 1; // return last row pointer
}

template< typename Real,
          typename Index,
          typename Device,
          CSRKernel KernelType,
          int warpSize >
void SpMVCSRAdaptivePrepare( const Real *inVector,
                             Real* outVector,
                             const CSR< Real, Device, Index, KernelType >& matrix,
                             const Index* rowPointers,
                             const Index* columnIndexes,
                             const Real* values,
                             const Index valuesSize,
                             const Index rows,
                             const Index getColumns) {
   /* Configuration ---------------------------------------------------*/
   /* Execute 1024 threads per block for float, (12 elements per thread) for 48KB cache
              512  threads per block for double (12 elements per thread) */
   constexpr size_t THREADS_PER_BLOCK = sizeof(Real) == 4 ? 1024 : 512;
   constexpr Index WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;
   constexpr Index SHARED = 49152/sizeof(Real); 
   constexpr Index SHARED_PER_WARP = SHARED / WARPS_PER_BLOCK;
   //--------------------------------------------------------------------
   Index blocks;
   const Index threads = THREADS_PER_BLOCK;

   /* Fill blocks */
   std::vector<Block<Index>> inBlock;
   Index start = 0;
   inBlock.emplace_back(0); // push start
   while (start != rows - 1) {
      Index startNext = findLimit(start, SHARED_PER_WARP, matrix, rows);
      Index sum = matrix.getRowPointers().getElement(startNext) -
            matrix.getRowPointers().getElement(start);
      
      /* block start is already inserted, +1 */
      Index parts = roundUpDivision(sum, ELEMENTS_PER_WARP);
      for (Index warpIndex = 1; warpIndex < parts; ++warpIndex)
         inBlock.emplace_back(start, warpIndex);

      inBlock.emplace_back(startNext);
      start = startNext;
   }

   /* blocks to GPU */
   Block<Index> *blocksAdaptive = nullptr;
   cudaMalloc((void **)&blocksAdaptive, sizeof(*blocksAdaptive) * inBlock.size());
   cudaMemcpy(blocksAdaptive, inBlock.data(), inBlock.size() * sizeof(*blocksAdaptive), cudaMemcpyHostToDevice);

   size_t neededThreads = inBlock.size() * 32; // one warp per block
   /* Execute kernels on device */
   for (Index grid = 0; neededThreads != 0; ++grid) {
      if (MAX_X_DIM * threads >= neededThreads) {
         blocks = roundUpDivision(neededThreads, threads);
         neededThreads = 0;
      } else {
         blocks = MAX_X_DIM;
         neededThreads -= MAX_X_DIM * threads;
      }

      SpMVCSRAdaptive<Real, Index, warpSize><<<blocks, threads>>>(
               inVector,
               outVector,
               rowPointers,
               columnIndexes,
               values,
               blocksAdaptive,
               inBlock.size() - 1, // last block shouldn't be used
               getColumns,
               grid,
               SHARED_PER_WARP
      );
   }

   cudaFree(blocksAdaptive);
}

#endif

template<>
class CSRDeviceDependentCode< Devices::Host >
{
   public:

      typedef Devices::Host Device;

      template< typename Real,
                typename Index,
                CSRKernel KernelType,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const CSR< Real, Device, Index, KernelType >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         const Index rows = matrix.getRows();
         const CSR< Real, Device, Index, KernelType >* matrixPtr = &matrix;
         const InVector* inVectorPtr = &inVector;
         OutVector* outVectorPtr = &outVector;
#ifdef HAVE_OPENMP
#pragma omp parallel for firstprivate( matrixPtr, inVectorPtr, outVectorPtr ), schedule(dynamic,100), if( Devices::Host::isOMPEnabled() )
#endif
         for( Index row = 0; row < rows; row ++ )
            ( *outVectorPtr )[ row ] = matrixPtr->rowVectorProduct( row, *inVectorPtr );
      }

};


#ifdef HAVE_CUSPARSE
template<>
class tnlCusparseCSRWrapper< float, int >
{
   public:

      typedef float Real;
      typedef int Index;

      static void vectorProduct( const Index rows,
                                 const Index columns,
                                 const Index nnz,
                                 const Real* values,
                                 const Index* columnIndexes,
                                 const Index* rowPointers,
                                 const Real* x,
                                 Real* y )
      {
#if CUDART_VERSION >= 11000
         throw std::runtime_error("cusparseScsrmv was removed in CUDA 11.");
#else
         cusparseHandle_t   cusparseHandle;
         cusparseMatDescr_t cusparseMatDescr;
         cusparseCreate( &cusparseHandle );
         cusparseCreateMatDescr( &cusparseMatDescr );
         cusparseSetMatType( cusparseMatDescr, CUSPARSE_MATRIX_TYPE_GENERAL );
         cusparseSetMatIndexBase( cusparseMatDescr, CUSPARSE_INDEX_BASE_ZERO );
         Real alpha( 1.0 ), beta( 0.0 );
         cusparseScsrmv( cusparseHandle,
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         rows,
                         columns,
                         nnz,
                         &alpha,
                         cusparseMatDescr,
                         values,
                         rowPointers,
                         columnIndexes,
                         x,
                         &beta,
                         y );
#endif
      };
};

template<>
class tnlCusparseCSRWrapper< double, int >
{
   public:

      typedef double Real;
      typedef int Index;

      static void vectorProduct( const Index rows,
                                 const Index columns,
                                 const Index nnz,
                                 const Real* values,
                                 const Index* columnIndexes,
                                 const Index* rowPointers,
                                 const Real* x,
                                 Real* y )
      {
#if CUDART_VERSION >= 11000
         throw std::runtime_error("cusparseDcsrmv was removed in CUDA 11.");
#else
         cusparseHandle_t   cusparseHandle;
         cusparseMatDescr_t cusparseMatDescr;
         cusparseCreate( &cusparseHandle );
         cusparseCreateMatDescr( &cusparseMatDescr );
         cusparseSetMatType( cusparseMatDescr, CUSPARSE_MATRIX_TYPE_GENERAL );
         cusparseSetMatIndexBase( cusparseMatDescr, CUSPARSE_INDEX_BASE_ZERO );
         Real alpha( 1.0 ), beta( 0.0 );
         cusparseDcsrmv( cusparseHandle,
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         rows,
                         columns,
                         nnz,
                         &alpha,
                         cusparseMatDescr,
                         values,
                         rowPointers,
                         columnIndexes,
                         x,
                         &beta,
                         y );
#endif
      };
};

#endif

template<>
class CSRDeviceDependentCode< Devices::Cuda >
{
   public:

      typedef Devices::Cuda Device;

      template< typename Real,
                typename Index,
                CSRKernel KernelType,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const CSR< Real, Device, Index, KernelType >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
#ifdef HAVE_CUDA
#ifdef HAVE_CUSPARSE
         tnlCusparseCSRWrapper< Real, Index >::vectorProduct( matrix.getRows(),
                                                              matrix.getColumns(),
                                                              matrix.values.getSize(),
                                                              matrix.values.getData(),
                                                              matrix.columnIndexes.getData(),
                                                              matrix.rowPointers.getData(),
                                                              inVector.getData(),
                                                              outVector.getData() );
#else
         // switch(KernelType)
         // {
         //    case CSRScalar:
               // SpMVCSRScalarPrepare<Real, Index, 32>(
               //    inVector.getData(),
               //    outVector.getData(),
               //    matrix.getRowPointers().getData(),
               //    matrix.getColumnIndexes().getData(),
               //    matrix.getValues().getData(),
               //    matrix.getRowPointers().getSize() - 1,
               //    matrix.getColumns()
               // );
         //       break;
         //    case CSRVector:
               // SpMVCSRVectorPrepare<Real, Index, 32>(
               //    inVector.getData(),
               //    outVector.getData(),
               //    matrix.getRowPointers().getData(),
               //    matrix.getColumnIndexes().getData(),
               //    matrix.getValues().getData(),
               //    matrix.getRowPointers().getSize() - 1,
               //    matrix.getColumns()
               // );
         //       break;
         //    case CSRLight:
               // SpMVCSRLightPrepare<Real, Index, 32>(
               //    inVector.getData(),
               //    outVector.getData(),
               //    matrix.getRowPointers().getData(),
               //    matrix.getColumnIndexes().getData(),
               //    matrix.getValues().getData(),
               //    matrix.getValues().getSize(),
               //    matrix.getRowPointers().getSize() - 1,
               //    matrix.getColumns()
               // );
         //       break;
         //    case CSRAdaptive:
               SpMVCSRAdaptivePrepare<Real, Index, Device, KernelType, 32>(
                  inVector.getData(),
                  outVector.getData(),
                  matrix,
                  matrix.getRowPointers().getData(),
                  matrix.getColumnIndexes().getData(),
                  matrix.getValues().getData(),
                  matrix.getValues().getSize(),
                  matrix.getRowPointers().getSize(),
                  matrix.getColumns()
               );
         //       break;
         //    case CSRMultiVector:
               // SpMVCSRMultiVectorPrepare<Real, Index, 32>(
               //    inVector.getData(),
               //    outVector.getData(),
               //    matrix.getRowPointers().getData(),
               //    matrix.getColumnIndexes().getData(),
               //    matrix.getValues().getData(),
               //    matrix.getValues().getSize(),
               //    matrix.getRowPointers().getSize() - 1,
               //    matrix.getColumns()
               // );
         //       break;
         //    case CSRLightWithoutAtomic:
               // SpMVCSRLightWithoutAtomicPrepare<Real, Index, 32>(
               //    inVector.getData(),
               //    outVector.getData(),
               //    matrix.getRowPointers().getData(),
               //    matrix.getColumnIndexes().getData(),
               //    matrix.getValues().getData(),
               //    matrix.getValues().getSize(),
               //    matrix.getRowPointers().getSize() - 1,
               //    matrix.getColumns()
               // );
         //       break;
         // }
#endif /* HAVE_CUDA */
#endif
      }
};

} //namespace Legacy
} // namespace Matrices
} // namespace TNL
