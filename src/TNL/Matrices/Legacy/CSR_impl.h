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
#include <vector>

#ifdef HAVE_CUSPARSE
#include <cuda.h>
#include <cusparse.h>
#endif

/* CONFIGURATION */
constexpr size_t WARP_SIZE = 32;
constexpr size_t THREADS_PER_BLOCK = 1024;
constexpr size_t WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

/* CSR DYNAMIC VECTOR */
constexpr size_t MAX_PER_WARP = 2048; // max elements per warp to start CSR Vector Dynamic
constexpr size_t ELEMENTS_PER_WARP = 1024; // how many elements should process new warp

/* CSR Light SPMV */
constexpr size_t THREADS_PER_ROW = 4; // how many elements should process new warp
//-------------------------------------

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
void CSR< Real, Device, Index, KernelType >::spmvCudaLightSpmv( const InVector& inVector,
                                                      OutVector& outVector,
                                                      int gridIdx) const
{
   const IndexType index = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   const IndexType THREADS_PER_ROW   = 4;
   const IndexType laneID      = index % warpSize;
   const IndexType groupID     = laneID / THREADS_PER_ROW;
   const IndexType inGroupID   = laneID % THREADS_PER_ROW;

   IndexType row, minID, column, maxID, idxMtx;
   __shared__ unsigned rowCnt;

   if (index == 0) rowCnt = 0;  // Init shared variable
   __syncthreads();

   while (true) {

      /* Get row number */
      if (inGroupID == 0) row = atomicAdd(&rowCnt, 1);

      /* Propagate row number in group */
      row = __shfl_sync((unsigned)(warpSize - 1), row, groupID * THREADS_PER_ROW);

      if (row >= this->rowPointers.getSize() - 1)
         return;

      minID = this->rowPointers[row];
      maxID = this->rowPointers[row + 1];

      Real result = 0.0;

      idxMtx = minID + inGroupID;
      while (idxMtx < maxID) {
         column = this->columnIndexes[idxMtx];
         if (column >= this->getColumns())
            break;

         result += this->values[idxMtx] * inVector[column];
         idxMtx += THREADS_PER_ROW;
      }

      /* Parallel reduction */
      for (int i = THREADS_PER_ROW / 2; i > 0; i /= 2)
         result += __shfl_down_sync((unsigned)(warpSize - 1), result, i);
      /* Write result */
      if (inGroupID == 0) {
         outVector[row] = result;
      }
   }
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
   template< typename InVector,
             typename OutVector,
             int warpSize >
__device__
void CSR< Real, Device, Index, KernelType >::spmvCSRAdaptive( const InVector& inVector,
                                                      OutVector& outVector,
                                                      int gridIdx,
                                                      int *blocks,
                                                      size_t blocks_size) const
{
   /* Configuration ---------------------------------------------------*/
   constexpr size_t SHARED = 49152/sizeof(Real);
   constexpr size_t SHARED_PER_WARP = SHARED / warpSize;
   constexpr size_t MAX_PER_WARP = 65536;
   //constexpr size_t ELEMENTS_PER_WARP = 1024;
   //constexpr size_t THREADS_PER_BLOCK = 1024;
   //constexpr size_t WARPS_PER_BLOCK = THREADS_PER_BLOCK / warpSize;
   //--------------------------------------------------------------------
   const size_t index = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   const size_t laneID = index % warpSize;
   size_t blockIdx = index / warpSize;
   __shared__ Real shared_res[SHARED];
   Real result = 0.0;
   if (blockIdx >= blocks_size - 1)
      return;
   const size_t minRow = blocks[blockIdx];
   const size_t maxRow = blocks[blockIdx + 1];
   const size_t minID = this->rowPointers[minRow];
   const size_t maxID = this->rowPointers[maxRow];
   const size_t elements = maxID - minID;
   /* rows per block more than 1 */
   if ((maxRow - minRow) > 1) {
      /////////////////////////////////////* CSR STREAM *//////////////
      /* Copy and calculate elements from global to shared memory, coalesced */
      const size_t offset = threadIdx.x / warpSize * SHARED_PER_WARP;
      for (size_t i = laneID; i < elements; i += warpSize) {
         const size_t elementIdx = i + minID;
         const size_t column = this->columnIndexes[elementIdx];
         if (column >= this->getColumns())
            continue;
         shared_res[i + offset] = this->values[elementIdx] * inVector[column];
      }

      const size_t row = minRow + laneID;
      if (row >= maxRow)
         return;
      /* Calculate result */
      const size_t to = this->rowPointers[row + 1] - minID;
      for (size_t i = this->rowPointers[row] - minID; i < to; ++i) {
         result += shared_res[i + offset];
      }
      outVector[row] = result; // Write result
   } else {
      /////////////////////////////////////* CSR VECTOR *//////////////
      for (size_t i = minID + laneID; i < maxID; i += warpSize) {
         size_t column = this->columnIndexes[i];
         if (column >= this->getColumns())
            break;

         result += this->values[i] * inVector[column];
      }
      /* Reduction */
      result += __shfl_down_sync((unsigned)(warpSize - 1), result, 16);
      result += __shfl_down_sync((unsigned)(warpSize - 1), result, 8);
      result += __shfl_down_sync((unsigned)(warpSize - 1), result, 4);
      result += __shfl_down_sync((unsigned)(warpSize - 1), result, 2);
      result += __shfl_down_sync((unsigned)(warpSize - 1), result, 1);
      if (laneID == 0) outVector[minRow] = result; // Write result
   } else {
      /////////////////////////////////////* CSR VECTOR LONG *//////////////
      //const size_t warps = (elements - ELEMENTS_PER_WARP) / ELEMENTS_PER_WARP + 1;
      //const size_t blocks = warps <= WARPS_PER_BLOCK ? 1 : warps / WARPS_PER_BLOCK + 1;
      //const size_t threads_per_block = blocks == 1 ? warps * warpSize : WARPS_PER_BLOCK * warpSize;
      // spmvCSRVectorHelper<InVector, warpSize> <<<blocks, threads_per_block>>>(
      //             inVector,
      //             &outVector[minRow],
      //             (size_t)(minID + ELEMENTS_PER_WARP),
      //             (size_t)maxID,
      //             (size_t)ELEMENTS_PER_WARP
      // );
   }
}

template< typename Real,
          typename Index,
          typename InVector,
          int warpSize >
__global__
void spmvCSRVectorHelper(const InVector& inVector,
                         const Index* columnIndexes,
                         const Real *values,
                         const Index getColumns,
                         Real *out,
                         size_t from,
                         size_t to,
                         size_t perWarp)
{
   const size_t index  = blockIdx.x * blockDim.x + threadIdx.x;
   const size_t warpID = index / warpSize;
   const size_t laneID = index % warpSize;
   const size_t minID  = from + warpID * perWarp;
   size_t maxID  = from + (warpID + 1) * perWarp;
   if (minID >= to)  return;
   if (maxID >= to ) maxID = to;
   
   Real result = 0;
   for (size_t i = minID + laneID; i < maxID; i += warpSize) {
      const size_t column = columnIndexes[i];
      if (column >= getColumns)
         break;
      result += values[i] * inVector[column];
   }
   
   atomicAdd(out, result);
}

template< typename Real,
          typename Index,
          typename InVector,
          typename OutVector,
          int warpSize >
__global__
void SpMVCSRAdaptiveGlobal( const InVector& inVector,
                            OutVector& outVector,
                            const Index* rowPointers,
                            const Index* columnIndexes,
                            const Real* values,
                            int *blocks,
                            size_t blocks_size,
                            Index getColumns)
{
   /* Configuration ---------------------------------------------------*/
   constexpr size_t SHARED = 49152/sizeof(Real); // number of elements in shared memory for block
   constexpr size_t SHARED_PER_WARP = SHARED / WARPS_PER_BLOCK;
   //--------------------------------------------------------------------
   const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   const size_t laneID = index % warpSize;
   const size_t blockIdx = index / warpSize;
   __shared__ Real shared_res[SHARED];
   Real result = 0;
   if (blockIdx >= blocks_size - 1)
      return;
   const size_t minRow = blocks[blockIdx];
   const size_t maxRow = blocks[blockIdx + 1];
   const size_t minID = rowPointers[minRow];
   const size_t maxID = rowPointers[maxRow];
   const size_t elements = maxID - minID;
   /* rows per block more than 1 */
   if ((maxRow - minRow) > 1) {
      /////////////////////////////////////* CSR STREAM *//////////////
      /* Copy and calculate elements from global to shared memory, coalesced */
      const size_t offset = threadIdx.x / warpSize * SHARED_PER_WARP;
      for (size_t i = laneID; i < elements; i += warpSize) {
         const size_t elementIdx = i + minID;
         const size_t column = columnIndexes[elementIdx];
         if (column >= getColumns)
            continue;
         
         shared_res[i + offset] = values[elementIdx] * inVector[column];
      }

      const size_t row = minRow + laneID;
      if (row >= maxRow)
         return;
      /* Calculate result */
      const size_t to = rowPointers[row + 1] - minID;
      for (size_t i = rowPointers[row] - minID; i < to; ++i) {
         result += shared_res[i + offset];
      }
      outVector[row] = result; // Write result
   } 
   else if (elements <= MAX_PER_WARP) {
      /////////////////////////////////////* CSR VECTOR *//////////////
      for (size_t i = minID + laneID; i < maxID; i += warpSize) {
         size_t column = columnIndexes[i];
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
      if (laneID == 0) outVector[minRow] = result; // Write result
   }
   else { // too long row
      /////////////////////////////////////* CSR DYNAMIC VECTOR *//////////////
      
      /* Number of warps we need.
         This warp can be used to calculate result too, -1 warp */
      size_t warps = elements / ELEMENTS_PER_WARP;
      warps = elements % ELEMENTS_PER_WARP ? warps : warps - 1;

      size_t blocks = warps / WARPS_PER_BLOCK;
      blocks = warps % WARPS_PER_BLOCK ? blocks + 1 : blocks;

      /* Execute a lot of CSR Vector */
      if (laneID == 0) {
         spmvCSRVectorHelper<Real, Index, InVector, warpSize> <<<blocks, THREADS_PER_BLOCK>>>(
                     inVector,
                     columnIndexes,
                     values,
                     getColumns,
                     &outVector[minRow],
                     minID + ELEMENTS_PER_WARP,
                     maxID,
                     ELEMENTS_PER_WARP
         );
      }
      /* CSR Vector */
      for (size_t i = minID + laneID; i < minID + ELEMENTS_PER_WARP; i += warpSize) {
         size_t column = columnIndexes[i];
         if (column >= getColumns)
            break;

         result += values[i] * inVector[column];
      }
      /* Write result */
      atomicAdd(&outVector[minRow], result);
   }
}


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
          typename Device,
          typename Index,
          CSRKernel KernelType >
   template< typename InVector,
             typename OutVector,
             int warpSize >
__device__
void CSR< Real, Device, Index, KernelType >::vectorProductCuda( const InVector& inVector,
                                                             OutVector& outVector,
                                                             int gridIdx ) const
{
   switch( KernelType )
   {
      case CSRScalar:
         // TODO:
         /* FIXME */
         spmvCudaLightSpmv< InVector, OutVector, warpSize >( inVector, outVector, gridIdx );
         break;
      case CSRVector:
         spmvCudaVectorized< InVector, OutVector, warpSize >( inVector, outVector, gridIdx );
         break;
      case CSRLight:
         spmvCudaLightSpmv< InVector, OutVector, warpSize >( inVector, outVector, gridIdx );
         break;
      case CSRAdaptive:
         // spmvCSRAdaptive< InVector, OutVector, warpSize >( inVector, outVector, gridIdx, blocks, size );
         /* FIXME */
         spmvCudaLightSpmv< InVector, OutVector, warpSize >( inVector, outVector, gridIdx );
         break;
      case CSRStream:
         // TODO:
         /* FIXME */
         spmvCudaLightSpmv< InVector, OutVector, warpSize >( inVector, outVector, gridIdx );
         break;
   }


   /*IndexType globalIdx = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   const IndexType warpStart = warpSize * ( globalIdx / warpSize );
   const IndexType warpEnd = min( warpStart + warpSize, this->getRows() );
   const IndexType inWarpIdx = globalIdx % warpSize;

   if( this->getCudaKernelType() == vector )
      

   /////
   // Hybrid mode
   //
   const Index firstRow = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x;
   const IndexType lastRow = min( this->getRows(), firstRow + blockDim. x );
   const IndexType nonzerosPerRow = ( this->rowPointers[ lastRow ] - this->rowPointers[ firstRow ] ) /
                                    ( lastRow - firstRow );

   if( nonzerosPerRow < this->getHybridModeSplit() )
   {
      /////
      // Use the scalar mode
      //
      if( globalIdx < this->getRows() )
          outVector[ globalIdx ] = this->rowVectorProduct( globalIdx, inVector );
   }
   else
   {
      ////
      // Use the vector mode
      //
      spmvCudaVectorized< InVector, OutVector, warpSize >( inVector, outVector, warpStart, warpEnd, inWarpIdx );
   }*/
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

#ifdef HAVE_CUDA

template< typename Real,
          typename Index,
          CSRKernel KernelType,
          typename InVector,
          typename OutVector,
          int warpSize >
__global__ void CSRVectorProductCudaKernel( const CSR< Real, Devices::Cuda, Index, KernelType >* matrix,
                                            const InVector* inVector,
                                            OutVector* outVector, 
                                            int gridIdx)
{
   typedef CSR< Real, Devices::Cuda, Index > Matrix;
   static_assert( std::is_same< typename Matrix::DeviceType, Devices::Cuda >::value, "" );
   const typename Matrix::IndexType rowIdx = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( KernelType == CSRScalar )
   {
      if( rowIdx < matrix->getRows() )
         ( *outVector )[ rowIdx ] = matrix->rowVectorProduct( rowIdx, *inVector );
   }
   else
   {
      matrix->template vectorProductCuda< InVector, OutVector, warpSize >
                                        ( *inVector, *outVector, gridIdx );
   }
}
#endif

template< typename Real,
          typename Index,
          CSRKernel KernelType,
          typename InVector,
          typename OutVector >
void CSRVectorProductCuda( const CSR< Real, Devices::Cuda, Index, KernelType >& matrix,
                                    const InVector& inVector,
                                    OutVector& outVector)
{
#ifdef HAVE_CUDA
   typedef CSR< Real, Devices::Cuda, Index, KernelType > Matrix;
   typedef typename Matrix::IndexType IndexType;
   Matrix* kernel_this = Cuda::passToDevice( matrix );
   InVector* kernel_inVector = Cuda::passToDevice( inVector );
   OutVector* kernel_outVector = Cuda::passToDevice( outVector );
   TNL_CHECK_CUDA_DEVICE;
   dim3 cudaBlockSize( 256 );
   //dim3 cudaGridSize( Cuda::getMaxGridSize() );
   const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
   const IndexType cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridSize() );
   for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
   {
      //if( gridIdx == cudaGrids - 1 )
      //   cudaGridSize.x = cudaBlocks % Cuda::getMaxGridSize();
      //const int sharedMemory = cudaBlockSize.x * sizeof( Real );
      //const int threads = cudaBlockSize.x;
      if( matrix.getCudaWarpSize() == 32 ) {
         // CSRVectorProductCudaKernel< Real, Index, KernelType, InVector, OutVector, 32 >
         //                                    <<< 2, 1024 >>>
         //                                    ( kernel_this,
         //                                      kernel_inVector,
         //                                      kernel_outVector,
         //                                      gridIdx, kernelBlocks, size );
         CSRScalarGlobal< Real, Index, KernelType, InVector, OutVector, 32 >
                                            <<< 2, 1024 >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx );
      if( matrix.getCudaWarpSize() == 16 )
         CSRVectorProductCudaKernel< Real, Index, KernelType, InVector, OutVector, 16 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx);
      if( matrix.getCudaWarpSize() == 8 )
         CSRVectorProductCudaKernel< Real, Index, KernelType, InVector, OutVector, 8 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx);
      if( matrix.getCudaWarpSize() == 4 )
         CSRVectorProductCudaKernel< Real, Index, KernelType, InVector, OutVector, 4 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx);
      if( matrix.getCudaWarpSize() == 2 )
         CSRVectorProductCudaKernel< Real, Index, KernelType, InVector, OutVector, 2 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx);
      if( matrix.getCudaWarpSize() == 1 )
         CSRVectorProductCudaKernel< Real, Index, KernelType, InVector, OutVector, 1 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx);
   }
   TNL_CHECK_CUDA_DEVICE;
   Cuda::freeFromDevice( kernel_this );
   Cuda::freeFromDevice( kernel_inVector );
   Cuda::freeFromDevice( kernel_outVector );
   TNL_CHECK_CUDA_DEVICE;
#endif
}


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
         // if (KernelType == CSRAdaptive) {
            /* Configuration ---------------------------------------------------*/
            constexpr size_t SHARED = 49152/sizeof(Real);
            constexpr size_t SHARED_PER_WARP = SHARED / WARPS_PER_BLOCK;
            //--------------------------------------------------------------------
            /* Fill in blocks */
            std::vector<int> inBlock;
            inBlock.push_back(0);
            size_t sum = 0;
            size_t i;
            int prev_i = 0;
            for (i = 1; i < matrix.getRowPointers().getSize() - 1; ++i) {
               size_t elements = matrix.getRowPointers().getElement(i) -
                                    matrix.getRowPointers().getElement(i - 1);
               sum += elements;
               if (sum > SHARED_PER_WARP) {
                  if (i - prev_i == 1) {
                     inBlock.push_back(i);
                  } else {
                     inBlock.push_back(i - 1);
                     --i;
                  }
                  sum = 0;
                  prev_i = i;
                  continue;
               }
               if (i - prev_i == 32) {
                  inBlock.push_back(i);
                  prev_i = i;
                  sum = 0;
               }
            }
            inBlock.push_back(matrix.getRowPointers().getSize() - 1);
            /* Copy memory to GPU */
            const InVector *kernelInVector = Cuda::passToDevice( inVector );
            OutVector *kernelOutVector = Cuda::passToDevice( outVector );

            /* blocks */
            int *kernelBlocks;
            cudaMalloc((void **)&kernelBlocks, sizeof(int) * inBlock.size());
            cudaMemcpy(kernelBlocks, inBlock.data(), inBlock.size() * sizeof(int), cudaMemcpyHostToDevice);

            /* values */
            Real *kernelValues;
            cudaMalloc((void **)&kernelValues, sizeof(Real) * matrix.getValues().getSize());
            cudaMemcpy(kernelValues,
                       (Real *)matrix.getValues().getData(),
                       matrix.getValues().getSize() * sizeof(Real),
                       cudaMemcpyHostToDevice);

            /* columns */
            Index *kernelColumns;
            cudaMalloc((void **)&kernelColumns, sizeof(Index) * matrix.getColumnIndexes().getSize());
            cudaMemcpy(kernelColumns,
                       (Index *)matrix.getColumnIndexes().getData(),
                       matrix.getColumnIndexes().getSize() * sizeof(Index),
                       cudaMemcpyHostToDevice);

            /* row pointers */
            Index *kernelRowPointers;
            cudaMalloc((void **)&kernelRowPointers, sizeof(Index) * matrix.getRowPointers().getSize());
            cudaMemcpy(kernelRowPointers,
                       (Index *)matrix.getRowPointers().getData(),
                       matrix.getRowPointers().getSize() * sizeof(Index),
                       cudaMemcpyHostToDevice);
            
            size_t needed_threads = 32 * (inBlock.size() - 1); // number of threads we need
            size_t blocks = needed_threads / THREADS_PER_BLOCK; // warp per block
            blocks = needed_threads % THREADS_PER_BLOCK ? blocks + 1 : blocks;
            
            SpMVCSRAdaptiveGlobal< Real, Index, InVector, OutVector, 32 ><<<blocks, THREADS_PER_BLOCK>>>(
                    *kernelInVector, 
                    *kernelOutVector,
                    kernelRowPointers,
                    kernelColumns,
                    kernelValues,
                    kernelBlocks,
                    inBlock.size(),
                    matrix.getColumns()
            );

            Cuda::freeFromDevice( kernelInVector );
            Cuda::freeFromDevice( kernelOutVector );
            cudaFree(kernelBlocks);
            cudaFree(kernelValues);
            cudaFree(kernelColumns);
            cudaFree(kernelRowPointers);

         // } else
#endif /* HAVE_CUDA */
            // CSRVectorProductCuda( matrix, inVector, outVector);
#endif
      }

};

} //namespace Legacy
} // namespace Matrices
} // namespace TNL
