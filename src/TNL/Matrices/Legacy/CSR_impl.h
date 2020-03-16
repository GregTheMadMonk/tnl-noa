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
#include <TNL/Exceptions/NotImplementedError.h>
#include <vector>

#ifdef HAVE_CUSPARSE
#include <cusparse.h>
#endif

namespace TNL {
namespace Matrices {
   namespace Legacy {

#ifdef HAVE_CUSPARSE
template< typename Real, typename Index >
class tnlCusparseCSRWrapper {};
#endif


template< typename Real,
          typename Device,
          typename Index >
CSR< Real, Device, Index >::CSR()
: spmvCudaKernel( hybrid ),
  cudaWarpSize( 32 ), //Cuda::getWarpSize() )
  hybridModeSplit( 4 )
{
};

template< typename Real,
          typename Device,
          typename Index >
String CSR< Real, Device, Index >::getSerializationType()
{
   return String( "Matrices::CSR< ") +
          TNL::getType< Real>() +
          ", [any_device], " +
          String( TNL::getType< Index >() ) +
          String( " >" );
}

template< typename Real,
          typename Device,
          typename Index >
String CSR< Real, Device, Index >::getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index >
void CSR< Real, Device, Index >::setDimensions( const IndexType rows,
                                                const IndexType columns )
{
   Sparse< Real, Device, Index >::setDimensions( rows, columns );
   this->rowPointers.setSize( this->rows + 1 );
   this->rowPointers.setValue( 0 );
}

template< typename Real,
          typename Device,
          typename Index >
void CSR< Real, Device, Index >::setCompressedRowLengths( ConstCompressedRowLengthsVectorView rowLengths )
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
          typename Index >
void CSR< Real, Device, Index >::getCompressedRowLengths( CompressedRowLengthsVectorView rowLengths ) const
{
   TNL_ASSERT_EQ( rowLengths.getSize(), this->getRows(), "invalid size of the rowLengths vector" );
   for( IndexType row = 0; row < this->getRows(); row++ )
      rowLengths.setElement( row, this->getRowLength( row ) );
}

template< typename Real,
          typename Device,
          typename Index >
Index CSR< Real, Device, Index >::getRowLength( const IndexType row ) const
{
   return this->rowPointers.getElement( row + 1 ) - this->rowPointers.getElement( row );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index CSR< Real, Device, Index >::getRowLengthFast( const IndexType row ) const
{
   return this->rowPointers[ row + 1 ] - this->rowPointers[ row ];
}

template< typename Real,
          typename Device,
          typename Index >
Index CSR< Real, Device, Index >::getNonZeroRowLength( const IndexType row ) const
{
    // TODO: Fix/Implement
    TNL_ASSERT( false, std::cerr << "TODO: Fix/Implement" );
    return 0;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index CSR< Real, Device, Index >::getNonZeroRowLengthFast( const IndexType row ) const
{
   ConstMatrixRow matrixRow = this->getRow( row );
   return matrixRow.getNonZeroElementsCount();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
void CSR< Real, Device, Index >::setLike( const CSR< Real2, Device2, Index2 >& matrix )
{
   Sparse< Real, Device, Index >::setLike( matrix );
   this->rowPointers.setLike( matrix.rowPointers );
}

template< typename Real,
          typename Device,
          typename Index >
void CSR< Real, Device, Index >::reset()
{
   Sparse< Real, Device, Index >::reset();
   this->rowPointers.reset();
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool CSR< Real, Device, Index >::setElementFast( const IndexType row,
                                                          const IndexType column,
                                                          const Real& value )
{
   return this->addElementFast( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
bool CSR< Real, Device, Index >::setElement( const IndexType row,
                                                      const IndexType column,
                                                      const Real& value )
{
   return this->addElement( row, column, value, 0.0 );
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool CSR< Real, Device, Index >::addElementFast( const IndexType row,
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
          typename Index >
bool CSR< Real, Device, Index >::addElement( const IndexType row,
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
          typename Index >
__cuda_callable__
bool CSR< Real, Device, Index > :: setRowFast( const IndexType row,
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
          typename Index >
bool CSR< Real, Device, Index > :: setRow( const IndexType row,
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
__cuda_callable__
bool CSR< Real, Device, Index > :: addRowFast( const IndexType row,
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
bool CSR< Real, Device, Index > :: addRow( const IndexType row,
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
__cuda_callable__
Real CSR< Real, Device, Index >::getElementFast( const IndexType row,
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
          typename Index >
Real CSR< Real, Device, Index >::getElement( const IndexType row,
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
          typename Index >
__cuda_callable__
void CSR< Real, Device, Index >::getRowFast( const IndexType row,
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
__cuda_callable__
typename CSR< Real, Device, Index >::MatrixRow
CSR< Real, Device, Index >::
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
          typename Index >
__cuda_callable__
typename CSR< Real, Device, Index >::ConstMatrixRow
CSR< Real, Device, Index >::
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
          typename Index >
   template< typename Vector >
__cuda_callable__
typename Vector::RealType CSR< Real, Device, Index >::rowVectorProduct( const IndexType row,
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
void CSR< Real, Device, Index >::vectorProduct( const InVector& inVector,
                                                OutVector& outVector ) const
{
   DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Index2 >
void CSR< Real, Device, Index >::addMatrix( const CSR< Real2, Device, Index2 >& matrix,
                                            const RealType& matrixMultiplicator,
                                            const RealType& thisMatrixMultiplicator )
{
   throw Exceptions::NotImplementedError( "CSR::addMatrix is not implemented." );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Index2 >
void CSR< Real, Device, Index >::getTransposition( const CSR< Real2, Device, Index2 >& matrix,
                                                                      const RealType& matrixMultiplicator )
{
   throw Exceptions::NotImplementedError( "CSR::getTransposition is not implemented." );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector1, typename Vector2 >
bool CSR< Real, Device, Index >::performSORIteration( const Vector1& b,
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
          typename Index >
CSR< Real, Device, Index >&
CSR< Real, Device, Index >::operator=( const CSR& matrix )
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
          typename Index >
   template< typename Real2, typename Device2, typename Index2, typename >
CSR< Real, Device, Index >&
CSR< Real, Device, Index >::operator=( const CSR< Real2, Device2, Index2 >& matrix )
{
   this->setLike( matrix );
   this->values = matrix.values;
   this->columnIndexes = matrix.columnIndexes;
   this->rowPointers = matrix.rowPointers;
   return *this;
}


template< typename Real,
          typename Device,
          typename Index >
void CSR< Real, Device, Index >::save( File& file ) const
{
   Sparse< Real, Device, Index >::save( file );
   file << this->rowPointers;
}

template< typename Real,
          typename Device,
          typename Index >
void CSR< Real, Device, Index >::load( File& file )
{
   Sparse< Real, Device, Index >::load( file );
   file >> this->rowPointers;
}

template< typename Real,
          typename Device,
          typename Index >
void CSR< Real, Device, Index >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void CSR< Real, Device, Index >::load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void CSR< Real, Device, Index >::print( std::ostream& str ) const
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

template< typename Real,
          typename Device,
          typename Index >
void CSR< Real, Device, Index >::setCudaKernelType( const SPMVCudaKernel kernel )
{
   this->spmvCudaKernel = kernel;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename CSR< Real, Device, Index >::SPMVCudaKernel CSR< Real, Device, Index >::getCudaKernelType() const
{
   return this->spmvCudaKernel;
}

template< typename Real,
          typename Device,
          typename Index >
void CSR< Real, Device, Index >::setCudaWarpSize( const int warpSize )
{
   this->cudaWarpSize = warpSize;
}

template< typename Real,
          typename Device,
          typename Index >
int CSR< Real, Device, Index >::getCudaWarpSize() const
{
   return this->cudaWarpSize;
}

template< typename Real,
          typename Device,
          typename Index >
void CSR< Real, Device, Index >::setHybridModeSplit( const IndexType hybridModeSplit )
{
   this->hybridModeSplit = hybridModeSplit;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index CSR< Real, Device, Index >::getHybridModeSplit() const
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
void CSR< Real, Device, Index >::spmvCudaLightSpmv( const InVector& inVector,
                                                      OutVector& outVector,
                                                      int gridIdx) const
{
   const IndexType index = blockIdx.x * blockDim.x + threadIdx.x;
   const IndexType elemPerGroup   = 4;
   const IndexType laneID      = index % 32;
   const IndexType groupID     = laneID / elemPerGroup;
   const IndexType inGroupID   = laneID % elemPerGroup;

   IndexType row, minID, column, maxID, idxMtx;
   __shared__ unsigned rowCnt;

   if (index == 0) rowCnt = 0;  // Init shared variable
   __syncthreads();

   while (true) {

      /* Get row number */
      if (inGroupID == 0) row = atomicAdd(&rowCnt, 1);

      /* Propagate row number in group */
      row = __shfl_sync((unsigned)(warpSize - 1), row, groupID * elemPerGroup);

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
         idxMtx += elemPerGroup;
      }

      /* Parallel reduction */
      for (int i = elemPerGroup/2; i > 0; i /= 2)
         result += __shfl_down_sync((unsigned)(warpSize - 1), result, i);
      /* Write result */
      if (inGroupID == 0) {
         outVector[row] = result;
      }
   }
}


template< typename Real,
          typename Device,
          typename Index >
   template< typename InVector,
             typename OutVector,
             int warpSize >
__device__
void CSR< Real, Device, Index >::spmvCSRAdaptive( const InVector& inVector,
                                                      OutVector& outVector,
                                                      int gridIdx,
                                                      unsigned *blocks,
                                                      size_t blocks_size) const
{
   constexpr IndexType SHARED = 1024; /* TODO: delete */
   const IndexType index = blockIdx.x * blockDim.x + threadIdx.x;
   __shared__ unsigned blockCnt;
   __shared__ float shared_res[SHARED];
   float result = 0.0;
   if (index == 0) blockCnt = 0;  // Init shared variable
   __syncthreads();
   
   const IndexType laneID = index % warpSize;
   IndexType blockIdx;

   while (true) {
      /* Get row number */
      if (laneID == 0) blockIdx = atomicAdd(&blockCnt, 1);
      /* Propagate row number in warp */
      blockIdx = __shfl_sync((unsigned)(warpSize - 1), blockIdx, 0);
      if (blockIdx >= blocks_size - 1)
         return;
      /* Number of elements */
      const IndexType minRow = blocks[blockIdx];
      const IndexType maxRow = blocks[blockIdx + 1];
      const IndexType minID = this->rowPointers[minRow];
      const IndexType maxID = this->rowPointers[maxRow];
      const IndexType elements = maxID - minID;
      result = 0.0;
      /* rows per block more than 1 */
      if ((maxRow - minRow) > 1) {
         /////////////////////////////////////* CSR STREAM *//////////////
         /* Copy and calculate elements from global to shared memory, coalesced */
         for (IndexType i = laneID; i < elements; i += warpSize) {
            const IndexType elementIdx = i + minID;
            const IndexType column = this->columnIndexes[elementIdx];
            if (column >= this->getColumns())
               break;
            shared_res[i] = this->values[elementIdx] * inVector[column];
         }

         const IndexType row = minRow + laneID;
         if (row >= maxRow - 1)
            continue;
         /* Calculate result */
         const IndexType to = this->rowPointers[row + 1] - minID;
         for (IndexType i = this->rowPointers[row] - minID; i < to; ++i) {
            result += shared_res[i];
         }
         outVector[row] = result; // Write result
      } else {
         /////////////////////////////////////* CSR VECTOR *//////////////
         for (IndexType i = minID + laneID; i < maxID; i += warpSize) {
            const IndexType column = this->columnIndexes[i];
            result += this->values[i] * inVector[column];
         }
         /* Reduction */
         result += __shfl_down_sync((unsigned)(warpSize - 1), result, 16);
         result += __shfl_down_sync((unsigned)(warpSize - 1), result, 8);
         result += __shfl_down_sync((unsigned)(warpSize - 1), result, 4);
         result += __shfl_down_sync((unsigned)(warpSize - 1), result, 2);
         result += __shfl_down_sync((unsigned)(warpSize - 1), result, 1);
         if (laneID == 0) outVector[minRow] = result; // Write result
      }
   }
}


template< typename Real,
          typename Device,
          typename Index >
   template< typename InVector,
             typename OutVector,
             int warpSize >
__device__
void CSR< Real, Device, Index >::spmvCudaVectorized( const InVector& inVector,
                                                              OutVector& outVector,
                                                              const IndexType warpStart,
                                                              const IndexType warpEnd,
                                                              const IndexType inWarpIdx ) const
{
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
          typename Index >
   template< typename InVector,
             typename OutVector,
             int warpSize >
__device__
void CSR< Real, Device, Index >::vectorProductCuda( const InVector& inVector,
                                                             OutVector& outVector,
                                                             int gridIdx,
                                                             unsigned *blocks, size_t size ) const
{
   spmvCSRAdaptive< InVector, OutVector, warpSize >( inVector, outVector, gridIdx, blocks, size );
   return;

   IndexType globalIdx = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   const IndexType warpStart = warpSize * ( globalIdx / warpSize );
   const IndexType warpEnd = min( warpStart + warpSize, this->getRows() );
   const IndexType inWarpIdx = globalIdx % warpSize;

   if( this->getCudaKernelType() == vector )
      spmvCudaVectorized< InVector, OutVector, warpSize >( inVector, outVector, warpStart, warpEnd, inWarpIdx );

   /****
    * Hybrid mode
    */
   const Index firstRow = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x;
   const IndexType lastRow = min( this->getRows(), firstRow + blockDim. x );
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
class CSRDeviceDependentCode< Devices::Host >
{
   public:

      typedef Devices::Host Device;

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const CSR< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         const Index rows = matrix.getRows();
         const CSR< Real, Device, Index >* matrixPtr = &matrix;
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
          typename InVector,
          typename OutVector,
          int warpSize >
__global__ void CSRVectorProductCudaKernel( const CSR< Real, Devices::Cuda, Index >* matrix,
                                                     const InVector* inVector,
                                                     OutVector* outVector,
                                                     int gridIdx,
                                                     unsigned *blocks, size_t size)
{
   typedef CSR< Real, Devices::Cuda, Index > Matrix;
   static_assert( std::is_same< typename Matrix::DeviceType, Devices::Cuda >::value, "" );
   const typename Matrix::IndexType rowIdx = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( matrix->getCudaKernelType() == Matrix::scalar )
   {
      if( rowIdx < matrix->getRows() )
         ( *outVector )[ rowIdx ] = matrix->rowVectorProduct( rowIdx, *inVector );
   }
   if( matrix->getCudaKernelType() == Matrix::vector ||
       matrix->getCudaKernelType() == Matrix::hybrid )
   {
      matrix->template vectorProductCuda< InVector, OutVector, warpSize >
                                        ( *inVector, *outVector, gridIdx, blocks, size );
   }
}
#endif

template< typename Real,
          typename Index,
          typename InVector,
          typename OutVector >
void CSRVectorProductCuda( const CSR< Real, Devices::Cuda, Index >& matrix,
                                    const InVector& inVector,
                                    OutVector& outVector,
                                    unsigned *blocks,
                                    size_t size )
{
#ifdef HAVE_CUDA
   typedef CSR< Real, Devices::Cuda, Index > Matrix;
   typedef typename Matrix::IndexType IndexType;
   Matrix* kernel_this = Cuda::passToDevice( matrix );
   InVector* kernel_inVector = Cuda::passToDevice( inVector );
   OutVector* kernel_outVector = Cuda::passToDevice( outVector );
   unsigned *kernelBlocks = Cuda::passToDevice( *blocks );
   TNL_CHECK_CUDA_DEVICE;
   dim3 cudaBlockSize( 256 ), cudaGridSize( Cuda::getMaxGridSize() );
   const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
   const IndexType cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridSize() );
   for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
   {
      if( gridIdx == cudaGrids - 1 )
         cudaGridSize.x = cudaBlocks % Cuda::getMaxGridSize();
      const int sharedMemory = cudaBlockSize.x * sizeof( Real );
      if( matrix.getCudaWarpSize() == 32 )
         CSRVectorProductCudaKernel< Real, Index, InVector, OutVector, 32 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx, kernelBlocks, size );
      if( matrix.getCudaWarpSize() == 16 )
         CSRVectorProductCudaKernel< Real, Index, InVector, OutVector, 16 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx, kernelBlocks, size );
      if( matrix.getCudaWarpSize() == 8 )
         CSRVectorProductCudaKernel< Real, Index, InVector, OutVector, 8 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx, kernelBlocks, size );
      if( matrix.getCudaWarpSize() == 4 )
         CSRVectorProductCudaKernel< Real, Index, InVector, OutVector, 4 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx, kernelBlocks, size );
      if( matrix.getCudaWarpSize() == 2 )
         CSRVectorProductCudaKernel< Real, Index, InVector, OutVector, 2 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx, kernelBlocks, size );
      if( matrix.getCudaWarpSize() == 1 )
         CSRVectorProductCudaKernel< Real, Index, InVector, OutVector, 1 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx, kernelBlocks, size );

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
                typename InVector,
                typename OutVector >
      static void vectorProduct( const CSR< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
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
         std::vector<unsigned> inBlock;
         inBlock.push_back(0);
         unsigned sum = 0;
         for (size_t i = 1; i < matrix.getRowPointers().getSize(); ++i) {
            printf("PTR %d\n", (int)matrix.getRowPointers().getElement(i));
         }
         for (size_t i = 1; i < matrix.getRowPointers().getSize() - 1; ++i) {
            size_t elements = matrix.getRowPointers().getElement(i + 1) -
                                 matrix.getRowPointers().getElement(i);
            if (elements > 1024) {
               if (sum == 0) {
                  inBlock.push_back(i);
               }
               else {
                  inBlock.push_back(i - 1);
                  inBlock.push_back(i);
               }
               sum = 0;
            }
            sum += elements;
            
            if (sum <= 1024)
               continue;
            else {
               inBlock.push_back(i - 1);
               sum = elements;
            }
         }
         inBlock.push_back(matrix.getRowPointers().getSize() - 1);
         CSRVectorProductCuda( matrix, inVector, outVector, inBlock.data(), inBlock.size() );
#endif
      }

};

} //namespace Legacy
} // namespace Matrices
} // namespace TNL
