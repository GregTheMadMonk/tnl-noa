/***************************************************************************
                          CSR_impl.h  -  description
                             -------------------
    begin                : Dec 10, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/CSR.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Math.h>

#ifdef HAVE_CUSPARSE
#include <cusparse.h>
#endif

namespace TNL {
namespace Matrices {   

#ifdef HAVE_CUSPARSE
template< typename Real, typename Index >
class tnlCusparseCSRWrapper {};
#endif


template< typename Real,
          typename Device,
          typename Index >
CSR< Real, Device, Index >::CSR()
: spmvCudaKernel( hybrid ),
  cudaWarpSize( 32 ), //Devices::Cuda::getWarpSize() )
  hybridModeSplit( 4 )
{
};

template< typename Real,
          typename Device,
          typename Index >
String CSR< Real, Device, Index >::getType()
{
   return String( "Matrices::CSR< ") +
          String( TNL::getType< Real>() ) +
          String( ", " ) +
          Device :: getDeviceType() +
          String( " >" );
}

template< typename Real,
          typename Device,
          typename Index >
String CSR< Real, Device, Index >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Real,
          typename Device,
          typename Index >
String CSR< Real, Device, Index >::getSerializationType()
{
   return HostType::getType();
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
   this->rowPointers.computeExclusivePrefixSum();
   this->maxRowLength = rowLengths.max();

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
   /*TNL_ASSERT( row >= 0 && row < this->rows &&
              column >= 0 && column <= this->rows,
              std::cerr << " row = " << row
                   << " column = " << column
                   << " this->rows = " << this->rows
                   << " this->columns = " << this-> columns );*/

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
   TNL_ASSERT( false, std::cerr << "TODO: implement" );
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
   TNL_ASSERT( false, std::cerr << "TODO: implement" );
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
bool CSR< Real, Device, Index >::save( File& file ) const
{
   if( ! Sparse< Real, Device, Index >::save( file ) ||
       ! this->rowPointers.save( file ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool CSR< Real, Device, Index >::load( File& file )
{
   if( ! Sparse< Real, Device, Index >::load( file ) ||
       ! this->rowPointers.load( file ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool CSR< Real, Device, Index >::save( const String& fileName ) const
{
   return Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
bool CSR< Real, Device, Index >::load( const String& fileName )
{
   return Object::load( fileName );
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
void CSR< Real, Device, Index >::spmvCudaVectorized( const InVector& inVector,
                                                              OutVector& outVector,
                                                              const IndexType warpStart,
                                                              const IndexType warpEnd,
                                                              const IndexType inWarpIdx ) const
{
   volatile Real* aux = Devices::Cuda::getSharedMemory< Real >();
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
                                                             int gridIdx ) const
{
   IndexType globalIdx = ( gridIdx * Devices::Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   const IndexType warpStart = warpSize * ( globalIdx / warpSize );
   const IndexType warpEnd = min( warpStart + warpSize, this->getRows() );
   const IndexType inWarpIdx = globalIdx % warpSize;

   if( this->getCudaKernelType() == vector )
      spmvCudaVectorized< InVector, OutVector, warpSize >( inVector, outVector, warpStart, warpEnd, inWarpIdx );

   /****
    * Hybrid mode
    */
   const Index firstRow = ( gridIdx * Devices::Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x;
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

#ifdef HAVE_MIC
template<>
class CSRDeviceDependentCode< Devices::MIC >
{
   public:

      typedef Devices::MIC Device;

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const CSR< Real, Device, Index >& matrix,      
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         std::cout <<"Not Implemented YET tnlCSRMatrixDeviceDependentCode for MIC" <<endl;
      };
  /*       const Index rows = matrix.getRows();
         const tnlCSRMatrix< Real, Device, Index >* matrixPtr = &matrix;
         const InVector* inVectorPtr = &inVector;
         OutVector* outVectorPtr = &outVector;
#ifdef HAVE_OPENMP
#pragma omp parallel for firstprivate( matrixPtr, inVectorPtr, outVectorPtr ), schedule(static ), if( Devices::Host::isOMPEnabled() )
#endif         
         for( Index row = 0; row < rows; row ++ )
            ( *outVectorPtr )[ row ] = matrixPtr->rowVectorProduct( row, *inVectorPtr );
      }*/

};
#endif

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename InVector,
          typename OutVector,
          int warpSize >
__global__ void CSRVectorProductCudaKernel( const CSR< Real, Devices::Cuda, Index >* matrix,
                                                     const InVector* inVector,
                                                     OutVector* outVector,
                                                     int gridIdx )
{
   typedef CSR< Real, Devices::Cuda, Index > Matrix;
   static_assert( std::is_same< typename Matrix::DeviceType, Devices::Cuda >::value, "" );
   const typename Matrix::IndexType rowIdx = ( gridIdx * Devices::Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
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
void CSRVectorProductCuda( const CSR< Real, Devices::Cuda, Index >& matrix,
                                    const InVector& inVector,
                                    OutVector& outVector )
{
#ifdef HAVE_CUDA
   typedef CSR< Real, Devices::Cuda, Index > Matrix;
   typedef typename Matrix::IndexType IndexType;
   Matrix* kernel_this = Devices::Cuda::passToDevice( matrix );
   InVector* kernel_inVector = Devices::Cuda::passToDevice( inVector );
   OutVector* kernel_outVector = Devices::Cuda::passToDevice( outVector );
   TNL_CHECK_CUDA_DEVICE;
   dim3 cudaBlockSize( 256 ), cudaGridSize( Devices::Cuda::getMaxGridSize() );
   const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
   const IndexType cudaGrids = roundUpDivision( cudaBlocks, Devices::Cuda::getMaxGridSize() );
   for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
   {
      if( gridIdx == cudaGrids - 1 )
         cudaGridSize.x = cudaBlocks % Devices::Cuda::getMaxGridSize();
      const int sharedMemory = cudaBlockSize.x * sizeof( Real );
      if( matrix.getCudaWarpSize() == 32 )
         CSRVectorProductCudaKernel< Real, Index, InVector, OutVector, 32 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx );
      if( matrix.getCudaWarpSize() == 16 )
         CSRVectorProductCudaKernel< Real, Index, InVector, OutVector, 16 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx );
      if( matrix.getCudaWarpSize() == 8 )
         CSRVectorProductCudaKernel< Real, Index, InVector, OutVector, 8 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx );
      if( matrix.getCudaWarpSize() == 4 )
         CSRVectorProductCudaKernel< Real, Index, InVector, OutVector, 4 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx );
      if( matrix.getCudaWarpSize() == 2 )
         CSRVectorProductCudaKernel< Real, Index, InVector, OutVector, 2 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx );
      if( matrix.getCudaWarpSize() == 1 )
         CSRVectorProductCudaKernel< Real, Index, InVector, OutVector, 1 >
                                            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                            ( kernel_this,
                                              kernel_inVector,
                                              kernel_outVector,
                                              gridIdx );

   }
   TNL_CHECK_CUDA_DEVICE;
   Devices::Cuda::freeFromDevice( kernel_this );
   Devices::Cuda::freeFromDevice( kernel_inVector );
   Devices::Cuda::freeFromDevice( kernel_outVector );
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
         CSRVectorProductCuda( matrix, inVector, outVector );
#endif
      }

};

} // namespace Matrices
} // namespace TNL
