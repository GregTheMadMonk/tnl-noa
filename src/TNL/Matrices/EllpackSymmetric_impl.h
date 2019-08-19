/***************************************************************************
                          EllpackSymmetric_impl.h  -  description
                             -------------------
    begin                : Aug 30, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/EllpackSymmetric.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Math.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Matrices {

template< typename Real,
          typename Device,
          typename Index >
EllpackSymmetric< Real, Device, Index > :: EllpackSymmetric()
: rowLengths( 0 ), alignedRows( 0 )
{
};

template< typename Real,
          typename Device,
          typename Index >
String EllpackSymmetric< Real, Device, Index > :: getType()
{
   return String( "EllpackSymmetric< ") +
          String( TNL::getType< Real >() ) +
          String( ", " ) +
          Device::getType() +
          String( " >" );
}

template< typename Real,
          typename Device,
          typename Index >
String EllpackSymmetric< Real, Device, Index >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Real,
          typename Device,
          typename Index >
void EllpackSymmetric< Real, Device, Index >::setDimensions( const IndexType rows,
                                                             const IndexType columns )
{
   TNL_ASSERT( rows > 0 && columns > 0,
             std::cerr << "rows = " << rows
                   << " columns = " << columns <<std::endl );
   this->rows = rows;
   this->columns = columns;   
   if( std::is_same< DeviceType, Devices::Cuda >::value )
      this->alignedRows = roundToMultiple( columns, Devices::Cuda::getWarpSize() );
   else this->alignedRows = rows;
   if( this->rowLengths != 0 )
      allocateElements();
}

template< typename Real,
          typename Device,
          typename Index >
void EllpackSymmetric< Real, Device, Index >::setCompressedRowLengths( ConstCompressedRowLengthsVectorView rowLengths )
{
   TNL_ASSERT( this->getRows() > 0, );
   TNL_ASSERT( this->getColumns() > 0, );
   //TNL_ASSERT( this->rowLengths > 0,
   //          std::cerr << "this->rowLengths = " << this->rowLengths );
   this->rowLengths = this->maxRowLength = max( rowLengths );
   allocateElements();
}

template< typename Real,
          typename Device,
          typename Index >
bool EllpackSymmetric< Real, Device, Index >::setConstantRowLengths( const IndexType& rowLengths )
{
   TNL_ASSERT( rowLengths > 0,
             std::cerr << " rowLengths = " << rowLengths );
   this->rowLengths = rowLengths;
   if( this->rows > 0 )
      allocateElements();
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
Index EllpackSymmetric< Real, Device, Index >::getRowLength( const IndexType row ) const
{
   return this->rowLengths;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool EllpackSymmetric< Real, Device, Index >::setLike( const EllpackSymmetric< Real2, Device2, Index2 >& matrix )
{
   if( ! Sparse< Real, Device, Index >::setLike( matrix ) )
      return false;
   this->rowLengths = matrix.rowLengths;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
void EllpackSymmetric< Real, Device, Index > :: reset()
{
   Sparse< Real, Device, Index >::reset();
   this->rowLengths = 0;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool EllpackSymmetric< Real, Device, Index >::operator == ( const EllpackSymmetric< Real2, Device2, Index2 >& matrix ) const
{
   TNL_ASSERT( this->getRows() == matrix.getRows() &&
              this->getColumns() == matrix.getColumns(),
             std::cerr << "this->getRows() = " << this->getRows()
                   << " matrix.getRows() = " << matrix.getRows()
                   << " this->getColumns() = " << this->getColumns()
                   << " matrix.getColumns() = " << matrix.getColumns()
                   << " this->getName() = " << this->getName()
                   << " matrix.getName() = " << matrix.getName() );
   // TODO: implement this
   throw Exceptions::NotImplementedError( "EllpackSymmetric::operator== is not implemented." );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool EllpackSymmetric< Real, Device, Index >::operator != ( const EllpackSymmetric< Real2, Device2, Index2 >& matrix ) const
{
   return ! ( ( *this ) == matrix );
}

/*template< typename Real,
          typename Device,
          typename Index >
   template< typename Matrix >
bool EllpackSymmetric< Real, Device, Index >::copyFrom( const Matrix& matrix,
                                                        const CompressedRowLengthsVector& rowLengths )
{
   return tnlMatrix< RealType, DeviceType, IndexType >::copyFrom( matrix, rowLengths );
}*/

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool EllpackSymmetric< Real, Device, Index > :: setElementFast( const IndexType row,
                                                                const IndexType column,
                                                                const Real& value )
{
   return this->addElementFast( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
bool EllpackSymmetric< Real, Device, Index > :: setElement( const IndexType row,
                                                            const IndexType column,
                                                            const Real& value )
{
   return this->addElement( row, column, value, 0.0 );
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool EllpackSymmetric< Real, Device, Index > :: addElementFast( const IndexType row,
                                                                const IndexType column,
                                                                const RealType& value,
                                                                const RealType& thisElementMultiplicator )
{
   // TODO: return this back when CUDA kernels supportstd::cerr
   /*TNL_ASSERT( row >= 0 && row < this->rows &&
              column >= 0 && column <= this->rows,
             std::cerr << " row = " << row
                   << " column = " << column
                   << " this->rows = " << this->rows
                   << " this->columns = " << this-> columns );*/
   typedef EllpackSymmetricDeviceDependentCode< DeviceType > DDCType;
   IndexType i = DDCType::getRowBegin( *this, row );
   const IndexType rowEnd = DDCType::getRowEnd( *this, row );
   const IndexType step = DDCType::getElementStep( *this );

   while( i < rowEnd &&
         this->columnIndexes[ i ] < column &&
         this->columnIndexes[ i ] != this->getPaddingIndex() ) i += step;
   if( i == rowEnd )
      return false;
   if( this->columnIndexes[ i ] == column )
   {
      this->values[ i ] = thisElementMultiplicator * this->values[ i ] + value;
      return true;
   }
   else
      if( this->columnIndexes[ i ] == this->getPaddingIndex() ) // artificial zero
      {
         this->columnIndexes[ i ] = column;
         this->values[ i ] = value;
      }
      else
      {
         Index j = rowEnd - step;
         while( j > i )
         {
            this->columnIndexes[ j ] = this->columnIndexes[ j - step ];
            this->values[ j ] = this->values[ j - step ];
            j -= step;
         }
         this->columnIndexes[ i ] = column;
         this->values[ i ] = value;
      }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool EllpackSymmetric< Real, Device, Index > :: addElement( const IndexType row,
                                                            const IndexType column,
                                                            const RealType& value,
                                                            const RealType& thisElementMultiplicator )
{
   typedef EllpackSymmetricDeviceDependentCode< DeviceType > DDCType;
   IndexType i = DDCType::getRowBegin( *this, row );
   const IndexType rowEnd = DDCType::getRowEnd( *this, row );
   const IndexType step = DDCType::getElementStep( *this );

   while( i < rowEnd &&
          this->columnIndexes.getElement( i ) < column &&
          this->columnIndexes.getElement( i ) != this->getPaddingIndex() ) i += step;
   if( i == rowEnd )
      return false;
   if( this->columnIndexes.getElement( i ) == column )
   {
      this->values.setElement( i, thisElementMultiplicator * this->values.getElement( i ) + value );
      return true;
   }
   else
      if( this->columnIndexes.getElement( i ) == this->getPaddingIndex() )
      {
         this->columnIndexes.setElement( i, column );
         this->values.setElement( i, value );
      }
      else
      {
         IndexType j = rowEnd - step;
         while( j > i )
         {
            this->columnIndexes.setElement( j, this->columnIndexes.getElement( j - step ) );
            this->values.setElement( j, this->values.getElement( j - step ) );
            j -= step;
         }
         this->columnIndexes.setElement( i, column );
         this->values.setElement( i, value );
      }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool EllpackSymmetric< Real, Device, Index > :: setRowFast( const IndexType row,
                                                            const IndexType* columnIndexes,
                                                            const RealType* values,
                                                            const IndexType elements )
{
   typedef EllpackSymmetricDeviceDependentCode< DeviceType > DDCType;
   IndexType elementPointer = DDCType::getRowBegin( *this, row );
   const IndexType rowEnd = DDCType::getRowEnd( *this, row );
   const IndexType step = DDCType::getElementStep( *this );

   if( elements > this->rowLengths )
      return false;
   for( Index i = 0; i < elements; i++ )
   {
      const IndexType column = columnIndexes[ i ];
      if( column < 0 || column >= this->getColumns() )
         return false;
      this->columnIndexes[ elementPointer ] = column;
      this->values[ elementPointer ] = values[ i ];
      elementPointer += step;
   }
   for( Index i = elements; i < this->rowLengths; i++ )
   {
      this->columnIndexes[ elementPointer ] = this->getPaddingIndex();
      elementPointer += step;
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool EllpackSymmetric< Real, Device, Index > :: setRow( const IndexType row,
                                                        const IndexType* columnIndexes,
                                                        const RealType* values,
                                                        const IndexType elements )
{
   typedef EllpackSymmetricDeviceDependentCode< DeviceType > DDCType;
   IndexType elementPointer = DDCType::getRowBegin( *this, row );
   const IndexType rowEnd = DDCType::getRowEnd( *this, row );
   const IndexType step = DDCType::getElementStep( *this );

   if( elements > this->rowLengths )
      return false;

   for( IndexType i = 0; i < elements; i++ )
   {
      const IndexType column = columnIndexes[ i ];
      if( column < 0 || column >= this->getColumns() )
         return false;
      this->columnIndexes.setElement( elementPointer, column );
      this->values.setElement( elementPointer, values[ i ] );
      elementPointer += step;
   }
   for( IndexType i = elements; i < this->rowLengths; i++ )
   {
      this->columnIndexes.setElement( elementPointer, this->getPaddingIndex() );
      elementPointer += step;
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool EllpackSymmetric< Real, Device, Index > :: addRowFast( const IndexType row,
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
bool EllpackSymmetric< Real, Device, Index > :: addRow( const IndexType row,
                                                        const IndexType* columns,
                                                        const RealType* values,
                                                        const IndexType numberOfElements,
                                                        const RealType& thisElementMultiplicator )
{
   return this->addRowFast( row, columns, values, numberOfElements );
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Real EllpackSymmetric< Real, Device, Index >::getElementFast( const IndexType row,
                                                              const IndexType column ) const
{
   if( row < column )
       return this->getElementFast( column, row );

   typedef EllpackSymmetricDeviceDependentCode< DeviceType > DDCType;
   IndexType elementPtr = DDCType::getRowBegin( *this, row );
   const IndexType rowEnd = DDCType::getRowEnd( *this, row );
   const IndexType step = DDCType::getElementStep( *this );

   while( elementPtr < rowEnd &&
          this->columnIndexes[ elementPtr ] < column &&
          this->columnIndexes[ elementPtr ] != this->getPaddingIndex() ) elementPtr += step;
   if( elementPtr < rowEnd && this->columnIndexes[ elementPtr ] == column )
      return this->values[ elementPtr ];
   return 0.0;
}

template< typename Real,
          typename Device,
          typename Index >
Real EllpackSymmetric< Real, Device, Index >::getElement( const IndexType row,
                                                          const IndexType column ) const
{
   if( row < column )
       return this->getElement( column, row );

   typedef EllpackSymmetricDeviceDependentCode< DeviceType > DDCType;
   IndexType elementPtr = DDCType::getRowBegin( *this, row );
   const IndexType rowEnd = DDCType::getRowEnd( *this, row );
   const IndexType step = DDCType::getElementStep( *this );

   while( elementPtr < rowEnd &&
          this->columnIndexes.getElement( elementPtr ) < column &&
          this->columnIndexes.getElement( elementPtr ) != this->getPaddingIndex() ) elementPtr += step;
   if( elementPtr < rowEnd && this->columnIndexes.getElement( elementPtr ) == column )
      return this->values.getElement( elementPtr );
   return 0.0;
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
void EllpackSymmetric< Real, Device, Index >::getRowFast( const IndexType row,
                                                          IndexType* columns,
                                                          RealType* values ) const
{
   typedef EllpackSymmetricDeviceDependentCode< DeviceType > DDCType;
   IndexType elementPtr = DDCType::getRowBegin( *this, row );
   const IndexType rowEnd = DDCType::getRowEnd( *this, row );
   const IndexType step = DDCType::getElementStep( *this );

   for( IndexType i = 0; i < this->rowLengths; i++ )
   {
      columns[ i ] = this->columnIndexes[ elementPtr ];
      values[ i ] = this->values[ elementPtr ];
      elementPtr += step;
   }
}

template< typename Real,
          typename Device,
          typename Index >
void EllpackSymmetric< Real, Device, Index >::getRow( const IndexType row,
                                                      IndexType* columns,
                                                      RealType* values ) const
{
   typedef EllpackSymmetricDeviceDependentCode< DeviceType > DDCType;
   IndexType elementPtr = DDCType::getRowBegin( *this, row );
   const IndexType rowEnd = DDCType::getRowEnd( *this, row );
   const IndexType step = DDCType::getElementStep( *this );

   for( IndexType i = 0; i < this->rowLengths; i++ )
   {
      columns[ i ] = this->columnIndexes.getElement( elementPtr );
      values[ i ] = this->values.getElement( elementPtr );
      elementPtr += step;
   }
}



template< typename Real,
          typename Device,
          typename Index >
   template< typename InVector,
             typename OutVector >
void EllpackSymmetric< Real, Device, Index >::vectorProduct( const InVector& inVector,
                                                                   OutVector& outVector ) const
{
   DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Index2 >
void EllpackSymmetric< Real, Device, Index > :: addMatrix( const EllpackSymmetric< Real2, Device, Index2 >& matrix,
                                                                 const RealType& matrixMultiplicator,
                                                                 const RealType& thisMatrixMultiplicator )
{
   throw Exceptions::NotImplementedError( "EllpackSymmetric::addMatrix is not implemented." );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Index2 >
void EllpackSymmetric< Real, Device, Index >::getTransposition( const EllpackSymmetric< Real2, Device, Index2 >& matrix,
                                                                      const RealType& matrixMultiplicator )
{
   throw Exceptions::NotImplementedError( "EllpackSymmetric::getTransposition is not implemented." );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
bool EllpackSymmetric< Real, Device, Index > :: performSORIteration( const Vector& b,
                                                                           const IndexType row,
                                                                           Vector& x,
                                                                           const RealType& omega ) const
{
   TNL_ASSERT( row >=0 && row < this->getRows(),
             std::cerr << "row = " << row
                   << " this->getRows() = " << this->getRows()
                   << " this->getName() = " << this->getName() <<std::endl );

   RealType diagonalValue( 0.0 );
   RealType sum( 0.0 );

   IndexType i( row * this->rowLengths );
   const IndexType rowEnd( i + this->rowLengths );
   IndexType column;
   while( i < rowEnd && ( column = this->columnIndexes[ i ] ) < this->columns )
   {
      if( column == row )
         diagonalValue = this->values.getElement( i );
      else
         sum += this->values.getElement( row * this->diagonalsShift.getSize() + i ) * x. getElement( column );
      i++;
   }
   if( diagonalValue == ( Real ) 0.0 )
   {
     std::cerr << "There is zero on the diagonal in " << row << "-th row of thge matrix " << this->getName() << ". I cannot perform SOR iteration." <<std::endl;
      return false;
   }
   x. setElement( row, x[ row ] + omega / diagonalValue * ( b[ row ] - sum ) );
   return true;
}


template< typename Real,
          typename Device,
          typename Index >
void EllpackSymmetric< Real, Device, Index >::save( File& file ) const
{
   Sparse< Real, Device, Index >::save( file);
   file.save( &this->rowLengths );
}

template< typename Real,
          typename Device,
          typename Index >
void EllpackSymmetric< Real, Device, Index >::load( File& file )
{
   Sparse< Real, Device, Index >::load( file);
   file.load( &this->rowLengths );
}

template< typename Real,
          typename Device,
          typename Index >
void EllpackSymmetric< Real, Device, Index >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void EllpackSymmetric< Real, Device, Index >::load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void EllpackSymmetric< Real, Device, Index >::print( std::ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      IndexType i( row * this->rowLengths );
      const IndexType rowEnd( i + this->rowLengths );
      while( i < rowEnd &&
             this->columnIndexes.getElement( i ) < this->columns &&
             this->columnIndexes.getElement( i ) != this->getPaddingIndex() )
      {
         const Index column = this->columnIndexes.getElement( i );
         str << " Col:" << column << "->" << this->values.getElement( i ) << "\t";
         i++;
      }
      str <<std::endl;
   }
}

template< typename Real,
          typename Device,
          typename Index >
void EllpackSymmetric< Real, Device, Index >::allocateElements()
{
   Sparse< Real, Device, Index >::allocateMatrixElements( this->alignedRows * this->rowLengths );
}

template<>
class EllpackSymmetricDeviceDependentCode< Devices::Host >
{
   public:

      typedef Devices::Host Device;

      template< typename Real,
                typename Index >
      static Index getRowBegin( const EllpackSymmetric< Real, Device, Index >& matrix,
                                const Index row )
      {
         return row * matrix.rowLengths;
      }

      template< typename Real,
                typename Index >
      static Index getRowEnd( const EllpackSymmetric< Real, Device, Index >& matrix,
                                const Index row )
      {
         //return row * matrix.rowLengths + row + 1;
         return min(row * matrix.rowLengths + row + 1, ( row + 1 ) * matrix.rowLengths );
      }

      template< typename Real,
                typename Index >
      static Index getElementStep( const EllpackSymmetric< Real, Device, Index >& matrix )
      {
         return 1;
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const EllpackSymmetric< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
          matrix.vectorProductHost( inVector, outVector );
      }

};

template< typename Real,
          typename Device,
          typename Index >
template< typename InVector,
          typename OutVector >
void EllpackSymmetric< Real, Device, Index >::vectorProductHost( const InVector& inVector,
                                                                    OutVector& outVector ) const
{
    for( Index row = 0; row < this->getRows(); row++ )
    {
        IndexType i = DeviceDependentCode::getRowBegin( *this, row );
        const IndexType rowEnd = DeviceDependentCode::getRowEnd( *this, row );
        const IndexType step = DeviceDependentCode::getElementStep( *this );

        while( i < rowEnd && this->columnIndexes[ i ] != this->getPaddingIndex() )
        {
            const IndexType column = this->columnIndexes[ i ];
            outVector[ row ] += this->values[ i ] * inVector[ column ];
            if( row != column )
                outVector[ column ] += this->values[ i ] * inVector[ row ];
            i += step;
        }
    }
};

template< typename Real,
        typename Device,
        typename Index >
template< typename Vector >
__cuda_callable__
typename Vector::RealType EllpackSymmetric< Real, Device, Index >::rowVectorProduct( const IndexType row,
                                                                                     const Vector& vector ) const
{
    IndexType i = DeviceDependentCode::getRowBegin( *this, row );
    const IndexType rowEnd = DeviceDependentCode::getRowEnd( *this, row );
    const IndexType step = DeviceDependentCode::getElementStep( *this );

    Real result = 0.0;
    while( i < rowEnd && this->columnIndexes[ i ] != this->getPaddingIndex() )
    {
        const Index column = this->columnIndexes[ i ];
        result += this->values[ i ] * vector[ column ];
        i += step;
    }
    return result;
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Device,
          typename Index >
template< typename InVector,
          typename OutVector >
__cuda_callable__
void EllpackSymmetric< Real, Device, Index >::spmvCuda( const InVector& inVector,
                                                           OutVector& outVector,
                                                           int rowId ) const
{
    IndexType i = DeviceDependentCode::getRowBegin( *this, rowId );
    const IndexType rowEnd = DeviceDependentCode::getRowEnd( *this, rowId );
    const IndexType step = DeviceDependentCode::getElementStep( *this );

    while( i < rowEnd && this->columnIndexes[ i ] != this->getPaddingIndex() )
    {
        const IndexType column = this->columnIndexes[ i ];
        outVector[ rowId ] += this->values[ i ] * inVector[ column ];
        if( rowId != column )
            outVector[ column ] += this->values[ i ] * inVector[ rowId ];
        i += step;
    }
};
#endif

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename InVector,
          typename OutVector >
__global__
void EllpackSymmetricVectorProductCuda( const EllpackSymmetric< Real, Devices::Cuda, Index >* matrix,
                                           const InVector* inVector,
                                           OutVector* outVector,
                                           const int gridIdx )
{
    int globalIdx = ( gridIdx * Devices::Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
    if( globalIdx >= matrix->getRows() )
        return;
    matrix->spmvCuda( *inVector, *outVector, globalIdx );
};
#endif

template<>
class EllpackSymmetricDeviceDependentCode< Devices::Cuda >
{
   public:

      typedef Devices::Cuda Device;

      template< typename Real,
                typename Index >
      __cuda_callable__
      static Index getRowBegin( const EllpackSymmetric< Real, Device, Index >& matrix,
                                const Index row )
      {
         return row;
      }

      template< typename Real,
                typename Index >
      __cuda_callable__
      static Index getRowEnd( const EllpackSymmetric< Real, Device, Index >& matrix,
                                const Index row )
      {
         // TODO: fix this: return row + getElementStep( matrix ) * matrix.rowLengths;
         return min( row + getElementStep( matrix ) * matrix.rowLengths, row + ( row + 1 ) * getElementStep( matrix ) );
      }

      template< typename Real,
                typename Index >
      __cuda_callable__
      static Index getElementStep( const EllpackSymmetric< Real, Device, Index >& matrix )
      {
         return matrix.alignedRows;
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const EllpackSymmetric< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
#ifdef HAVE_CUDA
          typedef EllpackSymmetric< Real, Devices::Cuda, Index > Matrix;
          typedef typename Matrix::IndexType IndexType;
          Matrix* kernel_this = Devices::Cuda::passToDevice( matrix );
          InVector* kernel_inVector = Devices::Cuda::passToDevice( inVector );
          OutVector* kernel_outVector = Devices::Cuda::passToDevice( outVector );
          dim3 cudaBlockSize( 256 ), cudaGridSize( Devices::Cuda::getMaxGridSize() );
          const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
          const IndexType cudaGrids = roundUpDivision( cudaBlocks, Devices::Cuda::getMaxGridSize() );
          for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
          {
              if( gridIdx == cudaGrids - 1 )
                  cudaGridSize.x = cudaBlocks % Devices::Cuda::getMaxGridSize();
              const int sharedMemory = cudaBlockSize.x * sizeof( Real );
              EllpackSymmetricVectorProductCuda< Real, Index, InVector, OutVector >
                                                <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
                                                  ( kernel_this,
                                                    kernel_inVector,
                                                    kernel_outVector,
                                                    gridIdx );
          }
          Devices::Cuda::freeFromDevice( kernel_this );
          Devices::Cuda::freeFromDevice( kernel_inVector );
          Devices::Cuda::freeFromDevice( kernel_outVector );
          TNL_CHECK_CUDA_DEVICE;
#endif
      }
};

} // namespace Matrices
} // namespace TNL
