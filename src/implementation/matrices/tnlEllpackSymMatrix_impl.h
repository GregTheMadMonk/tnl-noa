/***************************************************************************
                          tnlEllpackSymMatrix_impl.h  -  description
                             -------------------
    begin                : Dec 7, 2013
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

#ifndef TNLELLPACKSYMMATRIX_IMPL_H_
#define TNLELLPACKSYMMATRIX_IMPL_H_

#include <matrices/tnlEllpackSymMatrix.h>
#include <core/vectors/tnlVector.h>
#include <core/mfuncs.h>

template< typename Real,
          typename Device,
          typename Index >
tnlEllpackSymMatrix< Real, Device, Index > :: tnlEllpackSymMatrix()
: rowLengths( 0 ), alignedRows( 0 )
{
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlEllpackSymMatrix< Real, Device, Index > :: getType()
{
   return tnlString( "tnlEllpackSymMatrix< ") +
          tnlString( ::getType< Real >() ) +
          tnlString( ", " ) +
          Device :: getDeviceType() +
          tnlString( " >" );
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlEllpackSymMatrix< Real, Device, Index >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlEllpackSymMatrix< Real, Device, Index >::setDimensions( const IndexType rows,
                                                             const IndexType columns )
{
   tnlAssert( rows > 0 && columns > 0,
              cerr << "rows = " << rows
                   << " columns = " << columns << endl );
   this->rows = rows;
   this->columns = columns;   
   if( Device::DeviceType == ( int ) tnlCudaDevice )
      this->alignedRows = roundToMultiple( columns, tnlCuda::getWarpSize() );
   else this->alignedRows = rows;
   if( this->rowLengths != 0 )
      return allocateElements();
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlEllpackSymMatrix< Real, Device, Index >::setRowLengths( const RowLengthsVector& rowLengths )
{
   tnlAssert( this->getRows() > 0, );
   tnlAssert( this->getColumns() > 0, );
   //tnlAssert( this->rowLengths > 0,
   //           cerr << "this->rowLengths = " << this->rowLengths );
   this->rowLengths = this->maxRowLength = rowLengths.max();
   return allocateElements();
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlEllpackSymMatrix< Real, Device, Index >::setConstantRowLengths( const IndexType& rowLengths )
{
   tnlAssert( rowLengths > 0,
              cerr << " rowLengths = " << rowLengths );
   this->rowLengths = rowLengths;
   if( this->rows > 0 )
      return allocateElements();
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlEllpackSymMatrix< Real, Device, Index >::getRowLength( const IndexType row ) const
{
   return this->rowLengths;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlEllpackSymMatrix< Real, Device, Index >::setLike( const tnlEllpackSymMatrix< Real2, Device2, Index2 >& matrix )
{
   if( ! tnlSparseMatrix< Real, Device, Index >::setLike( matrix ) )
      return false;
   this->rowLengths = matrix.rowLengths;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlEllpackSymMatrix< Real, Device, Index > :: reset()
{
   tnlSparseMatrix< Real, Device, Index >::reset();
   this->rowLengths = 0;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlEllpackSymMatrix< Real, Device, Index >::operator == ( const tnlEllpackSymMatrix< Real2, Device2, Index2 >& matrix ) const
{
   tnlAssert( this->getRows() == matrix.getRows() &&
              this->getColumns() == matrix.getColumns(),
              cerr << "this->getRows() = " << this->getRows()
                   << " matrix.getRows() = " << matrix.getRows()
                   << " this->getColumns() = " << this->getColumns()
                   << " matrix.getColumns() = " << matrix.getColumns()
                   << " this->getName() = " << this->getName()
                   << " matrix.getName() = " << matrix.getName() );
   // TODO: implement this
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlEllpackSymMatrix< Real, Device, Index >::operator != ( const tnlEllpackSymMatrix< Real2, Device2, Index2 >& matrix ) const
{
   return ! ( ( *this ) == matrix );
}

/*template< typename Real,
          typename Device,
          typename Index >
   template< typename Matrix >
bool tnlEllpackSymMatrix< Real, Device, Index >::copyFrom( const Matrix& matrix,
                                                        const RowLengthsVector& rowLengths )
{
   return tnlMatrix< RealType, DeviceType, IndexType >::copyFrom( matrix, rowLengths );
}*/

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlEllpackSymMatrix< Real, Device, Index > :: setElementFast( const IndexType row,
                                                                const IndexType column,
                                                                const Real& value )
{
   return this->addElementFast( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlEllpackSymMatrix< Real, Device, Index > :: setElement( const IndexType row,
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
bool tnlEllpackSymMatrix< Real, Device, Index > :: addElementFast( const IndexType row,
                                                                const IndexType column,
                                                                const RealType& value,
                                                                const RealType& thisElementMultiplicator )
{
   // TODO: return this back when CUDA kernels support cerr
   /*tnlAssert( row >= 0 && row < this->rows &&
              column >= 0 && column <= this->rows,
              cerr << " row = " << row
                   << " column = " << column
                   << " this->rows = " << this->rows
                   << " this->columns = " << this-> columns );*/
   typedef tnlEllpackSymMatrixDeviceDependentCode< DeviceType > DDCType;
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
bool tnlEllpackSymMatrix< Real, Device, Index > :: addElement( const IndexType row,
                                                            const IndexType column,
                                                            const RealType& value,
                                                            const RealType& thisElementMultiplicator )
{
   typedef tnlEllpackSymMatrixDeviceDependentCode< DeviceType > DDCType;
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
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlEllpackSymMatrix< Real, Device, Index > :: setRowFast( const IndexType row,
                                                            const IndexType* columnIndexes,
                                                            const RealType* values,
                                                            const IndexType elements )
{
   typedef tnlEllpackSymMatrixDeviceDependentCode< DeviceType > DDCType;
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
bool tnlEllpackSymMatrix< Real, Device, Index > :: setRow( const IndexType row,
                                                        const IndexType* columnIndexes,
                                                        const RealType* values,
                                                        const IndexType elements )
{
   typedef tnlEllpackSymMatrixDeviceDependentCode< DeviceType > DDCType;
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
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlEllpackSymMatrix< Real, Device, Index > :: addRowFast( const IndexType row,
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
bool tnlEllpackSymMatrix< Real, Device, Index > :: addRow( const IndexType row,
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
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Real tnlEllpackSymMatrix< Real, Device, Index >::getElementFast( const IndexType row,
                                                              const IndexType column ) const
{
   if( row < column )
       return this->getElementFast( column, row );

   typedef tnlEllpackSymMatrixDeviceDependentCode< DeviceType > DDCType;
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
Real tnlEllpackSymMatrix< Real, Device, Index >::getElement( const IndexType row,
                                                          const IndexType column ) const
{
   if( row < column )
       return this->getElement( column, row );

   typedef tnlEllpackSymMatrixDeviceDependentCode< DeviceType > DDCType;
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
#ifdef HAVE_CUDA
   __device__ __host__
#endif
void tnlEllpackSymMatrix< Real, Device, Index >::getRowFast( const IndexType row,
                                                          IndexType* columns,
                                                          RealType* values ) const
{
   typedef tnlEllpackSymMatrixDeviceDependentCode< DeviceType > DDCType;
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
void tnlEllpackSymMatrix< Real, Device, Index >::getRow( const IndexType row,
                                                      IndexType* columns,
                                                      RealType* values ) const
{
   typedef tnlEllpackSymMatrixDeviceDependentCode< DeviceType > DDCType;
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
void tnlEllpackSymMatrix< Real, Device, Index >::vectorProduct( const InVector& inVector,
                                                                   OutVector& outVector ) const
{
   DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Index2 >
void tnlEllpackSymMatrix< Real, Device, Index > :: addMatrix( const tnlEllpackSymMatrix< Real2, Device, Index2 >& matrix,
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
void tnlEllpackSymMatrix< Real, Device, Index >::getTransposition( const tnlEllpackSymMatrix< Real2, Device, Index2 >& matrix,
                                                                      const RealType& matrixMultiplicator )
{
   tnlAssert( false, cerr << "TODO: implement" );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
bool tnlEllpackSymMatrix< Real, Device, Index > :: performSORIteration( const Vector& b,
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
      cerr << "There is zero on the diagonal in " << row << "-th row of thge matrix " << this->getName() << ". I cannot perform SOR iteration." << endl;
      return false;
   }
   x. setElement( row, x[ row ] + omega / diagonalValue * ( b[ row ] - sum ) );
   return true;
}


template< typename Real,
          typename Device,
          typename Index >
bool tnlEllpackSymMatrix< Real, Device, Index >::save( tnlFile& file ) const
{
   if( ! tnlSparseMatrix< Real, Device, Index >::save( file) ) return false;
#ifdef HAVE_NOT_CXX11
   if( ! file.write< IndexType, tnlHost, IndexType >( &this->rowLengths, 1 ) ) return false;
#else      
   if( ! file.write( &this->rowLengths ) ) return false;
#endif   
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlEllpackSymMatrix< Real, Device, Index >::load( tnlFile& file )
{
   if( ! tnlSparseMatrix< Real, Device, Index >::load( file) ) return false;
#ifdef HAVE_NOT_CXX11
   if( ! file.read< IndexType, tnlHost, IndexType >( &this->rowLengths, 1 ) ) return false;
#else   
   if( ! file.read( &this->rowLengths ) ) return false;
#endif   
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlEllpackSymMatrix< Real, Device, Index >::save( const tnlString& fileName ) const
{
   return tnlObject::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlEllpackSymMatrix< Real, Device, Index >::load( const tnlString& fileName )
{
   return tnlObject::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlEllpackSymMatrix< Real, Device, Index >::print( ostream& str ) const
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
      str << endl;
   }
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlEllpackSymMatrix< Real, Device, Index >::allocateElements()
{
   if( ! tnlSparseMatrix< Real, Device, Index >::allocateMatrixElements( this->alignedRows * this->rowLengths ) )
      return false;
   return true;
}

template<>
class tnlEllpackSymMatrixDeviceDependentCode< tnlHost >
{
   public:

      typedef tnlHost Device;

      template< typename Real,
                typename Index >
      static Index getRowBegin( const tnlEllpackSymMatrix< Real, Device, Index >& matrix,
                                const Index row )
      {
         return row * matrix.rowLengths;
      }

      template< typename Real,
                typename Index >
      static Index getRowEnd( const tnlEllpackSymMatrix< Real, Device, Index >& matrix,
                                const Index row )
      {
         //return row * matrix.rowLengths + row + 1;
         return min(row * matrix.rowLengths + row + 1, ( row + 1 ) * matrix.rowLengths );
      }

      template< typename Real,
                typename Index >
      static Index getElementStep( const tnlEllpackSymMatrix< Real, Device, Index >& matrix )
      {
         return 1;
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const tnlEllpackSymMatrix< Real, Device, Index >& matrix,
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
void tnlEllpackSymMatrix< Real, Device, Index >::vectorProductHost( const InVector& inVector,
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
#ifdef HAVE_CUDA
__device__ __host__
#endif
typename Vector::RealType tnlEllpackSymMatrix< Real, Device, Index >::rowVectorProduct( const IndexType row,
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
__device__
void tnlEllpackSymMatrix< Real, Device, Index >::spmvCuda( const InVector& inVector,
                                                           OutVector& outVector,
                                                           int rowId ) const
{
    IndexType i = DeviceDependentCode::getRowBegin( *this, rowId );
    const IndexType rowEnd = DeviceDependentCode::getRowEnd( *this, rowId );
    const IndexTpe step = DeviceDependentCode::getElementStep( *this );

    while( i < rowEnd && this->columnIndexes.getElement( i ) != this->getPaddingIndex() )
    {
        const IndexType column = this->columnIndexes.getElemnt( i );
        outVector[ rowId ] += this->values.getElement( i ) * inVector[ column ];
        if( rowId != column )
            outVector[ column ].add( this->values.getElement( i ) * inVector[ row ] );
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
tnlEllpackSymMatrixVectorProductCuda< Real, tnlCuda, Index >( const tnlEllpackSymMatrix& matrix,
                                                              const InVector& inVector,
                                                              OutVector& outVector,
                                                              const int gridIdx )
{
    int globalIdx = ( gridIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
    if( globalIdx >= matrix->getRows() )
        return;
    matrix->spmvCuda( inVector, outVector, globalIdx );
};
#endif

template<>
class tnlEllpackSymMatrixDeviceDependentCode< tnlCuda >
{
   public:

      typedef tnlCuda Device;

      template< typename Real,
                typename Index >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      static Index getRowBegin( const tnlEllpackSymMatrix< Real, Device, Index >& matrix,
                                const Index row )
      {
         return row;
      }

      template< typename Real,
                typename Index >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      static Index getRowEnd( const tnlEllpackSymMatrix< Real, Device, Index >& matrix,
                                const Index row )
      {
         // TODO: fix this: return row + getElementStep( matrix ) * matrix.rowLengths;
         return min( row + getElementStep( matrix ) * matrix.rowLengths, row + ( row + 1 ) * getElementStep( matrix ) );
      }

      template< typename Real,
                typename Index >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      static Index getElementStep( const tnlEllpackSymMatrix< Real, Device, Index >& matrix )
      {
         return matrix.alignedRows;
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const tnlEllpackSymMatrix< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
#ifdef HAVE_CUDA
          typedef tnlEllpackSymMatrix< Real, tnlCuda, Index > Matrix;
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
              tnlEllpackSymMatrixVectorProductCuda< Real, Index, StripSize, InVector, OutVector >
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
};




#endif /* TNLELLPACKMATRIX_IMPL_H_ */
