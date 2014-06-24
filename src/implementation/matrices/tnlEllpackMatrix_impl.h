/***************************************************************************
                          tnlEllpackMatrix_impl.h  -  description
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

#ifndef TNLELLPACKMATRIX_IMPL_H_
#define TNLELLPACKMATRIX_IMPL_H_

#include <matrices/tnlEllpackMatrix.h>
#include <core/vectors/tnlVector.h>
#include <core/mfuncs.h>

template< typename Real,
          typename Device,
          typename Index >
tnlEllpackMatrix< Real, Device, Index > :: tnlEllpackMatrix()
: rowLengths( 0 ), alignedRows( 0 )
{
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlEllpackMatrix< Real, Device, Index > :: getType()
{
   return tnlString( "tnlEllpackMatrix< ") +
          tnlString( GetParameterType( Real( 0.0 ) ) ) +
          tnlString( ", " ) +
          Device :: getDeviceType() +
          tnlString( " >" );
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlEllpackMatrix< Real, Device, Index >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlEllpackMatrix< Real, Device, Index >::setDimensions( const IndexType rows,
                                                             const IndexType columns )
{
   tnlAssert( rows > 0 && columns > 0,
              cerr << "rows = " << rows
                   << " columns = " << columns << endl );
   this->rows = rows;
   this->columns = columns;
   if( Device::DeviceType == tnlCudaDevice )
      this->alignedRows = roundToMultiple( columns, tnlCuda::getWarpSize() );
   else this->alignedRows = rows;
   if( this->rowLengths != 0 )
      return allocateElements();
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlEllpackMatrix< Real, Device, Index >::setRowLengths( const RowLengthsVector& rowLengths )
{
   this->rowLengths = rowLengths.max();
   tnlAssert( this->rowLengths > 0,
              cerr << "this->rowLengths = " << this->rowLengths );
   if( this->rows > 0 )
      return allocateElements();
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlEllpackMatrix< Real, Device, Index >::setConstantRowLengths( const IndexType& rowLengths )
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
Index tnlEllpackMatrix< Real, Device, Index >::getRowLength( const IndexType row ) const
{
   return this->rowLengths;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlEllpackMatrix< Real, Device, Index >::setLike( const tnlEllpackMatrix< Real2, Device2, Index2 >& matrix )
{
   if( ! tnlSparseMatrix< Real, Device, Index >::setLike( matrix ) )
      return false;
   this->rowLengths = matrix.rowLengths;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlEllpackMatrix< Real, Device, Index > :: reset()
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
bool tnlEllpackMatrix< Real, Device, Index >::operator == ( const tnlEllpackMatrix< Real2, Device2, Index2 >& matrix ) const
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
bool tnlEllpackMatrix< Real, Device, Index >::operator != ( const tnlEllpackMatrix< Real2, Device2, Index2 >& matrix ) const
{
   return ! ( ( *this ) == matrix );
}

/*template< typename Real,
          typename Device,
          typename Index >
   template< typename Matrix >
bool tnlEllpackMatrix< Real, Device, Index >::copyFrom( const Matrix& matrix,
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
bool tnlEllpackMatrix< Real, Device, Index > :: setElementFast( const IndexType row,
                                                                const IndexType column,
                                                                const Real& value )
{
   return this->addElementFast( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlEllpackMatrix< Real, Device, Index > :: setElement( const IndexType row,
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
bool tnlEllpackMatrix< Real, Device, Index > :: addElementFast( const IndexType row,
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
   typedef tnlEllpackMatrixDeviceDependentCode< DeviceType > DDCType;
   IndexType i = DDCType::getRowBegin( *this, row );
   const IndexType rowEnd = DDCType::getRowEnd( *this, row );
   const IndexType step = DDCType::getElementStep( *this );

   while( i < rowEnd && this->columnIndexes[ i ] < column ) i += step;
   if( i == rowEnd )
      return false;
   if( this->columnIndexes[ i ] == column )
   {
      this->values[ i ] = thisElementMultiplicator * this->values[ i ] + value;
      return true;
   }
   else
      if( this->columnIndexes[ i ] == this->columns ) // artificial zero
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
bool tnlEllpackMatrix< Real, Device, Index > :: addElement( const IndexType row,
                                                            const IndexType column,
                                                            const RealType& value,
                                                            const RealType& thisElementMultiplicator )
{
   typedef tnlEllpackMatrixDeviceDependentCode< DeviceType > DDCType;
   IndexType i = DDCType::getRowBegin( *this, row );
   const IndexType rowEnd = DDCType::getRowEnd( *this, row );
   const IndexType step = DDCType::getElementStep( *this );

   while( i < rowEnd && this->columnIndexes.getElement( i ) < column ) i += step;
   if( i == rowEnd )
      return false;
   if( this->columnIndexes.getElement( i ) == column )
   {
      this->values.setElement( i, thisElementMultiplicator * this->values.getElement( i ) + value );
      return true;
   }
   else
      if( this->columnIndexes.getElement( i ) == this->columns )
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
bool tnlEllpackMatrix< Real, Device, Index > :: setRowFast( const IndexType row,
                                                            const IndexType* columnIndexes,
                                                            const RealType* values,
                                                            const IndexType elements )
{
   typedef tnlEllpackMatrixDeviceDependentCode< DeviceType > DDCType;
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
      this->columnIndexes[ elementPointer ] = this->getColumns();
      elementPointer += step;
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlEllpackMatrix< Real, Device, Index > :: setRow( const IndexType row,
                                                        const IndexType* columnIndexes,
                                                        const RealType* values,
                                                        const IndexType elements )
{
   typedef tnlEllpackMatrixDeviceDependentCode< DeviceType > DDCType;
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
      this->columnIndexes.setElement( elementPointer, this->getColumns() );
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
bool tnlEllpackMatrix< Real, Device, Index > :: addRowFast( const IndexType row,
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
bool tnlEllpackMatrix< Real, Device, Index > :: addRow( const IndexType row,
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
Real tnlEllpackMatrix< Real, Device, Index >::getElementFast( const IndexType row,
                                                              const IndexType column ) const
{
   typedef tnlEllpackMatrixDeviceDependentCode< DeviceType > DDCType;
   IndexType elementPtr = DDCType::getRowBegin( *this, row );
   const IndexType rowEnd = DDCType::getRowEnd( *this, row );
   const IndexType step = DDCType::getElementStep( *this );

   while( elementPtr < rowEnd && this->columnIndexes[ elementPtr ] < column ) elementPtr += step;
   if( elementPtr < rowEnd && this->columnIndexes[ elementPtr ] == column )
      return this->values[ elementPtr ];
   return 0.0;
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlEllpackMatrix< Real, Device, Index >::getElement( const IndexType row,
                                                          const IndexType column ) const
{
   typedef tnlEllpackMatrixDeviceDependentCode< DeviceType > DDCType;
   IndexType elementPtr = DDCType::getRowBegin( *this, row );
   const IndexType rowEnd = DDCType::getRowEnd( *this, row );
   const IndexType step = DDCType::getElementStep( *this );

   while( elementPtr < rowEnd && this->columnIndexes.getElement( elementPtr ) < column ) elementPtr += step;
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
void tnlEllpackMatrix< Real, Device, Index >::getRowFast( const IndexType row,
                                                          IndexType* columns,
                                                          RealType* values ) const
{
   typedef tnlEllpackMatrixDeviceDependentCode< DeviceType > DDCType;
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
void tnlEllpackMatrix< Real, Device, Index >::getRow( const IndexType row,
                                                      IndexType* columns,
                                                      RealType* values ) const
{
   typedef tnlEllpackMatrixDeviceDependentCode< DeviceType > DDCType;
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
  template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
typename Vector::RealType tnlEllpackMatrix< Real, Device, Index >::rowVectorProduct( const IndexType row,
                                                                                     const Vector& vector ) const
{
   IndexType i = DeviceDependentCode::getRowBegin( *this, row );
   const IndexType rowEnd = DeviceDependentCode::getRowEnd( *this, row );
   const IndexType step = DeviceDependentCode::getElementStep( *this );

   Real result = 0.0;
   while( i < rowEnd && this->columnIndexes[ i ] < this->columns )
   {
      const Index column = this->columnIndexes[ i ];
      result += this->values[ i ] * vector[ column ];
      i += step;
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename InVector,
             typename OutVector >
void tnlEllpackMatrix< Real, Device, Index >::vectorProduct( const InVector& inVector,
                                                                   OutVector& outVector ) const
{
   DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Index2 >
void tnlEllpackMatrix< Real, Device, Index > :: addMatrix( const tnlEllpackMatrix< Real2, Device, Index2 >& matrix,
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
void tnlEllpackMatrix< Real, Device, Index >::getTransposition( const tnlEllpackMatrix< Real2, Device, Index2 >& matrix,
                                                                      const RealType& matrixMultiplicator )
{
   tnlAssert( false, cerr << "TODO: implement" );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
bool tnlEllpackMatrix< Real, Device, Index > :: performSORIteration( const Vector& b,
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
bool tnlEllpackMatrix< Real, Device, Index >::save( tnlFile& file ) const
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
bool tnlEllpackMatrix< Real, Device, Index >::load( tnlFile& file )
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
bool tnlEllpackMatrix< Real, Device, Index >::save( const tnlString& fileName ) const
{
   return tnlObject::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlEllpackMatrix< Real, Device, Index >::load( const tnlString& fileName )
{
   return tnlObject::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlEllpackMatrix< Real, Device, Index >::print( ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      IndexType i( row * this->rowLengths );
      const IndexType rowEnd( i + this->rowLengths );
      while( i < rowEnd && this->columnIndexes.getElement( i ) < this->columns )
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
bool tnlEllpackMatrix< Real, Device, Index >::allocateElements()
{
   if( ! tnlSparseMatrix< Real, Device, Index >::allocateMatrixElements( this->alignedRows * this->rowLengths ) )
      return false;
   return true;
}

template<>
class tnlEllpackMatrixDeviceDependentCode< tnlHost >
{
   public:

      typedef tnlHost Device;

      template< typename Real,
                typename Index >
      static Index getRowBegin( const tnlEllpackMatrix< Real, Device, Index >& matrix,
                                const Index row )
      {
         return row * matrix.rowLengths;
      }

      template< typename Real,
                typename Index >
      static Index getRowEnd( const tnlEllpackMatrix< Real, Device, Index >& matrix,
                                const Index row )
      {
         return ( row + 1 ) * matrix.rowLengths;
      }

      template< typename Real,
                typename Index >
      static Index getElementStep( const tnlEllpackMatrix< Real, Device, Index >& matrix )
      {
         return 1;
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const tnlEllpackMatrix< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         for( Index row = 0; row < matrix.getRows(); row ++ )
            outVector[ row ] = matrix.rowVectorProduct( row, inVector );
      }
};

template<>
class tnlEllpackMatrixDeviceDependentCode< tnlCuda >
{
   public:

      typedef tnlCuda Device;

      template< typename Real,
                typename Index >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      static Index getRowBegin( const tnlEllpackMatrix< Real, Device, Index >& matrix,
                                const Index row )
      {
         return row;
      }

      template< typename Real,
                typename Index >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      static Index getRowEnd( const tnlEllpackMatrix< Real, Device, Index >& matrix,
                                const Index row )
      {
         return row + getElementStep( matrix ) * matrix.rowLengths;
      }

      template< typename Real,
                typename Index >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      static Index getElementStep( const tnlEllpackMatrix< Real, Device, Index >& matrix )
      {
         return matrix.alignedRows;
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const tnlEllpackMatrix< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         tnlMatrixVectorProductCuda( matrix, inVector, outVector );
      }
};




#endif /* TNLELLPACKMATRIX_IMPL_H_ */
