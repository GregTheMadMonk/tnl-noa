/***************************************************************************
                          tnlMultidiagonalMatrix.h  -  description
                             -------------------
    begin                : Dec 4, 2013
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

#ifndef TNLMULTIDIAGONALMATRIX_IMPL_H_
#define TNLMULTIDIAGONALMATRIX_IMPL_H_

#include <matrices/tnlMultidiagonalMatrix.h>
#include <core/vectors/tnlVector.h>
#include <core/mfuncs.h>

template< typename Device >
class tnlMultidiagonalMatrixDeviceDependentCode;

template< typename Real,
          typename Device,
          typename Index >
tnlMultidiagonalMatrix< Real, Device, Index > :: tnlMultidiagonalMatrix()
{
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlMultidiagonalMatrix< Real, Device, Index > :: getType()
{
   return tnlString( "tnlMultidiagonalMatrix< ") +
          tnlString( ::getType< Real >() ) +
          tnlString( ", " ) +
          Device :: getDeviceType() +
          tnlString( " >" );
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlMultidiagonalMatrix< Real, Device, Index >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMultidiagonalMatrix< Real, Device, Index >::setDimensions( const IndexType rows,
                                                                   const IndexType columns )
{
   tnlAssert( rows > 0 && columns > 0,
              cerr << "rows = " << rows
                   << " columns = " << columns << endl );
   if( ! tnlMatrix< Real, Device, Index >::setDimensions( rows, columns ) )
      return false;
   if( this->diagonalsShift.getSize() != 0 )
   {
      if( ! this->values.setSize( Min( this->rows, this->columns ) * this->diagonalsShift.getSize() ) )
         return false;
      this->values.setValue( 0.0 );
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMultidiagonalMatrix< Real, Device, Index >::setCompressedRowsLengths( const CompressedRowsLengthsVector& rowLengths )
{
   /****
    * TODO: implement some check here similar to the one in the tridiagonal matrix
    */
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlMultidiagonalMatrix< Real, Device, Index >::getRowLength( const IndexType row ) const
{
   IndexType rowLength( 0 );
   for( IndexType i = 0; i < diagonalsShift.getSize(); i++ )
   {
      const IndexType column = row + diagonalsShift.getElement( i );
      if( column >= 0 && column < this->getColumns() )
         rowLength++;
   }
   return rowLength;
}

template< typename Real,
          typename Device,
          typename Index >
Index
tnlMultidiagonalMatrix< Real, Device, Index >::
getMaxRowLength() const
{
   return diagonalsShift.getSize();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
bool tnlMultidiagonalMatrix< Real, Device, Index > :: setDiagonals(  const Vector& diagonals )
{
   tnlAssert( diagonals.getSize() > 0,
              cerr << "New number of diagonals = " << diagonals.getSize() << endl );
   this->diagonalsShift.setLike( diagonals );
   this->diagonalsShift = diagonals;
   if( this->rows != 0 && this->columns != 0 )
   {
      if( ! this->values.setSize( Min( this->rows, this->columns ) * this->diagonalsShift.getSize() ) )
         return false;
      this->values.setValue( 0.0 );
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
const tnlVector< Index, Device, Index >& tnlMultidiagonalMatrix< Real, Device, Index > :: getDiagonals() const
{
   return this->diagonalsShift;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlMultidiagonalMatrix< Real, Device, Index > :: setLike( const tnlMultidiagonalMatrix< Real2, Device2, Index2 >& matrix )
{
   if( ! this->setDimensions( matrix.getRows(), matrix.getColumns() ) )
      return false;
   if( ! setDiagonals( matrix.getDiagonals() ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlMultidiagonalMatrix< Real, Device, Index > :: getNumberOfMatrixElements() const
{
   return this->values.getSize();
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlMultidiagonalMatrix< Real, Device, Index > :: getNumberOfNonzeroMatrixElements() const
{
   IndexType nonzeroElements;
   for( IndexType i = 0; i < this->values.getSize(); i++ )
      if( this->values.getElement( i ) != 0 )
         nonzeroElements++;
   return nonzeroElements;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlMultidiagonalMatrix< Real, Device, Index > :: reset()
{
   this->rows = 0;
   this->columns = 0;
   this->values.reset();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlMultidiagonalMatrix< Real, Device, Index >::operator == ( const tnlMultidiagonalMatrix< Real2, Device2, Index2 >& matrix ) const
{
   tnlAssert( this->getRows() == matrix.getRows() &&
              this->getColumns() == matrix.getColumns(),
              cerr << "this->getRows() = " << this->getRows()
                   << " matrix.getRows() = " << matrix.getRows()
                   << " this->getColumns() = " << this->getColumns()
                   << " matrix.getColumns() = " << matrix.getColumns() );
   return ( this->diagonals == matrix.diagonals &&
            this->values == matrix.values );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlMultidiagonalMatrix< Real, Device, Index >::operator != ( const tnlMultidiagonalMatrix< Real2, Device2, Index2 >& matrix ) const
{
   return ! ( ( *this ) == matrix );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlMultidiagonalMatrix< Real, Device, Index >::setValue( const RealType& v )
{
   this->values.setValue( v );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool tnlMultidiagonalMatrix< Real, Device, Index > :: setElementFast( const IndexType row,
                                                                      const IndexType column,
                                                                      const Real& value )
{
   IndexType index;
   if( ! this->getElementIndexFast( row, column, index  ) )
      return false;
   this->values[ index ] = value;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMultidiagonalMatrix< Real, Device, Index > :: setElement( const IndexType row,
                                                                  const IndexType column,
                                                                  const Real& value )
{
   IndexType index;
   if( ! this->getElementIndex( row, column, index  ) )
      return false;
   this->values.setElement( index, value );
   return true;
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool tnlMultidiagonalMatrix< Real, Device, Index > :: addElementFast( const IndexType row,
                                                                      const IndexType column,
                                                                      const RealType& value,
                                                                      const RealType& thisElementMultiplicator )
{
   Index index;
   if( ! this->getElementIndexFast( row, column, index  ) )
      return false;
   RealType& aux = this->values[ index ];
   aux = thisElementMultiplicator * aux + value;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMultidiagonalMatrix< Real, Device, Index > :: addElement( const IndexType row,
                                                                  const IndexType column,
                                                                  const RealType& value,
                                                                  const RealType& thisElementMultiplicator )
{
   Index index;
   if( ! this->getElementIndex( row, column, index  ) )
      return false;
   this->values.setElement( index, thisElementMultiplicator * this->values.getElement( index ) + value );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool tnlMultidiagonalMatrix< Real, Device, Index > :: setRowFast( const IndexType row,
                                                                  const IndexType* columns,
                                                                  const RealType* values,
                                                                  const IndexType numberOfElements )
{
   return this->addRowFast( row, columns, values, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMultidiagonalMatrix< Real, Device, Index > :: setRow( const IndexType row,
                                                              const Index* columns,
                                                              const Real* values,
                                                              const Index numberOfElements )
{
   return this->addRow( row, columns, values, numberOfElements, 0.0 );
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool tnlMultidiagonalMatrix< Real, Device, Index > :: addRowFast( const IndexType row,
                                                                  const IndexType* columns,
                                                                  const RealType* values,
                                                                  const IndexType numberOfElements,
                                                                  const RealType& thisElementMultiplicator )
{
   if( this->diagonalsShift.getSize() < numberOfElements )
      return false;
   typedef tnlMultidiagonalMatrixDeviceDependentCode< Device > DDCType;
   const IndexType elements = Min( this->diagonalsShift.getSize(), numberOfElements );
   IndexType i( 0 );
   while( i < elements )
   {
      const IndexType index = DDCType::getElementIndex( this->getRows(), this->diagonalsShift.getSize(), row, i );
      RealType& aux = this->values[ index ];
      aux = thisElementMultiplicator * aux + values[ i ];
      i++;
   }
   while( i < this->diagonalsShift.getSize() )
   {
      const IndexType index = DDCType::getElementIndex( this->getRows(), this->diagonalsShift.getSize(), row, i );
      this->values[ index ] = 0;
      i++;
   }
   return true;

}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMultidiagonalMatrix< Real, Device, Index > :: addRow( const IndexType row,
                                                              const Index* columns,
                                                              const Real* values,
                                                              const Index numberOfElements,
                                                              const RealType& thisElementMultiplicator )
{
   if( this->diagonalsShift.getSize() < numberOfElements )
      return false;
   typedef tnlMultidiagonalMatrixDeviceDependentCode< Device > DDCType;
   const IndexType elements = Min( this->diagonalsShift.getSize(), numberOfElements );
   IndexType i( 0 );
   while( i < elements )
   {
      const IndexType index = DDCType::getElementIndex( this->getRows(), this->diagonalsShift.getSize(), row, i );
      if( thisElementMultiplicator == 0.0 )
         this->values.setElement( index, values[ i ] );
      else
         this->values.setElement( index, thisElementMultiplicator * this->values.getElement( index ) + values[ i ] );
      i++;
   }
   while( i < this->diagonalsShift.getSize() )
   {
      const IndexType index = DDCType::getElementIndex( this->getRows(), this->diagonalsShift.getSize(), row, i );
      this->values.setElement( index, 0 );
      i++;
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Real tnlMultidiagonalMatrix< Real, Device, Index >::getElementFast( const IndexType row,
                                                                    const IndexType column ) const
{
   Index index;
   if( ! this->getElementIndexFast( row, column, index  ) )
      return 0.0;
   return this->values[ index ];
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlMultidiagonalMatrix< Real, Device, Index >::getElement( const IndexType row,
                                                                const IndexType column ) const
{
   Index index;
   if( ! this->getElementIndex( row, column, index  ) )
      return 0.0;
   return this->values.getElement( index );
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
void tnlMultidiagonalMatrix< Real, Device, Index >::getRowFast( const IndexType row,
                                                                IndexType* columns,
                                                                RealType* values ) const
{
   IndexType pointer( 0 );
   for( IndexType i = 0; i < diagonalsShift.getSize(); i++ )
   {
      const IndexType column = row + diagonalsShift[ i ];
      if( column >= 0 && column < this->getColumns() )
      {
         columns[ pointer ] = column;
         values[ pointer ] = this->getElementFast( row, column );
         pointer++;
      }
   }
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename tnlMultidiagonalMatrix< Real, Device, Index >::MatrixRow
tnlMultidiagonalMatrix< Real, Device, Index >::
getRow( const IndexType rowIndex )
{
   IndexType firstRowElement( 0 );
   while( rowIndex + this->diagonalsShift[ firstRowElement ] < 0 )
      firstRowElement ++;

   IndexType firstRowElementIndex;
   this->getElementIndexFast( rowIndex, rowIndex + this->diagonalsShift[ firstRowElement ], firstRowElementIndex );
   if( Device::getDevice() == tnlHostDevice )
      return MatrixRow( &this->values.getData()[ firstRowElementIndex ],
                        &this->diagonalsShift.getData()[ firstRowElement ],
                        this->diagonalsShift.getSize() - firstRowElement,
                        rowIndex,
                        this->getColumns(),
                        1 );
   if( Device::getDevice() == tnlCudaDevice )
      return MatrixRow( &this->values.getData()[ firstRowElementIndex ],
                        &this->diagonalsShift.getData()[ firstRowElement ],
                        this->diagonalsShift.getSize()- firstRowElement,
                        rowIndex,
                        this->getColumns(),
                        this->rows );

}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const typename tnlMultidiagonalMatrix< Real, Device, Index >::MatrixRow
tnlMultidiagonalMatrix< Real, Device, Index >::
getRow( const IndexType rowIndex ) const
{
   IndexType firstRowElement( 0 );
   while( rowIndex + this->diagonalsShift[ firstRowElement ] < 0 )
      firstRowElement ++;

   IndexType firstRowElementIndex;
   this->getElementIndexFast( rowIndex, rowIndex + this->diagonalsShift[ firstRowElement ], firstRowElementIndex );
   if( Device::getDevice() == tnlHostDevice )
      return MatrixRow( &this->values.getData()[ firstRowElementIndex ],
                        &this->diagonalsShift.getData()[ firstRowElement ],
                        this->diagonalsShift.getSize() - firstRowElement,
                        rowIndex,
                        this->getColumns(),
                        1 );
   if( Device::getDevice() == tnlCudaDevice )
      return MatrixRow( &this->values.getData()[ firstRowElementIndex ],
                        &this->diagonalsShift.getData()[ firstRowElement ],
                        this->diagonalsShift.getSize()- firstRowElement,
                        rowIndex,
                        this->getColumns(),
                        this->rows );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
__cuda_callable__
typename Vector::RealType tnlMultidiagonalMatrix< Real, Device, Index >::rowVectorProduct( const IndexType row,
                                                                                           const Vector& vector ) const
{
   typedef tnlMultidiagonalMatrixDeviceDependentCode< Device > DDCType;
   Real result = 0.0;
   for( Index i = 0;
        i < this->diagonalsShift.getSize();
        i ++ )
   {
      const Index column = row + this->diagonalsShift[ i ];
      if( column >= 0 && column < this->getColumns() )
         result += this->values[
                      DDCType::getElementIndex( this->getRows(),
                                                this->diagonalsShift.getSize(),
                                                row,
                                                i ) ] * vector[ column ];
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename InVector,
             typename OutVector >
void tnlMultidiagonalMatrix< Real, Device, Index >::vectorProduct( const InVector& inVector,
                                                                   OutVector& outVector ) const
{
   tnlAssert( this->getColumns() == inVector.getSize(),
            cerr << "Matrix columns: " << this->getColumns() << endl
                 << "Vector size: " << inVector.getSize() << endl );
   tnlAssert( this->getRows() == outVector.getSize(),
               cerr << "Matrix rows: " << this->getRows() << endl
                    << "Vector size: " << outVector.getSize() << endl );

   DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Index2 >
void tnlMultidiagonalMatrix< Real, Device, Index > :: addMatrix( const tnlMultidiagonalMatrix< Real2, Device, Index2 >& matrix,
                                                                 const RealType& matrixMultiplicator,
                                                                 const RealType& thisMatrixMultiplicator )
{
   tnlAssert( false, cerr << "TODO: implement" );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Index2 >
void tnlMultidiagonalMatrix< Real, Device, Index >::getTransposition( const tnlMultidiagonalMatrix< Real2, Device, Index2 >& matrix,
                                                                      const RealType& matrixMultiplicator )
{
   tnlVector< Index > auxDiagonals;
   auxDiagonals.setLike( matrix.getDiagonals() );
   const Index numberOfDiagonals = matrix.getDiagonals().getSize();
   for( Index i = 0; i < numberOfDiagonals; i++ )
      auxDiagonals[ i ] = -1.0 * matrix.getDiagonals().getElement( numberOfDiagonals - i - 1 );
   this->setDimensions( matrix.getColumns(),
                        matrix.getRows() );
   this->setDiagonals( auxDiagonals );
   for( Index row = 0; row < matrix.getRows(); row++ )
      for( Index diagonal = 0; diagonal < numberOfDiagonals; diagonal++ )
      {
         const Index column = row + matrix.getDiagonals().getElement( diagonal );
         if( column >= 0 && column < matrix.getColumns() )
            this->setElement( column, row, matrixMultiplicator * matrix.getElement( row, column ) );
      }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
bool tnlMultidiagonalMatrix< Real, Device, Index > :: performSORIteration( const Vector& b,
                                                                           const IndexType row,
                                                                           Vector& x,
                                                                           const RealType& omega ) const
{
   tnlAssert( row >=0 && row < this->getRows(),
              cerr << "row = " << row
                   << " this->getRows() = " << this->getRows() << endl );

   RealType diagonalValue( 0.0 );
   RealType sum( 0.0 );

   const IndexType maxRowLength = this->getMaxRowLength();

   for( IndexType i = 0; i < maxRowLength; i++ )
   {
      const IndexType column = row + this->diagonalsShift[ i ];
      if( column >= 0 && column < this->getColumns() )
      {
         IndexType elementIndex;
         this->getElementIndex( row, column, elementIndex );
         if( column == row )
            diagonalValue = this->values[ elementIndex ];
         else
            sum += this->values[ elementIndex ] * x[ column ];
      }
   }
   if( diagonalValue == ( Real ) 0.0 )
   {
      cerr << "There is zero on the diagonal in " << row << "-th row of thge matrix. I cannot perform SOR iteration." << endl;
      return false;
   }
   x[ row ] = ( 1.0 - omega ) * x[ row ] + omega / diagonalValue * ( b[ row ] - sum );
   return true;
}


template< typename Real,
          typename Device,
          typename Index >
bool tnlMultidiagonalMatrix< Real, Device, Index >::save( tnlFile& file ) const
{
   if( ! tnlMatrix< Real, Device, Index >::save( file ) ) return false;
   if( ! this->values.save( file ) ) return false;
   if( ! this->diagonalsShift.save( file ) ) return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMultidiagonalMatrix< Real, Device, Index >::load( tnlFile& file )
{
   if( ! tnlMatrix< Real, Device, Index >::load( file ) ) return false;
   if( ! this->values.load( file ) ) return false;
   if( ! this->diagonalsShift.load( file ) ) return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMultidiagonalMatrix< Real, Device, Index >::save( const tnlString& fileName ) const
{
   return tnlObject::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMultidiagonalMatrix< Real, Device, Index >::load( const tnlString& fileName )
{
   return tnlObject::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlMultidiagonalMatrix< Real, Device, Index >::print( ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      for( IndexType i = 0; i < this->diagonalsShift.getSize(); i++ )
      {
         const IndexType column = row + diagonalsShift.getElement( i );
         if( column >=0 && column < this->columns )
            str << " Col:" << column << "->" << this->getElement( row, column ) << "\t";
      }
      str << endl;
   }
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMultidiagonalMatrix< Real, Device, Index >::getElementIndex( const IndexType row,
                                                                     const IndexType column,
                                                                     Index& index ) const
{
   tnlAssert( row >=0 && row < this->rows,
            cerr << "row = " << row
                 << " this->rows = " << this->rows << endl );
   tnlAssert( column >=0 && column < this->columns,
            cerr << "column = " << column
                 << " this->columns = " << this->columns << endl );

   typedef tnlMultidiagonalMatrixDeviceDependentCode< Device > DDCType;
   IndexType i( 0 );
   while( i < this->diagonalsShift.getSize() )
   {
      if( diagonalsShift.getElement( i ) == column - row )
      {
         index = DDCType::getElementIndex( this->getRows(), this->diagonalsShift.getSize(), row, i );
         return true;
      }
      i++;
   }
   return false;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool tnlMultidiagonalMatrix< Real, Device, Index >::getElementIndexFast( const IndexType row,
                                                                         const IndexType column,
                                                                         Index& index ) const
{
   tnlAssert( row >=0 && row < this->rows,
            cerr << "row = " << row
                 << " this->rows = " << this->rows << endl );
   tnlAssert( column >=0 && column < this->columns,
            cerr << "column = " << column
                 << " this->columns = " << this->columns << endl );

   typedef tnlMultidiagonalMatrixDeviceDependentCode< Device > DDCType;
   IndexType i( 0 );
   while( i < this->diagonalsShift.getSize() )
   {
      if( diagonalsShift[ i ] == column - row )
      {
         index = DDCType::getElementIndex( this->getRows(), this->diagonalsShift.getSize(), row, i );
         return true;
      }
      i++;
   }
   return false;
}

template<>
class tnlMultidiagonalMatrixDeviceDependentCode< tnlHost >
{
   public:

      typedef tnlHost Device;

      template< typename Index >
      __cuda_callable__
      static Index getElementIndex( const Index rows,
                                    const Index diagonals,
                                    const Index row,
                                    const Index diagonal )
      {
         return row*diagonals + diagonal;
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const tnlMultidiagonalMatrix< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
#ifdef HAVE_OPENMP
#pragma omp parallel for if( tnlOmp::isEnabled() )
#endif           
         for( Index row = 0; row < matrix.getRows(); row ++ )
            outVector[ row ] = matrix.rowVectorProduct( row, inVector );
      }
};

template<>
class tnlMultidiagonalMatrixDeviceDependentCode< tnlCuda >
{
   public:

      typedef tnlCuda Device;

      template< typename Index >
      __cuda_callable__
      static Index getElementIndex( const Index rows,
                                    const Index diagonals,
                                    const Index row,
                                    const Index diagonal )
      {
         return diagonal*rows + row;
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const tnlMultidiagonalMatrix< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         tnlMatrixVectorProductCuda( matrix, inVector, outVector );
      }
};


#endif /* TNLMULTIDIAGONALMATRIX_IMPL_H_ */
