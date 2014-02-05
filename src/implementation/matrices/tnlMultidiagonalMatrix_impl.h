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
          tnlString( GetParameterType( Real( 0.0 ) ) ) +
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
bool tnlMultidiagonalMatrix< Real, Device, Index >::setRowLengths( const RowLengthsVector& rowLengths )
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
bool tnlMultidiagonalMatrix< Real, Device, Index > :: setDiagonals(  const IndexType diagonalsNumber,
                                                                     const IndexType* diagonalsShift )
{
   tnlAssert( diagonalsNumber > 0,
            cerr << "New number of diagonals = " << diagonalsNumber << endl );
   this->diagonalsShift.setSize( diagonalsNumber );
   for( IndexType i = 0; i < diagonalsNumber; i++ )
      this->diagonalsShift.setElement( i, diagonalsShift[ i ] );
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
   setDimensions( matrix.getRows(), matrix.getColumns() );
   if( ! setDiagonals( matrix.getDiagonals().getSize(),
                       matrix.getDiagonals().getData() ) )
      return false;
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
                   << " matrix.getColumns() = " << matrix.getColumns()
                   << " this->getName() = " << this->getName()
                   << " matrix.getName() = " << matrix.getName() );
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
bool tnlMultidiagonalMatrix< Real, Device, Index > :: setElementFast( const IndexType row,
                                                                      const IndexType column,
                                                                      const Real& value )
{
   IndexType index;
   if( ! this->getElementIndex( row, column, index  ) )
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
   return this->setElementFast( row, column, value );
}


template< typename Real,
          typename Device,
          typename Index >
bool tnlMultidiagonalMatrix< Real, Device, Index > :: addElementFast( const IndexType row,
                                                                      const IndexType column,
                                                                      const RealType& value,
                                                                      const RealType& thisElementMultiplicator )
{
   Index index;
   if( ! this->getElementIndex( row, column, index  ) )
      return false;
   this->values.addElement( index, value, thisElementMultiplicator );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMultidiagonalMatrix< Real, Device, Index > :: setRowFast( const IndexType row,
                                                                  const IndexType* columns,
                                                                  const RealType* values,
                                                                  const IndexType numberOfElements )
{
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMultidiagonalMatrix< Real, Device, Index > :: addRowFast( const IndexType row,
                                                                  const IndexType* columns,
                                                                  const RealType* values,
                                                                  const IndexType numberOfElements,
                                                                  const RealType& thisElementMultiplicator )
{
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlMultidiagonalMatrix< Real, Device, Index >::getElementFast( const IndexType row,
                                                                    const IndexType column ) const
{
   Index index;
   if( ! this->getElementIndex( row, column, index  ) )
      return 0.0;
   return this->values[ index ];
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlMultidiagonalMatrix< Real, Device, Index >::getElement( const IndexType row,
                                                                const IndexType column ) const
{
   return this->getElementFast();
}


template< typename Real,
          typename Device,
          typename Index >
void tnlMultidiagonalMatrix< Real, Device, Index >::getRowFast( const IndexType row,
                                                                IndexType* columns,
                                                                RealType* values ) const
{
   IndexType pointer( 0 );
   for( IndexType i = 0; i < diagonalsShift.getSize(); i++ )
   {
      const IndexType column = row + diagonalsShift.getElement( i );
      if( column >= 0 && column < this->getColumns() )
      {
         columns[ pointer ] = column;
         values[ pointer ] = this->getElement( row, column );
         pointer++;
      }
   }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
void tnlMultidiagonalMatrix< Real, Device, Index >::vectorProduct( const Vector& inVector,
                                                                   Vector& outVector ) const
{

   for( Index row = 0; row < this->getRows(); row ++ )
   {
      Real result = 0.0;
      for( Index i = 0;
           i < this->diagonalsShift.getSize();
           i ++ )
      {
         const Index column = row + this->diagonalsShift.getElement( i );
         if( column >= 0 && column < this->getColumns() )
            result += this->values.getElement( row * this->diagonalsShift.getSize() + i ) *
                      inVector.getElement( column );
      }
      outVector.setElement( row, result );
   }
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
   tnlAssert( false, cerr << "TODO: implement" );
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
                   << " this->getRows() = " << this->getRows()
                   << " this->getName() = " << this->getName() << endl );

   RealType diagonalValue( 0.0 );
   RealType sum( 0.0 );

   for( IndexType i = 0; i < this->getDiagonalsShift.getSize(); i++ )
   {
      const IndexType column = i + this -> diagonalsShift. getElement( i );
      if( column >= 0 && column < this->getColumns() )
      {
         if( column == row )
            diagonalValue = this->values.getElement( row * this->diagonalsShift.getSize() + i );
         else
            sum += this->values.getElement( row * this->diagonalsShift.getSize() + i ) * x. getElement( column );
      }
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
         const IndexType column = row + diagonalsShift[ i ];
         if( column >=0 && column < this->columns )
            str << " Col:" << column << "->" << this->operator()( row, column ) << "\t";
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
                 << " this->rows = " << this->rows
                 << " this->getName() = " << this->getName() << endl );
   tnlAssert( column >=0 && column < this->columns,
            cerr << "column = " << column
                 << " this->columns = " << this->columns
                 << " this->getName() = " << this->getName() << endl );

   IndexType i( 0 );
   while( i < this->diagonalsShift.getSize() )
   {
      if( diagonalsShift[ i ] == column - row )
      {
         index = row*this->diagonalsShift.getSize() + i;
         return true;
      }
      i++;
   }
   return false;
}

#endif /* TNLMULTIDIAGONALMATRIX_IMPL_H_ */
