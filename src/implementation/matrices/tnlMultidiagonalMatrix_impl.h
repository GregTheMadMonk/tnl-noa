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

#include <matrices/tnlMultidiagonalMatrix_impl.h>

template< typename Real,
          typename Device,
          typename Index >
tnlMultidiagonalMatrix< Real, Device, Index > :: tnlMultidiagonalMatrix()
: rows( 0 ),
  columns( 0 )
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
bool tnlMultidiagonalMatrix< Real, Device, Index > :: setDimensions( const IndexType rows,
                                                                     const IndexType columns )
{
   tnlAssert( rows > 0 && columns > 0,
              cerr << "rows = " << rows
                   << " columns = " << columns << endl );
   this->rows = rows;
   this->columns = columns;
   if( this->diagonalsOffsets.getSize() != 0 )
      return values.setSize( tnlMin( this->rows, this->columns ) * this->diagonalsOffsets.getSize() );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMultidiagonalMatrix< Real, Device, Index > :: setDiagonals( const tnlVector< Index, Device, Index >& diagonals )
{
   tnlAssert( diagonals.getSize() > 0,
            cerr << "New number of diagonals = " << diagonals.getSize() << endl );
   this->diagonalsOffsets = diagonals;
   diagonalsOffsets. setSize( this -> numberOfDiagonals );
   if( this->rows != 0 && this->columns != 0 )
      return values.setSize( tnlMin( this->rows, this->columns ) * this->diagonalsOffsets.getSize() );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
const tnlVector< Real, Device, Index >& tnlMultidiagonalMatrix< Real, Device, Index > :: getDiagonals() const
{
   return this->diagonalsOffsets;
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
   if( ! setDiagonals( matrix.getDiagonals() ) )
      return false;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlMultidiagonalMatrix< Real, Device, Index > :: getNumberOfAllocatedElements() const
{
   return this->values.getSize();
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
Index tnlMultidiagonalMatrix< Real, Device, Index >::getRows() const
{
   return this->rows;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlMultidiagonalMatrix< Real, Device, Index >::getColumns() const
{
   return this->columns;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool operator == ( const tnlMultidiagonalMatrix< Real2, Device2, Index2 >& matrix ) const
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
bool operator != ( const tnlMultidiagonalMatrix< Real2, Device2, Index2 >& matrix ) const
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
bool tnlMultidiagonalMatrix< Real, Device, Index > :: setElement( const IndexType row,
                                                                  const IndexType column,
                                                                  const Real& value )
{
   Index i;
   if( ! this->getElementIndex( row, column, index  ) )
      return false;
   this->values[ i ] = value;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlMultiDiagonalMatrix< Real, Device, Index >::getElement( const IndexType row,
                                                                const IndexType column ) const
{
   Index i;
   if( ! this->getElementIndex( row, column, index  ) )
      return 0.0;
   return this->values[ i ];
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMultiDiagonalMatrix< Real, Device, Index > :: addToElement( const IndexType row,
                                                                    const IndexType column,
                                                                    const RealType& value,
                                                                    const RealType& thisElementMultiplicator )
{
   Index i;
   if( ! this->getElementIndex( row, column, index  ) )
      return false;
   this->values[ i ] = thisElementMultiplicator * this->values[ i ] * value;
   return true;
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
           i < this->diagonalsOffsets.getSize();
           i ++ )
      {
         const Index column = row + this->diagonalsOffsets.getElement( i );
         if( column >= 0 && column < this->getColumns() )
            result += this->values.getElement( row * this->diagonalsOffsets() + i ) *
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
bool tnlMultiDiagonalMatrix< Real, Device, Index > :: performSORIteration( const Vector& b,
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

   for( IndexType i = 0; i < this->getDiagonalsOffsets.getSize(); i++ )
   {
      const IndexType column = i + this -> diagonalsOffsets. getElement( j );
      if( column >= 0 && column < this->getColumns() )
      {
         if( column == row )
            diagonal = this->values.getElement( row * this->diagonalsOffsets.getSize() + i );
         else
            update += this->values.getElement( row * this->diagonalsOffsets.getSize() + i ) * x. getElement( column );
      }
   }
   if( diagonal == ( Real ) 0.0 )
   {
      cerr << "There is zero on the diagonal in " << row << "-th row of thge matrix " << this->getName() << ". I cannot perform SOR iteration." << endl;
      return false;
   }
   x. setElement( row, x[ i ] + omega / diagonal * ( b[ row ] - update ) );
   return true;
}


template< typename Real,
          typename Device,
          typename Index >
bool tnlMultidiagonalMatrix< Real, Device, Index >::save( tnlFile& file ) const
{
   if( ! file.save( this->rows ) ) return false;
   if( ! file.save( this->columns ) ) return false;
   if( ! this->values.save( file ) ) return false;
   if( ! this->diagonalsOffsets.save( file ) ) return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMultidiagonalMatrix< Real, Device, Index >::load( tnlFile& file )
{
   if( ! file.load( this->rows ) ) return false;
   if( ! file.load( this->columns ) ) return false;
   if( ! this->values.load( file ) ) return false;
   if( ! this->diagonalsOffsets.load( file ) ) return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMultiDiagonalMatrix< Real, Device, Index >::save( const tnlString& fileName ) const
{
   return tnlObject::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMultiDiagonalMatrix< Real, Device, Index >::load( const tnlString& fileName )
{
   return tnlObject::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlMultiDiagonalMatrix< Real, Device, Index >::printOut( ostream& str,
                                                              const tnlString& format,
                                                              const Index lines ) const
{
   str << "Multi-diagonal matrix " << this -> getName() << endl;
   str << "Number of diagonals: " << this -> getNumberOfDiagonals() << endl;
   for( Index row = 0; row < this -> getSize(); row ++ )
   {
      str << "Row " << row << ": ";
      for( Index i = 0; i < this -> getNumberOfDiagonals(); i ++ )
         str << nonzeroElements. getElement( row * this -> getNumberOfDiagonals() + i )
             << " @ " << row + this -> diagonalsOffsets. getElement( i )
             << ", ";
   }
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMultiDiagonalMatrix< Real, Device, Index >::getElementIndex( const IndexType row,
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
   while( i < this->diagonals.getSize() )
   {
      if( diagonals[ i ] == column - row )
      {
         index = row*this->diagonals.getSize() + i;
         return true;
      }
      i++;
   }
   return false;
}


#endif /* TNLMULTIDIAGONALMATRIX_IMPL_H_ */
