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
: rows( 0 ),
  columns( 0 ),
  rowLengths( 0 )
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
bool tnlEllpackMatrix< Real, Device, Index > :: setDimensions( const IndexType rows,
                                                               const IndexType columns )
{
   tnlAssert( rows > 0 && columns > 0,
              cerr << "rows = " << rows
                   << " columns = " << columns << endl );
   this->rows = rows;
   this->columns = columns;
   if( this->rowLengths != 0 )
      return allocateElements();
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >          
bool tnlEllpackMatrix< Real, Device, Index > :: setRowLengths( const Vector& rowLengths )
{
   this->rowLengths = 0;
   for( IndexType i = 0; i < rowLengths.getSize(); i++ )
      this->rowLengths = Max( this->rowLengths, rowLengths[ i ] );
   tnlAssert( this->rowLengths > 0,
              cerr << "this->rowLengths = " << this->rowLengths );
   if( this->rows > 0 )
      return allocateElements();
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlEllpackMatrix< Real, Device, Index > :: setConstantRowLengths( const IndexType& rowLengths )
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
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlEllpackMatrix< Real, Device, Index > :: setLike( const tnlEllpackMatrix< Real2, Device2, Index2 >& matrix )
{
   this->rowLengths = 0;
   this->setDimensions( matrix.getRows(), matrix.getColumns() );
   if( ! this->setConstantRowLengths( matrix.rowLengths ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlEllpackMatrix< Real, Device, Index > :: getNumberOfAllocatedElements() const
{
   return this->values.getSize();
}

template< typename Real,
          typename Device,
          typename Index >
void tnlEllpackMatrix< Real, Device, Index > :: reset()
{
   this->rows = 0;
   this->columns = 0;
   this->rowLengths = 0;
   this->values.reset();
   this->columnIndexes.reset();
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlEllpackMatrix< Real, Device, Index >::getRows() const
{
   return this->rows;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlEllpackMatrix< Real, Device, Index >::getColumns() const
{
   return this->columns;
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
Real tnlEllpackMatrix< Real, Device, Index >::getElement( const IndexType row,
                                                          const IndexType column ) const
{
   IndexType i( row * this->rowLengths );
   const IndexType rowEnd( i + this->rowLengths );
   while( i < rowEnd && this->columnIndexes[ i ] < column ) i++;
   if( i == rowEnd || this->columnIndexes[ i ] != column )
      return 0.0;
   return this->values.getElement( i );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlEllpackMatrix< Real, Device, Index > :: addElement( const IndexType row,
                                                            const IndexType column,
                                                            const RealType& value,
                                                            const RealType& thisElementMultiplicator )
{
   tnlAssert( row >= 0 && row < this->rows &&
              column >= 0 && column <= this->rows,
              cerr << " row = " << row
                   << " column = " << column
                   << " this->rows = " << this->rows
                   << " this->columns = " << this-> columns );
   IndexType i( row * this->rowLengths );
   const IndexType rowEnd( i + this->rowLengths );
   while( i < rowEnd && this->columnIndexes[ i ] < column ) i++;
   if( i == rowEnd )
      return false;
   if( this->columnIndexes[ i ] == column )
   {
      this->values[ i ] = thisElementMultiplicator * this->values[ i ] + value;
      return true;
   }
   else
      if( this->columnIndexes[ i ] == this->columns )
      {
         this->columnIndexes[ i ] = column;
         this->values[ i ] = value;
         return true;
      }
      else
      {
         IndexType j = rowEnd - 1;
         while( j > i )
         {
            this->columnIndexes[ j ] = this->columnIndexes[ j - 1 ];
            this->values[ j ] = this->values[ j - 1 ];
            j--;
         }
         this->columnIndexes[ i ] = column;
         this->values[ i ] = value;
         return true;
      }
   return false;
}


template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
void tnlEllpackMatrix< Real, Device, Index >::vectorProduct( const Vector& inVector,
                                                                   Vector& outVector ) const
{
   for( Index row = 0; row < this->getRows(); row ++ )
   {
      Real result = 0.0;
      IndexType i( row * this->rowLengths );
      const IndexType rowEnd( i + this->rowLengths );
      while( i < rowEnd && this->columnIndexes[ i ] < this->columns )
      {
         const Index column = this->columnIndexes.getElement( i );
         result += this->values.getElement( i++ ) * inVector.getElement( column );
      }
      outVector.setElement( row, result );
   }
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
   if( ! file.write( &this->rows ) ) return false;
   if( ! file.write( &this->columns ) ) return false;
   if( ! file.write( &this->rowLengths ) ) return false;
   if( ! this->values.save( file ) ) return false;
   if( ! this->columnIndexes.save( file ) ) return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlEllpackMatrix< Real, Device, Index >::load( tnlFile& file )
{
   if( ! file.read( &this->rows ) ) return false;
   if( ! file.read( &this->columns ) ) return false;
   if( ! file.read( &this->rowLengths ) ) return false;
   if( ! this->values.load( file ) ) return false;
   if( ! this->columnIndexes.load( file ) ) return false;
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
      while( i < rowEnd && this->columnIndexes[ i ] < this->columns )
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
   if( ! this->values.setSize( this->rows * this->rowLengths ) ||
       ! this->columnIndexes.setSize( this->rows * this->rowLengths ) )
      return false;

   /****
    * Setting a column index to this->columns means that the
    * index is undefined.
    */
   this->columnIndexes.setValue( this->columns );
   return true;
}

#endif /* TNLELLPACKMATRIX_IMPL_H_ */
