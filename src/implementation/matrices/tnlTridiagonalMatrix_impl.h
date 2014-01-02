/***************************************************************************
                          tnlTridiagonalMatrix_impl.h  -  description
                             -------------------
    begin                : Nov 30, 2013
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

#ifndef TNLTRIDIAGONALMATRIX_IMPL_H_
#define TNLTRIDIAGONALMATRIX_IMPL_H_

#include <core/tnlAssert.h>
#include <matrices/tnlTridiagonalMatrix.h>

template< typename Real,
          typename Device,
          typename Index >
tnlTridiagonalMatrix< Real, Device, Index >::tnlTridiagonalMatrix()
{
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlTridiagonalMatrix< Real, Device, Index >::getType()
{
   return tnlString( "tnlTridiagonalMatrix< " ) +
          tnlString( getParameterType< RealType >() ) + ", " +
          tnlString( Device :: getDeviceType() ) + ", " +
          tnlString( getParameterType< IndexType >() ) + " >";
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlTridiagonalMatrix< Real, Device, Index >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlTridiagonalMatrix< Real, Device, Index >::setDimensions( const IndexType rows,
                                                                 const IndexType columns )
{
   if( ! tnlMatrix< Real, Device, Index >::setDimensions( rows, columns ) )
      return false;
   if( ! values.setSize( 3*Min( rows, columns ) ) )
      return false;
   this->values.setValue( 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlTridiagonalMatrix< Real, Device, Index >::setRowLengths( const RowLengthsVector& rowLengths )
{
   if( rowLengths[ 0 ] > 2 )
      return false;
   const IndexType diagonalLength = Min( this->getRows(), this->getColumns() );
   for( Index i = 1; i < diagonalLength-1; i++ )
      if( rowLengths[ i ] > 3 )
         return false;
   if( this->getRows() > this->getColumns() )
      if( rowLengths[ this->getRows()-1 ] > 1 )
         return false;
   if( this->getRows() == this->getColumns() )
      if( rowLengths[ this->getRows()-1 ] > 2 )
         return false;
   if( this->getRows() < this->getColumns() )
      if( rowLengths[ this->getRows()-1 ] > 3 )
         return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
IndexType tnlTridiagonalMatrix< Real, Device, Index >::getRowLength( const IndexType row ) const
{
   const IndexType diagonalLength = Min( this->getRows(), this->getColumns() );
   if( row == 0 )
      return 2;
   if( row > 0 && row < diagonalLength - 1 )
      return 3;
   if( this->getRows() > this->getColumns() )
      return 1;
   if( this->getRows() == this->getColumns() )
      return 2;
   return 3;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2, typename Device2, typename Index2 >
bool tnlTridiagonalMatrix< Real, Device, Index >::setLike( const tnlTridiagonalMatrix< Real2, Device2, Index2 >& m )
{
   return this->setDimensions( m.getRows(), m.getColumns() );
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlTridiagonalMatrix< Real, Device, Index >::getNumberOfMatrixElements() const
{
   return 3 * Min( this->getRows(), this->getColumns() );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlTridiagonalMatrix< Real, Device, Index >::reset()
{
   tnlMatrix< Real, Device, Index >::reset();
   this->values.reset();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2, typename Device2, typename Index2 >
bool tnlTridiagonalMatrix< Real, Device, Index >::operator == ( const tnlTridiagonalMatrix< Real2, Device2, Index2 >& matrix ) const
{
   return this->values == matrix.values;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2, typename Device2, typename Index2 >
bool tnlTridiagonalMatrix< Real, Device, Index >::operator != ( const tnlTridiagonalMatrix< Real2, Device2, Index2 >& matrix ) const
{
   return this->values != matrix.values;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlTridiagonalMatrix< Real, Device, Index >::setValue( const RealType& v )
{
   this->values.setValue( v );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlTridiagonalMatrix< Real, Device, Index >::setElement( const IndexType row,
                                                              const IndexType column,
                                                              const RealType& value )
{
   this->values.setElement( this->getElementIndex( row, column ), value );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlTridiagonalMatrix< Real, Device, Index >::addElement( const IndexType row,
                                                              const IndexType column,
                                                              const RealType& value,
                                                              const RealType& thisElementMultiplicator )
{
   this->values.addElement( this->getElementIndex( row, column ), value, thisElementMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlTridiagonalMatrix< Real, Device, Index >::setRow( const IndexType row,
                                                          const IndexType* columns,
                                                          const RealType* values,
                                                          const IndexType elements )
{
   tnlAssert( elements <= this->columns,
            cerr << " elements = " << elements
                 << " this->columns = " << this->columns
                 << " this->getName() = " << this->getName() );
   return this->addRow( row, columns, values, elements, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlTridiagonalMatrix< Real, Device, Index >::addRow( const IndexType row,
                                                          const IndexType* columns,
                                                          const RealType* values,
                                                          const IndexType elements,
                                                          const RealType thisRowMultiplicator )
{
   tnlAssert( elements <= this->columns,
            cerr << " elements = " << elements
                 << " this->columns = " << this->columns
                 << " this->getName() = " << this->getName() );
   if( elements > 3 )
      return false;
   for( IndexType i = 0; i < elements; i++ )
   {
      const IndexType& column = columns[ i ];
      if( column < row - 1 || column > row + 1 )
         return false;
      addElement( row, column, values[ i ], thisRowMultiplicator );
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlTridiagonalMatrix< Real, Device, Index >::getElement( const IndexType row,
                                                              const IndexType column ) const
{
   if( abs( column - row ) > 1 )
      return 0.0;
   return this->values.getElement( this->getElementIndex( row, column ) );
}

template< typename Real,
          typename Device,
          typename Index >
Real& tnlTridiagonalMatrix< Real, Device, Index >::operator()( const IndexType row,
                                                               const IndexType column )
{
   return this->values[ this->getElementIndex( row, column ) ];
}

template< typename Real,
          typename Device,
          typename Index >
const Real& tnlTridiagonalMatrix< Real, Device, Index >::operator()( const IndexType row,
                                                                     const IndexType column ) const
{
   return this->values[ this->getElementIndex( row, column ) ];
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlTridiagonalMatrix< Real, Device, Index >::addElement( const IndexType row,
                                                              const IndexType column,
                                                              const RealType& value,
                                                              const RealType& thisElementMultiplicator )
{
   const IndexType elementIndex = this->getElementIndex( row, column );
   if( thisElementMultiplicator == 1.0 )
      this->values[ elementIndex ] += value;
   else
      this->values[ elementIndex ] =
         thisElementMultiplicator * this->values[ elementIndex ] + value;
}

template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
typename Vector::RealType tnlTridiagonalMatrix< Real, Device, Index >::rowVectorProduct( const IndexType row,
                                                                                         const Vector& vector ) const
{
   if( row == 0 )
      return vector[ 0 ] * this->values[ 0 ] +
             vector[ 1 ] * this->values[ 1 ];
   IndexType i = 3 * row - 1;
   if( row == this->rows - 1 )
      return vector[ row - 1 ] * this->values[ i++ ] +
             vector[ row ] * this->values[ i ];
   return vector[ row - 1 ] * this->values[ i++ ] +
          vector[ row ] * this->values[ i++ ] +
          vector[ row + 1 ] * this->values[ i ];
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
void tnlTridiagonalMatrix< Real, Device, Index >::vectorProduct( const Vector& inVector,
                                                                 Vector& outVector ) const
{
   tnlAssert( this->getColumns() == inVector.getSize(),
            cerr << "Matrix columns: " << this->getColumns() << endl
                 << "Matrix name: " << this->getName() << endl
                 << "Vector size: " << inVector.getSize() << endl
                 << "Vector name: " << inVector.getName() << endl );
   tnlAssert( this->getRows() == outVector.getSize(),
               cerr << "Matrix rows: " << this->getRows() << endl
                    << "Matrix name: " << this->getName() << endl
                    << "Vector size: " << outVector.getSize() << endl
                    << "Vector name: " << outVector.getName() << endl );

   for( IndexType row = 0; row < this->getRows(); row++ )
      outVector[ row ] = this->rowVectorProduct( row, inVector );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2, typename Index2 >
void tnlTridiagonalMatrix< Real, Device, Index >::addMatrix( const tnlTridiagonalMatrix< Real2, Device, Index2 >& matrix,
                                                             const RealType& matrixMultiplicator,
                                                             const RealType& thisMatrixMultiplicator )
{
   tnlAssert( this->getRows() == matrix.getRows(),
            cerr << "This matrix columns: " << this->getColumns() << endl
                 << "This matrix rows: " << this->getRows() << endl
                 << "This matrix name: " << this->getName() << endl );

   const IndexType lastI = this->values.getSize();
   if( thisMatrixMultiplicator == 1.0 )
   {
      for( IndexType i = 0; i < lastI; i++ )
         this->values[ i ] += matrixMultiplicator*matrix.values[ i ];
   }
   else
   {
      for( IndexType i = 0; i < lastI; i++ )
         this->values[ i ] = thisMatrixMultiplicator * this->values[ i ] +
            matrixMultiplicator*matrix.values[ i ];
   }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2, typename Index2, int tileDim >
void tnlTridiagonalMatrix< Real, Device, Index >::getTransposition( const tnlTridiagonalMatrix< Real2, Device, Index2 >& matrix,
                                                                    const RealType& matrixMultiplicator )
{
   tnlAssert( this->getRows() == matrix.getRows(),
               cerr << "This matrix rows: " << this->getRows() << endl
                    << "That matrix rows: " << matrix.getRows() << endl );

   const IndexType& rows = matrix.getRows();
   for( IndexType i = 1; i < rows; i++ )
   {
      RealType aux = matrix. getElement( i, i - 1 );
      this->setElement( i, i - 1, matrix.getElement( i - 1, i ) );
      this->setElement( i, i, matrix.getElement( i, i ) );
      this->setElement( i - 1, i, aux );
   }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
void tnlTridiagonalMatrix< Real, Device, Index >::performSORIteration( const Vector& b,
                                                                 const IndexType row,
                                                                 Vector& x,
                                                                 const RealType& omega ) const
{
   RealType sum( 0.0 );
   for( IndexType i = 0; i < this->getColumns(); i++ )
      sum += this->operator()( row, i ) * x[ i ];
   x[ row ] += omega / this->operator()( row, row )( b[ row ] - sum );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlTridiagonalMatrix< Real, Device, Index >::save( tnlFile& file ) const
{
   if( ! tnlMatrix< Real, Device, Index >::save( file ) ||
       ! this->values.save( file ) )
   {
      cerr << "Unable to save the tridiagonal matrix " << this->getName() << "." << endl;
      return false;
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlTridiagonalMatrix< Real, Device, Index >::load( tnlFile& file )
{
   if( ! tnlMatrix< Real, Device, Index >::load( file ) ||
       ! this->values.load( file ) )
   {
      cerr << "Unable to save the tridiagonal matrix " << this->getName() << "." << endl;
      return false;
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlTridiagonalMatrix< Real, Device, Index >::save( const tnlString& fileName ) const
{
   return tnlObject::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlTridiagonalMatrix< Real, Device, Index >::load( const tnlString& fileName )
{
   return tnlObject::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlTridiagonalMatrix< Real, Device, Index >::print( ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      for( IndexType column = row - 1; column < row + 2; column++ )
         if( column >= 0 && column < this->columns )
            str << " Col:" << column << "->" << this->operator()( row, column ) << "\t";
      str << endl;
   }
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlTridiagonalMatrix< Real, Device, Index >::getElementIndex( const IndexType row,
                                                                    const IndexType column ) const
{
   tnlAssert( row >= 0 && column >= 0 && row < this->rows && column < this->rows,
              cerr << " this->rows = " << this->rows
                   << " row = " << row << " column = " << column );
   tnlAssert( abs( row - column ) < 2,
              cerr << "row = " << row << " column = " << column << endl );
   return 3*row + column - row;
}




#endif /* TNLTRIDIAGONALMATRIX_IMPL_H_ */
