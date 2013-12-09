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
: rows( 0 )
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
bool tnlTridiagonalMatrix< Real, Device, Index >::setDimensions( const IndexType rows )
{
   if( ! values.setSize( 3*rows - 2 ) )
      return false;
   this->rows = rows;
   this->values.setValue( 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2, typename Device2, typename Index2 >
bool tnlTridiagonalMatrix< Real, Device, Index >::setLike( const tnlTridiagonalMatrix< Real2, Device2, Index2 >& m )
{
   return this->setDimensions( m.getRows() );
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlTridiagonalMatrix< Real, Device, Index >::getNumberOfAllocatedElements() const
{
   return 3 * rows - 2;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlTridiagonalMatrix< Real, Device, Index >::reset()
{
   this->rows = 0;
   this->values.reset();
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlTridiagonalMatrix< Real, Device, Index >::getRows() const
{
   return this->rows;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlTridiagonalMatrix< Real, Device, Index >::getColumns() const
{
   return this->rows;
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
void tnlTridiagonalMatrix< Real, Device, Index >::setElement( const IndexType row,
                                                              const IndexType column,
                                                              const RealType& value )
{
   this->values.setElement( this->getElementIndex( row, column ), value );
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
bool tnlTridiagonalMatrix< Real, Device, Index >::addToElement( const IndexType row,
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

   IndexType i( 0 );
   outVector[ 0 ] = inVector[ 0 ] * this->values[ i++ ] +
                    inVector[ 1 ] * this->values[ i++ ];
   IndexType row;
   for( row = 1; row < this->getRows() - 1; row++ )
   {

      outVector[ row ] = inVector[ row - 1 ] * this->values[ i++ ] +
                         inVector[ row ] * this->values[ i++ ] +
                         inVector[ row + 1 ] * this->values[ i++ ];
   }
   outVector[ row ] = inVector[ row - 1 ] * this->values[ i++ ] +
                      inVector[ row ] * this->values[ i ];
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
   if( ! file.write( &this->rows ) ||
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
   if( ! file.read( &this->rows ) ||
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
         if( column >=0 && columns < this-columns )
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