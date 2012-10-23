/***************************************************************************
                          tnlMultiDiagonalMatrix.h  -  description
                             -------------------
    begin                : Oct 13, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
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

#ifndef TNLMULTIDIAGONALMATRIX_H_
#define TNLMULTIDIAGONALMATRIX_H_

#include <matrix/tnlMatrix.h>

template< typename Real, typename Device = tnlHost, typename Index = int >
class tnlMultiDiagonalMatrix : public tnlMatrix< Real, Device, Index >
{
   public:

   //! Basic constructor
   tnlMultiDiagonalMatrix( const tnlString& name );

   const tnlString& getMatrixClass() const;

   tnlString getType() const;

   //! Sets the number of row and columns.
   bool setSize( Index new_size );

   bool setNumberOfDiagonals( const Index& diagonals );

   const Index& getNumberOfDiagonals() const;

   void setDiagonalsOffsets( const tnlVector< Index, Device, Index >& diagonalsOffsets );

   bool setNonzeroElements( Index elements );

   Index getNonzeroElements() const;

   const tnlVector< Index, Device, Index >& getDiagonalsOffsets() const;

   bool setLike( const tnlCSRMatrix< Real, Device, Index >& matrix );

   void reset();

   //! Insert one row to the matrix.
   /*! If there is some data already in this @param row it will be rewritten.
    *  @param elements says number of non-zero elements which will be inserted.
    *  @param data is pointer to the elements values.
    *  @param first_column is the column of the first non-zero element.
    *  @param offsets is a pointer to field with offsets of the elements with
    *  respect to the first one. All of them must sorted increasingly.
    *  The elements which do not fit to the matrix are omitted.
    */
   bool insertRow( Index row,
                   Index elements,
                   Real* data,
                   Index first_column,
                   Index* offsets );

   bool setElement( Index row,
                    Index colum,
                    const Real& value );

   bool addToElement( Index row,
                      Index column,
                      const Real& value );

   Real getElement( Index row,
                    Index column ) const;

   Real rowProduct( Index row,
                    const tnlVector< Real, Device, Index >& vector ) const;

   void vectorProduct( const tnlVector< Real, Device, Index >& x,
                       tnlVector< Real, Device, Index >& b ) const;

   bool performSORIteration( const Real& omega,
                             const tnlVector< Real, Device, Index >& b,
                             tnlVector< Real, Device, Index >& x,
                             Index firstRow,
                             Index lastRow ) const;

   Real getRowL1Norm( Index row ) const;

   void multiplyRow( Index row, const Real& value );

   bool read( istream& str, int verbose = 0 );

   //! Method for saving the matrix to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the matrix from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   tnlCSRMatrix< Real, Device, Index >& operator = ( const tnlCSRMatrix< Real, Device, Index >& csrMatrix );

   //! Prints out the matrix structure
   void printOut( ostream& str,
                  const tnlString& format = tnlString( "" ),
                  const Index lines = 0 ) const;

   protected:

   enum elementOperation { set_element, add_to_element };

   Index numberOfDiagonals;

   tnlVector< Real, Device, Index > nonzeroElements;

   tnlVector< Index, Device, Index > diagonalsOffsets;

};

template< typename Real, typename Device, typename Index >
tnlMultiDiagonalMatrix< Real, Device, Index > :: tnlMultiDiagonalMatrix( const tnlString& name )
   : tnlMatrix< Real, Device, Index >( name ),
     nonzeroElements( name + " : nonzeroElements" ),
     diagonalsOffsets( name + " : diagonalsOffsets" ),
     numberOfDiagonals( 0 )
{
};

template< typename Real, typename Device, typename Index >
const tnlString& tnlMultiDiagonalMatrix< Real, Device, Index > :: getMatrixClass() const
{
   return tnlMatrixClass :: main;
};

template< typename Real, typename Device, typename Index >
tnlString tnlMultiDiagonalMatrix< Real, Device, Index > :: getType() const
{
   return tnlString( "tnlMultiDiagonalMatrix< ") +
          tnlString( GetParameterType( Real( 0.0 ) ) ) +
          tnlString( ", " ) +
          getDeviceType( Device ) +
          tnlString( " >" );
}

template< typename Real, typename Device, typename Index >
bool tnlMultiDiagonalMatrix< Real, Device, Index > :: setSize( Index new_size )
{
   tnlAssert( new_size > 0, cerr << "new_size = " << new_size << endl; );
   this -> size = new_size;
   if( this -> numberOfDiagonals != 0 )
      return nonzeroElements. setSize( this -> size * this -> numberOfDiagonals );
   return true;
}

template< typename Real, typename Device, typename Index >
bool tnlMultiDiagonalMatrix< Real, Device, Index > :: setNumberOfDiagonals( const Index& diagonals )
{
   tnlAssert( diagonals > 0, cerr << "New number of diagonals = " << diagonals << endl );
   this -> numberOfDiagonals = diagonals;
   diagonalsOffsets. setSize( this -> numberOfDiagonals );
   if( this -> size != 0 )
      return nonzeroElements. setSize( this -> size * this -> numberOfDiagonals );
   return true;
}

template< typename Real, typename Device, typename Index >
const Index& tnlMultiDiagonalMatrix< Real, Device, Index > :: getNumberOfDiagonals() const
{
   return this -> numberOfDiagonals;
}

template< typename Real, typename Device, typename Index >
bool tnlMultiDiagonalMatrix< Real, Device, Index > :: setNonzeroElements( Index elements )
{

}

template< typename Real, typename Device, typename Index >
Index tnlMultiDiagonalMatrix< Real, Device, Index > :: getNonzeroElements() const
{
   return this -> nonzeroElements. getSize();
}

template< typename Real, typename Device, typename Index >
void tnlMultiDiagonalMatrix< Real, Device, Index > :: setDiagonalsOffsets( const tnlVector< Index, Device, Index >& diagonalsOffsets )
{
   tnlAssert( diagonalsOffsets. getSize() == this -> diagonalsOffsest. getSize(),
              cerr << "diagonalsOffsets. getSize() = " << diagonalsOffsets. getSize()
                   << " this -> diagonalsOffsets. getSize() = " << this -> diagonalsOffsets. getSize() << endl; );
   this -> diagonalsOffsets = diagonalsOffsets;
}

template< typename Real, typename Device, typename Index >
const tnlVector< Index, Device, Index >& tnlMultiDiagonalMatrix< Real, Device, Index > :: getDiagonalsOffsets() const
{
   return this -> diagonalsOffsets();
}

template< typename Real, typename Device, typename Index >
bool tnlMultiDiagonalMatrix< Real, Device, Index > :: setLike( const tnlCSRMatrix< Real, Device, Index >& matrix )
{
   if( ! this -> setSize( matrix. getSize() ) )
      return false;
   if( ! this -> setNumberOfDiagonals( matrix. getNumberOfDiagonals() ) )
      return false;
   return true;
}

template< typename Real, typename Device, typename Index >
void tnlMultiDiagonalMatrix< Real, Device, Index > :: reset()
{
   this -> nonzeroElements. reset();
   this -> diagonalsOffsets. reset();
}

template< typename Real, typename Device, typename Index >
bool tnlMultiDiagonalMatrix< Real, Device, Index > :: insertRow( Index row,
                 Index elements,
                 Real* data,
                 Index first_column,
                 Index* offsets )
{
   tnlAssert( false, );
   // TODO: implement this
}

template< typename Real, typename Device, typename Index >
bool tnlMultiDiagonalMatrix< Real, Device, Index > :: setElement( Index row,
                  Index column,
                  const Real& value )
{
   Index i = 0;
   while( i < this -> getNumberOfDiagonals() &&
          this -> diagonalsOffsets. getElement( i ) + row < column )
      i ++ ;
   if( i < this -> getNumberOfDiagonals() )
   {
      nonzeroElements. setElement( row * this -> getNumberOfDiagonals() + i, value );
      return true;
   }
   cerr << "Cannot insert element at position ( " << row << ", " << column << " ) in the multidiagonal matrix." << endl;
   return false;
}

template< typename Real, typename Device, typename Index >
bool tnlMultiDiagonalMatrix< Real, Device, Index > :: addToElement( Index row,
                    Index column,
                    const Real& value )
{
   Index i = 0;
   while( i < this -> getNumberOfDiagonals() &&
          this -> diagonalsOffsets. getElement( i ) + row < column )
      i ++ ;
   if( i < this -> getNumberOfDiagonals() )
   {
      const Real v = this -> nonzeroElements. getElement( row * this -> getNumberOfDiagonals() + i );
      this -> nonzeroElements. setElement( row * this -> getNumberOfDiagonals() + i, value );
      return true;
   }
   cerr << "Cannot insert element at position ( " << row << ", " << column << " ) in the multidiagonal matrix." << endl;
   abort();
   return false;
}

template< typename Real, typename Device, typename Index >
Real tnlMultiDiagonalMatrix< Real, Device, Index > :: getElement( Index row,
                  Index column ) const
{
   Index i = 0;
   while( i < this -> getNumberOfDiagonals() &&
          this -> diagonalsOffsets. getElement( i ) + row < column )
      i ++ ;
   if( i < this -> getNumberOfDiagonals() )
      return this -> nonzeroElements. getElement( row * this -> getNumberOfDiagonals() + i );
   return 0.0;
}

template< typename Real, typename Device, typename Index >
Real tnlMultiDiagonalMatrix< Real, Device, Index > :: rowProduct( Index row,
                  const tnlVector< Real, Device, Index >& vector ) const
{
   Real result = 0.0;
   for( Index i = 0;
        i <  this -> getNumberOfDiagonals();
        i ++ )
   {
      const Index column = row + this -> diagonalsOffsets. getElement( i );
      if( column >= 0 )
         result += this -> nonzeroElements. getElement( row * this -> getNumberOfDiagonals() + i ) *
                   vector. getElement( column );
   }
   return result;
}

template< typename Real, typename Device, typename Index >
void tnlMultiDiagonalMatrix< Real, Device, Index > :: vectorProduct( const tnlVector< Real, Device, Index >& x,
                                                                     tnlVector< Real, Device, Index >& b ) const
{

   for( Index row = 0; row < this -> getSize(); row ++ )
   {
      Real result = 0.0;
      for( Index i = 0;
           i < this -> getNumberOfDiagonals();
           i ++ )
      {
         const Index column = row + this -> diagonalsOffsets. getElement( i );
         if( column >= 0 )
            result += this -> nonzeroElements. getElement( row * this -> getNumberOfDiagonals() + i ) *
                      x. getElement( column );
      }
      b. setElement( row, result );
   }
}

template< typename Real, typename Device, typename Index >
bool tnlMultiDiagonalMatrix< Real, Device, Index > :: performSORIteration( const Real& omega,
                                                                           const tnlVector< Real, Device, Index >& b,
                                                                           tnlVector< Real, Device, Index >& x,
                                                                           Index firstRow,
                                                                           Index lastRow ) const
{
   tnlAssert( firstRow >=0 && firstRow < this -> getSize(),
              cerr << "Wrong parameter firstRow. Should be in 0..." << this -> getSize()
                   << " but it equals " << firstRow << endl; );
   tnlAssert( lastRow >=0 && lastRow < this -> getSize(),
              cerr << "Wrong parameter lastRow. Should be in 0..." << this -> getSize()
                   << " but it equals " << lastRow << endl; );

   if( lastRow == 0 )
      lastRow = this -> getSize();
   for( Index i = firstRow; i < lastRow; i ++ )
   {
      Real diagonal( 0.0 );
      Real update = b[ i ];
      for( Index j = 0; j < this -> getNumberOfDiagonals(); j ++ )
      {
         const Index column = i + this -> diagonalsOffsets. getElement( j );
         if( column >= 0 )
         {
            if( column == i )
               diagonal = this -> nonzeroElements. getElement( i * this -> getNumberOfDiagonals() + j );
            else
               update -= this -> nonzeroElements. getElement( i * this -> getNumberOfDiagonals() + j ) * x. getElement( column );
         }
      }
      if( diagonal == ( Real ) 0.0 )
      {
         cerr << "There is zero on the diagonal in " << i << "-th row. I cannot perform SOR iteration." << endl;
         return false;
      }
      x. setElement( i, ( 1.0 - omega ) * x[ i ] + omega / diagonal * update );
   }
   return true;
}

template< typename Real, typename Device, typename Index >
Real tnlMultiDiagonalMatrix< Real, Device, Index > :: getRowL1Norm( Index row ) const
{
   Real norm( 0.0 );
   for( Index i = row * this -> getNumberOfDiagonals();
              i < ( row + 1 ) * this -> getNumberOfDiagonals();
              i ++ )
      norm += fabs( this -> nonzeroElements. getElement( i ) );
   return norm;
}

template< typename Real, typename Device, typename Index >
void tnlMultiDiagonalMatrix< Real, Device, Index > :: multiplyRow( Index row, const Real& value )
{
   for( Index i = row * this -> getNumberOfDiagonals();
              i < ( row + 1 ) * this -> getNumberOfDiagonals();
              i ++ )
      this -> nonzeroElements[ i ] *= value;
}

template< typename Real, typename Device, typename Index >
bool tnlMultiDiagonalMatrix< Real, Device, Index > :: read( istream& str, int verbose )
{
   tnlAssert( false, );
}

template< typename Real, typename Device, typename Index >
bool tnlMultiDiagonalMatrix< Real, Device, Index > :: save( tnlFile& file ) const
{
   if( ! tnlMatrix< Real, Device, Index > :: save( file ) ) return false;
   if( ! this -> nonzeroElements. save( file ) ) return false;
   if( ! this -> diagonalsOffsets. save( file ) ) return false;
   if( ! file. write( &this -> numberOfDiagonals, 1 ) )
      return false;
   return true;
}

template< typename Real, typename Device, typename Index >
bool tnlMultiDiagonalMatrix< Real, Device, Index > :: load( tnlFile& file )
{
   if( ! tnlMatrix< Real, Device, Index > :: load( file ) ) return false;
   if( ! this -> nonzeroElements. load( file ) ) return false;
   if( ! this -> diagonalsOffsets. load( file ) ) return false;
   if( ! file. read( &this -> numberOfDiagonals, 1 ) )
      return false;
   return true;
}

template< typename Real, typename Device, typename Index >
bool tnlMultiDiagonalMatrix< Real, Device, Index > :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
}

template< typename Real, typename Device, typename Index >
bool tnlMultiDiagonalMatrix< Real, Device, Index > :: load( const tnlString& fileName )
{
   return tnlObject :: load( fileName );
}

template< typename Real, typename Device, typename Index >
tnlCSRMatrix< Real, Device, Index >& tnlMultiDiagonalMatrix< Real, Device, Index > :: operator = ( const tnlCSRMatrix< Real, Device, Index >& csrMatrix )
{
   tnlAssert( false, );
}

template< typename Real, typename Device, typename Index >
void tnlMultiDiagonalMatrix< Real, Device, Index > :: printOut( ostream& str,
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



#endif /* TNLMULTIDIAGONALMATRIX_H_ */
