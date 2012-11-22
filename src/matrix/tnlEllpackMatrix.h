/***************************************************************************
                          tnlEllpackMatrix.h  -  description
                             -------------------
    begin                : Jul 28, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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

#ifndef TNLELLPACKMATRIX_H_
#define TNLELLPACKMATRIX_H_

#include <iostream>
#include <iomanip>
#include <core/tnlVector.h>
#include <core/tnlAssert.h>
#include <core/mfuncs.h>
#include <matrix/tnlCSRMatrix.h>
#include <debug/tnlDebug.h>

using namespace std;

//! Implementation of the ELLPACK format
template< typename Real, typename device = tnlHost, typename Index = int >
class tnlEllpackMatrix
{
};

template< typename Real, typename Index >
class tnlEllpackMatrix< Real, tnlHost, Index > : public tnlMatrix< Real, tnlHost, Index >
{
   public:
   //! Basic constructor
   tnlEllpackMatrix( const tnlString& name, Index _row );

   const tnlString& getMatrixClass() const;

   tnlString getType() const;

   //! Sets the number of row and columns.
   bool setSize( Index new_size );

   Index getRowLength() const;

   bool setNonzeroElements( Index elements );

   void reset();

   Index getNonzeroElements() const;

   Index getArtificialZeroElements() const;

   bool setElement( Index row,
                    Index colum,
                    const Real& value )
   { abort(); };

   bool addToElement( Index row,
                      Index column,
                      const Real& value )
   { abort(); };

   bool copyFrom( const tnlCSRMatrix< Real, tnlHost >& csr_matrix );

   Real getElement( Index row,
                    Index column ) const;

   Real rowProduct( Index row,
                    const tnlVector< Real, tnlHost, Index >& vector ) const;

   void vectorProduct( const tnlVector< Real, tnlHost, Index >& x,
                       tnlVector< Real, tnlHost, Index >& b ) const;

   Real getRowL1Norm( Index row ) const
   { abort(); };

   void multiplyRow( Index row, const Real& value )
   { abort(); };

   //! Prints out the matrix structure
   void printOut( ostream& str ) const;

   protected:

   tnlVector< Real > ellpack_nonzero_elements;

   tnlVector< Index > ellpack_columns;

   Index row_length;

   Index artificial_zeros;

   friend class tnlEllpackMatrix< Real, tnlCuda >;
};

template< typename Real, typename Index >
tnlEllpackMatrix< Real, tnlHost, Index > :: tnlEllpackMatrix( const tnlString& name, Index _row_length )
: tnlMatrix< Real >( name ),
  ellpack_nonzero_elements( "ellpack-nonzero-elements" ),
  ellpack_columns( "ellpack-columns" ),
  row_length( _row_length ),
  artificial_zeros( 0 )
{
   tnlAssert( row_length > 0, );
};

template< typename Real, typename Index >
const tnlString& tnlEllpackMatrix< Real, tnlHost, Index > :: getMatrixClass() const
{
   return tnlMatrixClass :: main;
};

template< typename Real, typename Index >
tnlString tnlEllpackMatrix< Real, tnlHost, Index > :: getType() const
{
   return tnlString( "tnlEllpackMatrix< ") + tnlString( GetParameterType( Real( 0.0 ) ) ) + tnlString( ", tnlHost >" );
};

template< typename Real, typename Index >
bool tnlEllpackMatrix< Real, tnlHost, Index > :: setSize( Index new_size )
{
   this -> size = new_size;
   if( ! ellpack_nonzero_elements. setSize( new_size * row_length ) )
      return false;
   ellpack_nonzero_elements. setValue( 0 );
   if( ! ellpack_columns. setSize( new_size * row_length ) )
      return false;
   ellpack_columns. setValue( -1 );
   return true;
};

template< typename Real, typename Index >
void tnlEllpackMatrix< Real, tnlHost, Index > :: reset()
{
   ellpack_nonzero_elements. reset();
   ellpack_columns. reset();
   row_length = 0;
   artificial_zeros = 0;
}

template< typename Real, typename Index >
Index tnlEllpackMatrix< Real, tnlHost, Index > :: getRowLength() const
{
   return row_length;
}

template< typename Real, typename Index >
bool tnlEllpackMatrix< Real, tnlHost, Index > :: setNonzeroElements( Index elements )
{
	return true;
}

template< typename Real, typename Index >
Index tnlEllpackMatrix< Real, tnlHost, Index > :: getNonzeroElements() const
{
   return ellpack_nonzero_elements. getSize() - artificial_zeros;
}

template< typename Real, typename Index >
Index tnlEllpackMatrix< Real, tnlHost, Index > :: getArtificialZeroElements() const
{
   return artificial_zeros;
}

template< typename Real, typename Index >
bool tnlEllpackMatrix< Real, tnlHost, Index > :: copyFrom( const tnlCSRMatrix< Real, tnlHost >& csr_matrix )
{
   dbgFunctionName( "tnlEllpackMatrix< Real, tnlHost >", "copyFrom" );
   /*if( ! row_length )
   {
      Index min_row_length, max_row_length;
      double average_row_length;
      csr_matrix. getRowStatistics( min_row_length,
                                    max_row_length,
                                    average_row_length );
      row_length = average_row_length;
   }*/

   if( ! this -> setSize( csr_matrix. getSize() ) )
   		return false;

   /*
    * Estimate number of non-zero elements which will not fit into the ELLPACK data array.
    * They will be stored in the COO data array.
    */
   Index coo_elements = 0;
   for( Index i = 0; i < this -> getSize(); i ++ )
   {
	   Index csr_row_length = csr_matrix. getRowLength( i );
	   if( csr_row_length > row_length )
		   coo_elements += csr_row_length - row_length;
   }
   dbgExpr( coo_elements );

   /*
    * Insert the data now.
    */
   dbgCout( "Inserting CSR row ... ");
   artificial_zeros = 0;
   dbgExpr( this -> getSize() );
   for( Index i = 0; i < this -> getSize(); i ++ )
   {
	   Index csr_row_length = csr_matrix. getRowLength( i );
	   Index csr_row_offset = csr_matrix. row_offsets[ i ];
	   Index j = 0;
	   /*
	    * First insert the data which will fit into the ELLPACK data array.
	    */
	   while( j < csr_row_length && j < row_length )
	   {
		   Index element_pos = j * this -> getSize() + i;
		   ellpack_nonzero_elements[ element_pos ] = csr_matrix. nonzero_elements[ csr_row_offset + j ];
		   ellpack_columns[ element_pos ] = csr_matrix. columns[ csr_row_offset + j ];
		   dbgExpr( element_pos );
		   dbgExpr( csr_matrix. nonzero_elements[ csr_row_offset + j ] );
		   dbgExpr( csr_matrix. columns[ csr_row_offset + j ] );
		   j ++;
	   }
	   if( j < row_length )
		   artificial_zeros += row_length - j;
   }
   return true;
}

template< typename Real, typename Index >
Real tnlEllpackMatrix< Real, tnlHost, Index > :: getElement( Index row,
                                                             Index column ) const
{
   dbgFunctionName( "tnlEllpackMatrix< Real, tnlHost >", "getElement" );
   //cout << "Ellpack getElement: " << row << " " << column << " \r" << flush;
   /*
    * We first search in the ELLPACK data arrays.
    */
   Index element_pos = row;
   Index i = 0;
   while( i < row_length &&
          ellpack_columns[ element_pos ] < column &&
          ellpack_columns[ element_pos ] != -1 )
   {
      dbgExpr( element_pos );
      i ++;
      element_pos += this -> getSize();
   }
   if( i < row_length && ellpack_columns[ element_pos ] == column )
      return ellpack_nonzero_elements[ element_pos ];
   return 0;
};

template< typename Real, typename Index >
Real tnlEllpackMatrix< Real, tnlHost, Index > :: rowProduct( Index row,
                                                             const tnlVector< Real, tnlHost, Index >& vector ) const
{
   tnlAssert( 0 <= row && row < this -> getSize(),
              cerr << "The row is outside the matrix." );
   tnlAssert( vector. getSize() == this -> getSize(),
              cerr << "The matrix and vector for multiplication have different sizes. "
                   << "The matrix size is " << this -> getSize() << "."
                   << "The vector size is " << vector. getSize() << endl; );

   Real product( 0.0 );
   /****
    * We first multiply the vector using the data in the ELLPACK array.
    */
   Index element_pos = row;
   Index i = 0;
   while( i < row_length &&
          ellpack_columns[ element_pos ] != -1 )
   {
      product += ellpack_nonzero_elements[ element_pos ] *
                 vector[ ellpack_columns[ element_pos ] ];
      i ++;
      element_pos += this -> getSize();
   }
   if( i < row_length )
      return product;
   return product;

};

template< typename Real, typename Index >
void tnlEllpackMatrix< Real, tnlHost, Index > :: vectorProduct( const tnlVector< Real, tnlHost, Index >& x,
                                                                tnlVector< Real, tnlHost, Index >& b ) const
{
   tnlAssert( x. getSize() == this -> getSize(),
              cerr << "The matrix and vector for a multiplication have different sizes. "
                   << "The matrix size is " << this -> getSize() << "."
                   << "The vector size is " << x. getSize() << endl; );
   tnlAssert( b. getSize() == this -> getSize(),
              cerr << "The matrix and result vector of a multiplication have different sizes. "
                   << "The matrix size is " << this -> getSize() << "."
                   << "The vector size is " << b. getSize() << endl; );

   for( Index i = 0; i < this -> getSize(); i ++)
      b[ i ] = rowProduct( i, x );
};

template< typename Real, typename Index >
void tnlEllpackMatrix< Real, tnlHost, Index > :: printOut( ostream& str ) const
{
   str << "Structure of tnlEllpackMatrix" << endl;
   str << "Matrix name:" << this -> getName() << endl;
   str << "Matrix size:" << this -> getSize() << endl;
   str << "Allocated elements:" << ellpack_nonzero_elements. getSize() << endl;
   str << "Matrix row length:" << row_length << endl;
   for( Index i = 0; i < this -> size; i ++ )
   {
      str << i << "th row data:    ";
      for( Index j = 0; j < row_length; j ++ )
         str << setprecision( 5 ) << setw( 8 ) << ellpack_nonzero_elements[ i + j * this -> getSize() ] << " ";

      str << endl << i << "th row columns: ";
      for( Index j = 0; j < row_length; j ++ )
         str << setprecision( 5 ) << setw( 8 ) << ellpack_columns[ i + j * this -> getSize() ] << " ";
      str << endl;
   }
}



#endif /* TNLELLPACKMATRIX_H_ */
