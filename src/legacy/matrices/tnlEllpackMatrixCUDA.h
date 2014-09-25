/***************************************************************************
                          tnlEllpackMatrixCUDA.h  -  description
                             -------------------
    begin                : Aug 1, 2010
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

#ifndef TNLELLPACKMATRIXCUDA_H_
#define TNLELLPACKMATRIXCUDA_H_

#include <matrices/tnlEllpackMatrix.h>

template< typename Real, typename Index >
class tnlEllpackMatrix< Real, tnlCuda, Index > : public tnlMatrix< Real, tnlCuda, Index >
{
   public:

   typedef Real RealType;
   typedef tnlCuda DeviceType;
   typedef Index IndexType;

   //! Basic constructor
   tnlEllpackMatrix( const tnlString& name, Index _row );

   const tnlString& getMatrixClass() const;

   tnlString getType() const;

   //! Sets the number of row and columns.
   bool setSize( Index new_size );

   void reset();

   Index getRowLength() const;

   bool setNonzeroElements( Index elements );

   //! Allocate memory for the nonzero elements stored in the COO data arrays.
   bool setNonzeroCOOElements( Index elements );

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

   bool copyFrom( const tnlEllpackMatrix< Real, tnlHost >& ellpack_matrix );

   Real getElement( Index row,
                    Index column ) const;

   Real rowProduct( Index row,
                    const tnlVector< Real, tnlCuda, Index >& vector ) const;

   void vectorProduct( const tnlVector< Real, tnlCuda, Index >& x,
                       tnlVector< Real, tnlCuda, Index >& b ) const;

   Real getRowL1Norm( Index row ) const
   { abort(); };

   void multiplyRow( Index row, const Real& value )
   { abort(); };

   //! Prints out the matrix structure
   void printOut( ostream& str ) const;

   protected:

   Index findCOORow( Index row ) const;

   tnlVector< Real, tnlCuda > ellpack_nonzero_elements;

   tnlVector< Index, tnlCuda > ellpack_columns;

   tnlVector< Real, tnlCuda > coo_nonzero_elements;

   tnlVector< Index, tnlCuda > coo_rows;

   tnlVector< Index, tnlCuda > coo_columns;

   Index row_length;

   Index artificial_zeros;

   //! The last non-zero element is at the position last_non_zero_element - 1
   Index last_coo_nonzero_element;
};

#ifdef HAVE_CUDA
template< typename Real, typename Index >
__global__ void sparseEllpackMatrixVectorProductKernel( Index size,
                                                        Index row_length,
                                                        const Real* ellpack_nonzero_elements,
                                                        const Index* ellpack_columns,
                                                        const Real* vec_x,
                                                        Real* vec_b )
{
   /*
    * Each thread process one matrix row
    */
   Index row = blockIdx. x * blockDim. x + threadIdx. x;
   if( row >= size )
      return;

   Real product( 0.0 );
   Index element_pos = row;
   Index i = 0;
   while( i < row_length &&
          ellpack_columns[ element_pos ] != -1 )
   {
      product += ellpack_nonzero_elements[ element_pos ] *
                    vec_x[ ellpack_columns[ element_pos ] ];
      i ++;
      element_pos += size;
   }
   vec_b[ row ] = product;
}
#endif


template< typename Real, typename Index >
tnlEllpackMatrix< Real, tnlCuda, Index > :: tnlEllpackMatrix( const tnlString& name, Index _row_length )
: tnlMatrix< Real >( name ),
  ellpack_nonzero_elements( "ellpack-nonzero-elements" ),
  ellpack_columns( "ellpack-columns" ),
  coo_nonzero_elements( "coo-nonzero-elements" ),
  coo_rows( "coo-rows" ),
  coo_columns( "coo-columns" ),
  row_length( _row_length ),
  artificial_zeros( 0 ),
  last_coo_nonzero_element( 0 )
{
   tnlAssert( row_length > 0, );
};

template< typename Real, typename Index >
const tnlString& tnlEllpackMatrix< Real, tnlCuda, Index > :: getMatrixClass() const
{
   return tnlMatrixClass :: main;
};

template< typename Real, typename Index >
tnlString tnlEllpackMatrix< Real, tnlCuda, Index > :: getType() const
{
   return tnlString( "tnlEllpackMatrix< ") + tnlString( getType( Real( 0.0 ) ) ) + tnlString( ", tnlCuda >" );
};

template< typename Real, typename Index >
bool tnlEllpackMatrix< Real, tnlCuda, Index > :: setSize( Index new_size )
{
   this -> size = new_size;
   if( ! ellpack_nonzero_elements. setSize( new_size * row_length ) )
      return false;
   ellpack_nonzero_elements. setValue( 0.0 );
   if( ! ellpack_columns. setSize( new_size * row_length ) )
      return false;
   ellpack_columns. setValue( -1 );
   return true;
};

template< typename Real, typename Index >
Index tnlEllpackMatrix< Real, tnlCuda, Index > :: getRowLength() const
{
   return row_length;
}

template< typename Real, typename Index >
bool tnlEllpackMatrix< Real, tnlCuda, Index > :: setNonzeroElements( Index elements )
{
   return true;
}

template< typename Real, typename Index >
bool tnlEllpackMatrix< Real, tnlCuda, Index > :: setNonzeroCOOElements( Index elements )
{
   if( ! coo_nonzero_elements. setSize( elements ) ||
      ! coo_rows. setSize( elements ) ||
      ! coo_columns. setSize( elements ) )
   return false;
   coo_nonzero_elements. setValue( 0.0 );
   coo_rows. setValue( -1 );
   coo_columns. setValue( -1 );
   last_coo_nonzero_element = 0;
   return true;
};

template< typename Real, typename Index >
void tnlEllpackMatrix< Real, tnlCuda, Index > :: reset()
{
   ellpack_nonzero_elements. reset();
   ellpack_columns. reset();
   coo_nonzero_elements. reset();
   coo_rows. reset();
   coo_columns. reset();
   row_length = 0;
   artificial_zeros = 0;
   last_coo_nonzero_element = 0;
};

template< typename Real, typename Index >
Index tnlEllpackMatrix< Real, tnlCuda, Index > :: getNonzeroElements() const
{
   return coo_nonzero_elements. getSize() + ellpack_nonzero_elements. getSize() - artificial_zeros;
};

template< typename Real, typename Index >
Index tnlEllpackMatrix< Real, tnlCuda, Index > :: getArtificialZeroElements() const
{
   return artificial_zeros;
};

template< typename Real, typename Index >
bool tnlEllpackMatrix< Real, tnlCuda, Index > :: copyFrom( const tnlEllpackMatrix< Real, tnlHost >& ellpack_matrix )
{
   dbgFunctionName( "tnlEllpackMatrix< Real, tnlCuda >", "copyFrom" );

   row_length = ellpack_matrix. getRowLength();
   if( ! this -> setSize( ellpack_matrix. getSize() ) )
   		return false;

   if( ! setNonzeroCOOElements( ellpack_matrix. coo_nonzero_elements. getSize() ) )
	   return false;

   ellpack_nonzero_elements. copyFrom( ellpack_matrix. ellpack_nonzero_elements );
   ellpack_columns = ellpack_matrix. ellpack_columns;

   artificial_zeros = ellpack_matrix. artificial_zeros;

   return true;
};

template< typename Real, typename Index >
Real tnlEllpackMatrix< Real, tnlCuda, Index > :: getElement( Index row,
                                                             Index column ) const
{
	tnlAssert( false, );
};

template< typename Real, typename Index >
void tnlEllpackMatrix< Real, tnlCuda, Index > :: vectorProduct( const tnlVector< Real, tnlCuda, Index >& x,
                                                                tnlVector< Real, tnlCuda, Index >& b ) const
{
   tnlAssert( x. getSize() == this -> getSize(),
              cerr << "The matrix and vector for a multiplication have different sizes. "
                   << "The matrix size is " << this -> getSize() << "."
                   << "The vector size is " << x. getSize() << endl; );
   tnlAssert( b. getSize() == this -> getSize(),
              cerr << "The matrix and result vector of a multiplication have different sizes. "
                   << "The matrix size is " << this -> getSize() << "."
                   << "The vector size is " << b. getSize() << endl; );
#ifdef HAVE_CUDA
	sparseEllpackMatrixVectorProductKernelCaller( this -> getSize(),
	                                       row_length,
	                                       ellpack_nonzero_elements. getData(),
	                                       ellpack_columns. getData(),
	                                       x,
	                                       b );
#else
	tnlCudaSupportMissingMessage;;
#endif
};

template< typename Real, typename Index >
Real tnlEllpackMatrix< Real, tnlCuda, Index > :: rowProduct( Index row,
                                                             const tnlVector< Real, tnlCuda, Index >& vector ) const
{
   tnlAssert( vector. getSize() == this -> getSize(),
              cerr << "The matrix and vector for a multiplication have different sizes. "
                   << "The matrix size is " << this -> getSize() << "."
                   << "The vector size is " << vector. getSize() << endl; );

	tnlAssert( false, );
};

template< typename Real, typename Index >
void tnlEllpackMatrix< Real, tnlCuda, Index > :: printOut( ostream& str ) const
{
	tnlAssert( false, );
};

#endif /* TNLELLPACKMATRIXCUDA_H_ */
