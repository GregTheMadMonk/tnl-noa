/***************************************************************************
                          Ellpack.h  -  description
                             -------------------
    begin                : Jul 28, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef Ellpack_H_
#define Ellpack_H_

#include <iostream>
#include <iomanip>
#include <TNL/Vectors/Vector.h>
#include <TNL/Assert.h>
#include <TNL/core/mfuncs.h>
#include <TNL/Matrices/CSR.h>
#include <TNL/debug/tnlDebug.h>


//! Implementation of the ELLPACK format
template< typename Real, typename Device = Devices::Host, typename Index = int >
class Ellpack
{
};

template< typename Real, typename Index >
class Ellpack< Real, Devices::Host, Index > : public Matrix< Real, Devices::Host, Index >
{
   public:

   typedef Real RealType;
   typedef Devices::Host DeviceType;
   typedef Index IndexType;

   //! Basic constructor
   Ellpack( const String& name, Index _row );

   const String& getMatrixClass() const;

   String getType() const;

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

   bool copyFrom( const CSR< Real, Devices::Host >& csr_matrix );

   Real getElement( Index row,
                    Index column ) const;

   Real rowProduct( Index row,
                    const Vector< Real, Devices::Host, Index >& vector ) const;

   void vectorProduct( const Vector< Real, Devices::Host, Index >& x,
                       Vector< Real, Devices::Host, Index >& b ) const;

   Real getRowL1Norm( Index row ) const
   { abort(); };

   void multiplyRow( Index row, const Real& value )
   { abort(); };

   //! Prints out the matrix structure
   void printOut( std::ostream& str ) const;

   protected:

   Vector< Real > ellpack_nonzero_elements;

   Vector< Index > ellpack_columns;

   Index row_length;

   Index artificial_zeros;

   friend class Ellpack< Real, Devices::Cuda >;
};

template< typename Real, typename Index >
Ellpack< Real, Devices::Host, Index > :: Ellpack( const String& name, Index _row_length )
: Matrix< Real >( name ),
  ellpack_nonzero_elements( "ellpack-nonzero-elements" ),
  ellpack_columns( "ellpack-columns" ),
  row_length( _row_length ),
  artificial_zeros( 0 )
{
   Assert( row_length > 0, );
};

template< typename Real, typename Index >
const String& Ellpack< Real, Devices::Host, Index > :: getMatrixClass() const
{
   return MatrixClass :: main;
};

template< typename Real, typename Index >
String Ellpack< Real, Devices::Host, Index > :: getType() const
{
   return String( "Ellpack< ") + String( getType( Real( 0.0 ) ) ) + String( ", Devices::Host >" );
};

template< typename Real, typename Index >
bool Ellpack< Real, Devices::Host, Index > :: setSize( Index new_size )
{
   this->size = new_size;
   if( ! ellpack_nonzero_elements. setSize( new_size * row_length ) )
      return false;
   ellpack_nonzero_elements. setValue( 0 );
   if( ! ellpack_columns. setSize( new_size * row_length ) )
      return false;
   ellpack_columns. setValue( -1 );
   return true;
};

template< typename Real, typename Index >
void Ellpack< Real, Devices::Host, Index > :: reset()
{
   ellpack_nonzero_elements. reset();
   ellpack_columns. reset();
   row_length = 0;
   artificial_zeros = 0;
}

template< typename Real, typename Index >
Index Ellpack< Real, Devices::Host, Index > :: getRowLength() const
{
   return row_length;
}

template< typename Real, typename Index >
bool Ellpack< Real, Devices::Host, Index > :: setNonzeroElements( Index elements )
{
	return true;
}

template< typename Real, typename Index >
Index Ellpack< Real, Devices::Host, Index > :: getNonzeroElements() const
{
   return ellpack_nonzero_elements. getSize() - artificial_zeros;
}

template< typename Real, typename Index >
Index Ellpack< Real, Devices::Host, Index > :: getArtificialZeroElements() const
{
   return artificial_zeros;
}

template< typename Real, typename Index >
bool Ellpack< Real, Devices::Host, Index > :: copyFrom( const CSR< Real, Devices::Host >& csr_matrix )
{
   dbgFunctionName( "Ellpack< Real, Devices::Host >", "copyFrom" );
   /*if( ! row_length )
   {
      Index min_row_length, max_row_length;
      double average_row_length;
      csr_matrix. getRowStatistics( min_row_length,
                                    max_row_length,
                                    average_row_length );
      row_length = average_row_length;
   }*/

   if( ! this->setSize( csr_matrix. getSize() ) )
   		return false;

   /*
    * Estimate number of non-zero elements which will not fit into the ELLPACK data array.
    * They will be stored in the COO data array.
    */
   Index coo_elements = 0;
   for( Index i = 0; i < this->getSize(); i ++ )
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
   dbgExpr( this->getSize() );
   for( Index i = 0; i < this->getSize(); i ++ )
   {
	   Index csr_row_length = csr_matrix. getRowLength( i );
	   Index csr_row_offset = csr_matrix. row_offsets[ i ];
	   Index j = 0;
	   /*
	    * First insert the data which will fit into the ELLPACK data array.
	    */
	   while( j < csr_row_length && j < row_length )
	   {
		   Index element_pos = j * this->getSize() + i;
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
Real Ellpack< Real, Devices::Host, Index > :: getElement( Index row,
                                                             Index column ) const
{
   dbgFunctionName( "Ellpack< Real, Devices::Host >", "getElement" );
   //cout << "Ellpack getElement: " << row << " " << column << " \r" << std::flush;
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
      element_pos += this->getSize();
   }
   if( i < row_length && ellpack_columns[ element_pos ] == column )
      return ellpack_nonzero_elements[ element_pos ];
   return 0;
};

template< typename Real, typename Index >
Real Ellpack< Real, Devices::Host, Index > :: rowProduct( Index row,
                                                             const Vector< Real, Devices::Host, Index >& vector ) const
{
   Assert( 0 <= row && row < this->getSize(),
              std::cerr << "The row is outside the matrix." );
   Assert( vector. getSize() == this->getSize(),
              std::cerr << "The matrix and vector for multiplication have different sizes. "
                   << "The matrix size is " << this->getSize() << "."
                   << "The vector size is " << vector. getSize() << std::endl; );

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
      element_pos += this->getSize();
   }
   if( i < row_length )
      return product;
   return product;

};

template< typename Real, typename Index >
void Ellpack< Real, Devices::Host, Index > :: vectorProduct( const Vector< Real, Devices::Host, Index >& x,
                                                                Vector< Real, Devices::Host, Index >& b ) const
{
   Assert( x. getSize() == this->getSize(),
              std::cerr << "The matrix and vector for a multiplication have different sizes. "
                   << "The matrix size is " << this->getSize() << "."
                   << "The vector size is " << x. getSize() << std::endl; );
   Assert( b. getSize() == this->getSize(),
              std::cerr << "The matrix and result vector of a multiplication have different sizes. "
                   << "The matrix size is " << this->getSize() << "."
                   << "The vector size is " << b. getSize() << std::endl; );

   for( Index i = 0; i < this->getSize(); i ++)
      b[ i ] = rowProduct( i, x );
};

template< typename Real, typename Index >
void Ellpack< Real, Devices::Host, Index > :: printOut( std::ostream& str,
                                                           const String& name ) const
{
   str << "Structure of Ellpack" << std::endl;
   str << "Matrix name:" << name << std::endl;
   str << "Matrix size:" << this->getSize() << std::endl;
   str << "Allocated elements:" << ellpack_nonzero_elements. getSize() << std::endl;
   str << "Matrix row length:" << row_length << std::endl;
   for( Index i = 0; i < this->size; i ++ )
   {
      str << i << "th row data:    ";
      for( Index j = 0; j < row_length; j ++ )
         str << std::setprecision( 5 ) << std::setw( 8 ) << ellpack_nonzero_elements[ i + j * this->getSize() ] << " ";

      str << std::endl << i << "th row columns: ";
      for( Index j = 0; j < row_length; j ++ )
         str << std::setprecision( 5 ) << std::setw( 8 ) << ellpack_columns[ i + j * this->getSize() ] << " ";
      str << std::endl;
   }
}

#endif /* Ellpack_H_ */
