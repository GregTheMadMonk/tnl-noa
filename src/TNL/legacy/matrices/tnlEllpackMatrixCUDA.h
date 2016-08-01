/***************************************************************************
                          EllpackMatrixCUDA.h  -  description
                             -------------------
    begin                : Aug 1, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef EllpackMatrixCUDA_H_
#define EllpackMatrixCUDA_H_

#include <TNL/Matrices/EllpackMatrix.h>

template< typename Real, typename Index >
class EllpackMatrix< Real, Devices::Cuda, Index > : public Matrix< Real, Devices::Cuda, Index >
{
   public:

   typedef Real RealType;
   typedef Devices::Cuda DeviceType;
   typedef Index IndexType;

   //! Basic constructor
   EllpackMatrix( const String& name, Index _row );

   const String& getMatrixClass() const;

   String getType() const;

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

   bool copyFrom( const EllpackMatrix< Real, Devices::Host >& ellpack_matrix );

   Real getElement( Index row,
                    Index column ) const;

   Real rowProduct( Index row,
                    const Vector< Real, Devices::Cuda, Index >& vector ) const;

   void vectorProduct( const Vector< Real, Devices::Cuda, Index >& x,
                       Vector< Real, Devices::Cuda, Index >& b ) const;

   Real getRowL1Norm( Index row ) const
   { abort(); };

   void multiplyRow( Index row, const Real& value )
   { abort(); };

   //! Prints out the matrix structure
   void printOut( std::ostream& str ) const;

   protected:

   Index findCOORow( Index row ) const;

   Vector< Real, Devices::Cuda > ellpack_nonzero_elements;

   Vector< Index, Devices::Cuda > ellpack_columns;

   Vector< Real, Devices::Cuda > coo_nonzero_elements;

   Vector< Index, Devices::Cuda > coo_rows;

   Vector< Index, Devices::Cuda > coo_columns;

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
EllpackMatrix< Real, Devices::Cuda, Index > :: EllpackMatrix( const String& name, Index _row_length )
: Matrix< Real >( name ),
  ellpack_nonzero_elements( "ellpack-nonzero-elements" ),
  ellpack_columns( "ellpack-columns" ),
  coo_nonzero_elements( "coo-nonzero-elements" ),
  coo_rows( "coo-rows" ),
  coo_columns( "coo-columns" ),
  row_length( _row_length ),
  artificial_zeros( 0 ),
  last_coo_nonzero_element( 0 )
{
   Assert( row_length > 0, );
};

template< typename Real, typename Index >
const String& EllpackMatrix< Real, Devices::Cuda, Index > :: getMatrixClass() const
{
   return MatrixClass :: main;
};

template< typename Real, typename Index >
String EllpackMatrix< Real, Devices::Cuda, Index > :: getType() const
{
   return String( "EllpackMatrix< ") + String( getType( Real( 0.0 ) ) ) + String( ", Devices::Cuda >" );
};

template< typename Real, typename Index >
bool EllpackMatrix< Real, Devices::Cuda, Index > :: setSize( Index new_size )
{
   this->size = new_size;
   if( ! ellpack_nonzero_elements. setSize( new_size * row_length ) )
      return false;
   ellpack_nonzero_elements. setValue( 0.0 );
   if( ! ellpack_columns. setSize( new_size * row_length ) )
      return false;
   ellpack_columns. setValue( -1 );
   return true;
};

template< typename Real, typename Index >
Index EllpackMatrix< Real, Devices::Cuda, Index > :: getRowLength() const
{
   return row_length;
}

template< typename Real, typename Index >
bool EllpackMatrix< Real, Devices::Cuda, Index > :: setNonzeroElements( Index elements )
{
   return true;
}

template< typename Real, typename Index >
bool EllpackMatrix< Real, Devices::Cuda, Index > :: setNonzeroCOOElements( Index elements )
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
void EllpackMatrix< Real, Devices::Cuda, Index > :: reset()
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
Index EllpackMatrix< Real, Devices::Cuda, Index > :: getNonzeroElements() const
{
   return coo_nonzero_elements. getSize() + ellpack_nonzero_elements. getSize() - artificial_zeros;
};

template< typename Real, typename Index >
Index EllpackMatrix< Real, Devices::Cuda, Index > :: getArtificialZeroElements() const
{
   return artificial_zeros;
};

template< typename Real, typename Index >
bool EllpackMatrix< Real, Devices::Cuda, Index > :: copyFrom( const EllpackMatrix< Real, Devices::Host >& ellpack_matrix )
{
   dbgFunctionName( "EllpackMatrix< Real, Devices::Cuda >", "copyFrom" );

   row_length = ellpack_matrix. getRowLength();
   if( ! this->setSize( ellpack_matrix. getSize() ) )
   		return false;

   if( ! setNonzeroCOOElements( ellpack_matrix. coo_nonzero_elements. getSize() ) )
	   return false;

   ellpack_nonzero_elements. copyFrom( ellpack_matrix. ellpack_nonzero_elements );
   ellpack_columns = ellpack_matrix. ellpack_columns;

   artificial_zeros = ellpack_matrix. artificial_zeros;

   return true;
};

template< typename Real, typename Index >
Real EllpackMatrix< Real, Devices::Cuda, Index > :: getElement( Index row,
                                                             Index column ) const
{
	Assert( false, );
};

template< typename Real, typename Index >
void EllpackMatrix< Real, Devices::Cuda, Index > :: vectorProduct( const Vector< Real, Devices::Cuda, Index >& x,
                                                                Vector< Real, Devices::Cuda, Index >& b ) const
{
   Assert( x. getSize() == this->getSize(),
              std::cerr << "The matrix and vector for a multiplication have different sizes. "
                   << "The matrix size is " << this->getSize() << "."
                   << "The vector size is " << x. getSize() << std::endl; );
   Assert( b. getSize() == this->getSize(),
              std::cerr << "The matrix and result vector of a multiplication have different sizes. "
                   << "The matrix size is " << this->getSize() << "."
                   << "The vector size is " << b. getSize() << std::endl; );
#ifdef HAVE_CUDA
	sparseEllpackMatrixVectorProductKernelCaller( this->getSize(),
	                                       row_length,
	                                       ellpack_nonzero_elements. getData(),
	                                       ellpack_columns. getData(),
	                                       x,
	                                       b );
#else
	CudaSupportMissingMessage;;
#endif
};

template< typename Real, typename Index >
Real EllpackMatrix< Real, Devices::Cuda, Index > :: rowProduct( Index row,
                                                             const Vector< Real, Devices::Cuda, Index >& vector ) const
{
   Assert( vector. getSize() == this->getSize(),
              std::cerr << "The matrix and vector for a multiplication have different sizes. "
                   << "The matrix size is " << this->getSize() << "."
                   << "The vector size is " << vector. getSize() << std::endl; );

	Assert( false, );
};

template< typename Real, typename Index >
void EllpackMatrix< Real, Devices::Cuda, Index > :: printOut( std::ostream& str ) const
{
	Assert( false, );
};

#endif /* EllpackMatrixCUDA_H_ */
