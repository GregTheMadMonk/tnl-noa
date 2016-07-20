/***************************************************************************
                          tnlFullMatrix.h  -  description
                             -------------------
    begin                : 2007/07/23
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef tnlFullMatrixH
#define tnlFullMatrixH

#include <TNL/Arrays/MultiArray.h>
#include <TNL/matrices/tnlMatrix.h>

const int tnlMaxFullMatrixSize = 65536;

template< typename Real, typename Device = tnlHost, typename Index = int >
class tnlFullMatrix : public tnlMatrix< Real, Device, Index >,
                      virtual public tnlMultiArray< 2, Real, Device, Index >
{

   Index nonzero_elements;

   public:

   //! Basic constructor
   tnlFullMatrix( const String& name );

   //! Constructor with matrix dimension
   tnlFullMatrix( const Index size );

   String getType() const;

   const String& getMatrixClass() const;

   bool setSize( Index new_size );

   bool setNonzeroElements( Index n );

   void reset();

   Index getNonzeroElements() const;

   Index getSize() const;
 
   Real getElement( Index i, Index j ) const;

   bool setElement( Index i, Index j, const Real& v );

   bool addToElement( Index i, Index j, const Real& v );

   Real rowProduct( const Index row,
                    const tnlVector< Real, Device, Index >& vec ) const;

   void vectorProduct( const tnlVector< Real, Device, Index >& vec,
                       tnlVector< Real, Device, Index >& result ) const;


   //! Multiply row
   void multiplyRow( const Index row, const Real& c );

   //! Get row L1 norm
   Real getRowL1Norm( const Index row ) const;

   bool operator == ( const tnlMatrix< Real, Device, Index >& m ) const;

   bool operator != ( const tnlMatrix< Real, Device, Index >& m ) const;

   //! Destructor
   ~tnlFullMatrix();
};

template< typename Real, typename Device, typename Index >
tnlFullMatrix< Real, Device, Index > :: tnlFullMatrix( const String& name )
: tnlMultiArray< 2, Real, Device, Index >( name ),
  tnlMatrix< Real, Device, Index >( name ),
  nonzero_elements( 0 )
{
};

template< typename Real, typename Device, typename Index >
tnlFullMatrix< Real, Device, Index > :: tnlFullMatrix( const Index size )
: tnlMultiArray< 2, Real, Device, Index >( size, size )
{
};

template< typename Real, typename Device, typename Index >
String tnlFullMatrix< Real, Device, Index > :: getType() const
{
   Real t;
   return String( "tnlFullMatrix< " ) + String( getType( t ) ) + String( " >" );
};

template< typename Real, typename Device, typename Index >
const String& tnlFullMatrix< Real, Device, Index > :: getMatrixClass() const
{
   return tnlMatrixClass :: main;
};

template< typename Real, typename Device, typename Index >
bool tnlFullMatrix< Real, Device, Index > :: setSize( Index new_size )
{
   if( new_size > tnlMaxFullMatrixSize )
   {
      std::cerr << "The matrix size " << new_size << " is too big. " << std::endl;
      std::cerr << "If you really need to allocate such matrix increase the limit constant in the file " << __FILE__ << std::endl;
      return false;
   }
   tnlMatrix< Real, Device, Index > :: size = 0;
   if( ! tnlMultiArray< 2, Real, Device, Index > :: setDimensions( tnlStaticVector< 2, Index >( new_size, new_size ) ) )
      return false;
   tnlMatrix< Real, Device, Index > :: size = new_size;
   tnlMultiArray< 2, Real, Device, Index > :: setValue( 0.0 );
   nonzero_elements = 0;
   return true;
};

template< typename Real, typename Device, typename Index >
bool tnlFullMatrix< Real, Device, Index > :: setNonzeroElements( Index n )
{
   return true;
};

template< typename Real, typename Device, typename Index >
void tnlFullMatrix< Real, Device, Index > :: reset()
{
   tnlMultiArray< 2, Real, Device, Index > :: reset();
};

template< typename Real, typename Device, typename Index >
Index tnlFullMatrix< Real, Device, Index > :: getNonzeroElements() const
{
   return nonzero_elements;
};

template< typename Real, typename Device, typename Index >
Index tnlFullMatrix< Real, Device, Index > :: getSize() const
{
   return tnlMatrix< Real, Device, Index > :: getSize(); // it is the same as GetYSize()
};

template< typename Real, typename Device, typename Index >
Real tnlFullMatrix< Real, Device, Index > :: getElement( Index i, Index j ) const
{
   return tnlMultiArray< 2, Real, Device, Index > :: getElement( i, j );
};

template< typename Real, typename Device, typename Index >
bool tnlFullMatrix< Real, Device, Index > :: setElement( Index i, Index j, const Real& v )
{
   Real d = tnlMultiArray< 2, Real, Device, Index > :: getElement( i, j );
   if( d == Real( 0.0 ) && v != Real( 0.0 ) )
      nonzero_elements ++;
   if( d != Real( 0.0 ) && v == Real( 0.0 ) )
      nonzero_elements --;
   tnlMultiArray< 2, Real, Device, Index > :: setElement( i, j, v );
    return true;
};

template< typename Real, typename Device, typename Index >
bool tnlFullMatrix< Real, Device, Index > :: addToElement( Index i, Index j, const Real& v )
{
   Real d1 = tnlMultiArray< 2, Real, Device, Index > :: getElement( i, j );
   Real d2 = d1;
   d1 += v;
   if( d2 == Real( 0.0 ) && d1 != Real( 0.0 ) )
      nonzero_elements ++;
   if( d2 != Real( 0.0 ) && d1 == Real( 0.0 ) )
      nonzero_elements --;
        tnlMultiArray< 2, Real, Device, Index > :: setElement( i, j, d1 );
   return true;
};

template< typename Real, typename Device, typename Index >
Real tnlFullMatrix< Real, Device, Index > :: rowProduct( const Index row,
                                                         const tnlVector< Real, Device, Index >& vec ) const
{
   Assert( 0 <= row && row < this->getSize(),
              std::cerr << "The row is outside the matrix." );
   Assert( vec. getSize() == this->getSize(),
              std::cerr << "The matrix and vector for multiplication have different sizes. "
                   << "The matrix size is " << this->getSize() << "."
                   << "The vector size is " << vec. getSize() << std::endl; );

   const Index size = getSize();
   Index pos = row * size;
   const Real* data = tnlMultiArray< 2, Real, Device, Index > :: getData();
   Real res( 0.0 );
   if( Device :: getDevice() == tnlHostDevice )
   {
      for( Index i = 0; i < size; i ++ )
      {
         res += data[ pos ] * vec[ i ];
         pos ++;
      }
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {
      Assert( false, );
      //TODO: implement this
   }
   return res;
};

template< typename Real, typename Device, typename Index >
void tnlFullMatrix< Real, Device, Index > :: vectorProduct( const tnlVector< Real, Device, Index >& vec,
                                                            tnlVector< Real, Device, Index >& result ) const
{
   Assert( vec. getSize() == this->getSize(),
              std::cerr << "The matrix and vector for a multiplication have different sizes. "
                   << "The matrix size is " << this->getSize() << "."
                   << "The vector size is " << vec. getSize() << std::endl; );
   Assert( result. getSize() == this->getSize(),
              std::cerr << "The matrix and result vector of a multiplication have different sizes. "
                   << "The matrix size is " << this->getSize() << "."
                   << "The vector size is " << result. getSize() << std::endl; );

   const Index size = getSize();
   Index pos( 0 );
   const Real* data = tnlMultiArray< 2, Real, Device, Index > :: getData();
   Real res;
   for( Index i = 0; i < size; i ++ )
   {
      res = 0.0;
      for( Index j = 0; j < size; j ++ )
      {
         res += data[ pos ] * vec[ j ];
         pos ++;
      }
      result[ i ] = res;
   }
};


template< typename Real, typename Device, typename Index >
void tnlFullMatrix< Real, Device, Index > :: multiplyRow( const Index row, const Real& c )
{
   const Index size = getSize();
   Real* data = tnlMultiArray< 2, Real, Device, Index > :: getData();
   Index pos = row * size;
   for( Index i = 0; i < size; i ++ )
   {
      data[ pos + i ] *= c;
   }
};

template< typename Real, typename Device, typename Index >
Real tnlFullMatrix< Real, Device, Index > :: getRowL1Norm( const Index row ) const
{
   const Index size = getSize();
   const Real* data = tnlMultiArray< 2, Real, Device, Index > :: getData();
   Real res( 0.0 );
   Index pos = row * size;
   for( Index i = 0; i < size; i ++ )
      res += fabs( data[ pos + i ] );
   return res;
};

template< typename Real, typename Device, typename Index >
bool tnlFullMatrix< Real, Device, Index > :: operator == ( const tnlMatrix< Real, Device, Index >& m ) const
{
   return tnlMatrix< Real, Device, Index > :: operator == ( m );
};

template< typename Real, typename Device, typename Index >
bool tnlFullMatrix< Real, Device, Index > :: operator != ( const tnlMatrix< Real, Device, Index >& m ) const
{
   return tnlMatrix< Real, Device, Index > :: operator != ( m );
}

template< typename Real, typename Device, typename Index >
tnlFullMatrix< Real, Device, Index > :: ~tnlFullMatrix()
{
};


//! Matrix product
template< typename Real, typename Device, typename Index >
void MatrixProduct( const tnlFullMatrix< Real, tnlHost, Index >& m1,
                    const tnlFullMatrix< Real, tnlHost, Index >& m2,
                    tnlFullMatrix< Real, tnlHost, Index >& result )
{
   assert( m1. GetSize() == m2. GetSize() && m2. GetSize() == result. GetSize() );
   Index size = result. GetSize();
   for( Index i = 0; i < size; i ++ )
      for( Index j = 0; j < size; j ++ )
      {
         Real res( 0.0 );
         for( Index k = 0; k < size; k ++ )
            res += m1( i, k ) * m2( k, j );
         result( i, j ) = res;
      }
};

//! Matrix sum
template< typename Real, typename Device, typename Index >
void MatrixSum( const tnlFullMatrix< Real, tnlHost, Index >& m1,
                const tnlFullMatrix< Real, tnlHost, Index >& m2,
                tnlFullMatrix< Real, tnlHost, Index >& result )
{
   assert( m1. GetSize() == m2. GetSize() && m2. GetSize() == result. GetSize() );
   Index size = result. GetSize();

   for( Index i = 0; i < size; i ++ )
      for( Index j = 0; j < size; j ++ )
         result( i, j ) = m1( i, j ) + m2( i, j );
};

template< typename Real, typename Device, typename Index >
ostream& operator << ( std::ostream& o_str, const tnlFullMatrix< Real, Device, Index >& A )
{
   return operator << ( o_str, ( const tnlMatrix< Real, Device, Index >& ) A );
};

#endif
