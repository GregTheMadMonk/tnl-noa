/***************************************************************************
                          tnlFullMatrix.h  -  description
                             -------------------
    begin                : 2007/07/23
    copyright            : (C) 2007 by Tomas Oberhuber
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

#ifndef tnlFullMatrixH
#define tnlFullMatrixH

#include <core/tnlArray.h>
#include <matrix/tnlMatrix.h>

const int tnlMaxFullMatrixSize = 65536;

template< typename Real, tnlDevice Device = tnlHost, typename Index = int >
class tnlFullMatrix : public tnlMatrix< Real, Device, Index >,
                      virtual public tnlArray< 2, Real, Device, Index >
{

   Index nonzero_elements;

   public:

   //! Basic constructor
   tnlFullMatrix( const tnlString& name );

   //! Constructor with matrix dimension
   tnlFullMatrix( const Index size );

   tnlString getType() const;

   const tnlString& getMatrixClass() const;

   bool setSize( Index new_size );

   bool setNonzeroElements( Index n );

   void reset();

   Index getNonzeroElements() const;

   Index getSize() const;
   
   Real getElement( Index i, Index j ) const;

   bool setElement( Index i, Index j, const Real& v );

   bool addToElement( Index i, Index j, const Real& v );

   Real rowProduct( const Index row,
                    const tnlLongVector< Real, Device, Index >& vec ) const;

   void vectorProduct( const tnlLongVector< Real, Device, Index >& vec,
                       tnlLongVector< Real, Device, Index >& result ) const;


   //! Multiply row
   void multiplyRow( const Index row, const Real& c );

   //! Get row L1 norm
   Real getRowL1Norm( const Index row ) const;

   bool operator == ( const tnlMatrix< Real, Device, Index >& m ) const;

   bool operator != ( const tnlMatrix< Real, Device, Index >& m ) const;

   //! Destructor
   ~tnlFullMatrix();
};

template< typename Real, tnlDevice Device, typename Index >
tnlFullMatrix< Real, Device, Index > :: tnlFullMatrix( const tnlString& name )
: tnlArray< 2, Real, Device, Index >( name ),
  tnlMatrix< Real, Device, Index >( name ),
  nonzero_elements( 0 )
{
};

template< typename Real, tnlDevice Device, typename Index >
tnlFullMatrix< Real, Device, Index > :: tnlFullMatrix( const Index size )
: tnlArray< 2, Real, Device, Index >( size, size )
{
};

template< typename Real, tnlDevice Device, typename Index >
tnlString tnlFullMatrix< Real, Device, Index > :: getType() const
{
   Real t;
   return tnlString( "tnlFullMatrix< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
};

template< typename Real, tnlDevice Device, typename Index >
const tnlString& tnlFullMatrix< Real, Device, Index > :: getMatrixClass() const
{
   return tnlMatrixClass :: main;
};

template< typename Real, tnlDevice Device, typename Index >
bool tnlFullMatrix< Real, Device, Index > :: setSize( Index new_size )
{
   if( new_size > tnlMaxFullMatrixSize )
   {
      cerr << "The matrix size " << new_size << " is too big. " << endl;
      cerr << "If you really need to allocate such matrix increase the limit constant in the file " << __FILE__ << endl;
      return false;
   }
   tnlMatrix< Real, Device, Index > :: size = 0;
   if( ! tnlArray< 2, Real, Device, Index > :: setDimensions( tnlVector< 2, Index >( new_size, new_size ) ) )
      return false;
   tnlMatrix< Real, Device, Index > :: size = new_size;
   tnlArray< 2, Real, Device, Index > :: setValue( 0.0 );
   nonzero_elements = 0;
   return true;
};

template< typename Real, tnlDevice Device, typename Index >
bool tnlFullMatrix< Real, Device, Index > :: setNonzeroElements( Index n )
{
   return true;
};

template< typename Real, tnlDevice Device, typename Index >
void tnlFullMatrix< Real, Device, Index > :: reset()
{
   tnlArray< 2, Real, Device, Index > :: reset();
};

template< typename Real, tnlDevice Device, typename Index >
Index tnlFullMatrix< Real, Device, Index > :: getNonzeroElements() const
{
   return nonzero_elements;
};

template< typename Real, tnlDevice Device, typename Index >
Index tnlFullMatrix< Real, Device, Index > :: getSize() const
{
   return tnlMatrix< Real, Device, Index > :: getSize(); // it is the same as GetYSize()
};

template< typename Real, tnlDevice Device, typename Index >
Real tnlFullMatrix< Real, Device, Index > :: getElement( Index i, Index j ) const
{
   return tnlArray< 2, Real, Device, Index > :: getElement( i, j );
};

template< typename Real, tnlDevice Device, typename Index >
bool tnlFullMatrix< Real, Device, Index > :: setElement( Index i, Index j, const Real& v )
{
   Real d = tnlArray< 2, Real, Device, Index > :: getElement( i, j );
   if( d == Real( 0.0 ) && v != Real( 0.0 ) )
      nonzero_elements ++;
   if( d != Real( 0.0 ) && v == Real( 0.0 ) )
      nonzero_elements --;
   tnlArray< 2, Real, Device, Index > :: setElement( i, j, v );
    return true;
};

template< typename Real, tnlDevice Device, typename Index >
bool tnlFullMatrix< Real, Device, Index > :: addToElement( Index i, Index j, const Real& v )
{
   Real d1 = tnlArray< 2, Real, Device, Index > :: getElement( i, j );
   Real d2 = d1;
   d1 += v;
   if( d2 == Real( 0.0 ) && d1 != Real( 0.0 ) )
      nonzero_elements ++;
   if( d2 != Real( 0.0 ) && d1 == Real( 0.0 ) )
      nonzero_elements --;
        tnlArray< 2, Real, Device, Index > :: setElement( i, j, d1 );
   return true;
};

template< typename Real, tnlDevice Device, typename Index >
Real tnlFullMatrix< Real, Device, Index > :: rowProduct( const Index row,
                                                         const tnlLongVector< Real, Device, Index >& vec ) const
{
   tnlAssert( 0 <= row && row < this -> getSize(),
              cerr << "The row is outside the matrix." );
   tnlAssert( vec. getSize() == this -> getSize(),
              cerr << "The matrix and vector for multiplication have different sizes. "
                   << "The matrix size is " << this -> getSize() << "."
                   << "The vector size is " << vec. getSize() << endl; );

   const Index size = getSize();
   Index pos = row * size;
   const Real* data = tnlArray< 2, Real, Device, Index > :: getVector();
   Real res( 0.0 );
   if( Device == tnlHost )
   {
      for( Index i = 0; i < size; i ++ )
      {
         res += data[ pos ] * vec[ i ];
         pos ++;
      }
   }
   if( Device == tnlCuda )
   {
      tnlAssert( false, );
      //TODO: implement this
   }
   return res;
};

template< typename Real, tnlDevice Device, typename Index >
void tnlFullMatrix< Real, Device, Index > :: vectorProduct( const tnlLongVector< Real, Device, Index >& vec,
                                                            tnlLongVector< Real, Device, Index >& result ) const
{
   tnlAssert( vec. getSize() == this -> getSize(),
              cerr << "The matrix and vector for a multiplication have different sizes. "
                   << "The matrix size is " << this -> getSize() << "."
                   << "The vector size is " << vec. getSize() << endl; );
   tnlAssert( result. getSize() == this -> getSize(),
              cerr << "The matrix and result vector of a multiplication have different sizes. "
                   << "The matrix size is " << this -> getSize() << "."
                   << "The vector size is " << result. getSize() << endl; );

   const Index size = getSize();
   Index pos( 0 );
   const Real* data = tnlArray< 2, Real, Device, Index > :: getVector();
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


template< typename Real, tnlDevice Device, typename Index >
void tnlFullMatrix< Real, Device, Index > :: multiplyRow( const Index row, const Real& c )
{
   const Index size = getSize();
   Real* data = tnlArray< 2, Real, Device, Index > :: getVector();
   Index pos = row * size;
   for( Index i = 0; i < size; i ++ )
   {
      data[ pos + i ] *= c;
   }
};

template< typename Real, tnlDevice Device, typename Index >
Real tnlFullMatrix< Real, Device, Index > :: getRowL1Norm( const Index row ) const
{
   const Index size = getSize();
   const Real* data = tnlArray< 2, Real, Device, Index > :: getVector();
   Real res( 0.0 );
   Index pos = row * size;
   for( Index i = 0; i < size; i ++ )
      res += fabs( data[ pos + i ] );
   return res;
};

template< typename Real, tnlDevice Device, typename Index >
bool tnlFullMatrix< Real, Device, Index > :: operator == ( const tnlMatrix< Real, Device, Index >& m ) const
{
   return tnlMatrix< Real, Device, Index > :: operator == ( m );
};

template< typename Real, tnlDevice Device, typename Index >
bool tnlFullMatrix< Real, Device, Index > :: operator != ( const tnlMatrix< Real, Device, Index >& m ) const
{
   return tnlMatrix< Real, Device, Index > :: operator != ( m );
}

template< typename Real, tnlDevice Device, typename Index >
tnlFullMatrix< Real, Device, Index > :: ~tnlFullMatrix()
{
};


//! Matrix product
template< typename Real, tnlDevice Device, typename Index >
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
template< typename Real, tnlDevice Device, typename Index >
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

template< typename Real, tnlDevice Device, typename Index >
ostream& operator << ( ostream& o_str, const tnlFullMatrix< Real, Device, Index >& A )
{
   return operator << ( o_str, ( const tnlMatrix< Real, Device, Index >& ) A );
};

#endif
