/***************************************************************************
                          tnlVectorHost.h
                             -------------------
    begin                : Jun 9, 2010
    copyright            : (C) 2009 by Tomas Oberhuber
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

#ifndef TNLVECTORHOST_H_
#define TNLVECTORHOST_H_

#include <core/mfuncs.h>
#include <core/tnlAssert.h>
#include <core/tnlArrayManager.h>
#include <core/tnlVector.h>
#include <debug/tnlDebug.h>

template< typename RealType, typename IndexType >
class tnlVector< RealType, tnlCuda, IndexType >;

template< typename Real, typename Index >
class tnlVector< Real, tnlHost, Index > : public tnlArrayManager< Real, tnlHost, Index >
{
   typedef Real RealType;
   typedef Index IndexType;

   //! We do not allow constructor without parameters.
   tnlVector(){};

   /****
    * We do not allow copy constructors as well to avoid having two
    * vectors with the same name.
    */
   tnlVector( const tnlVector< RealType, tnlHost, IndexType >& v ){};

   tnlVector( const tnlVector< RealType, tnlCuda, IndexType >& v ){};

   public:

   //! Basic constructor with given size
   tnlVector( const tnlString& name, IndexType _size = 0 );

   //! Constructor with another long vector as template
   tnlVector( const tnlString& name, const tnlVector< RealType, tnlHost, IndexType >& v );

   //! Constructor with another long vector as template
   tnlVector( const tnlString& name, const tnlVector< RealType, tnlCuda, IndexType >& v );

   tnlVector< RealType, tnlHost, IndexType >& operator = ( const tnlVector< RealType, tnlHost, IndexType >& long_vector );

   tnlVector< RealType, tnlHost, IndexType >& operator = ( const tnlVector< RealType, tnlCuda, IndexType >& cuda_vector );

   template< typename RealType2, typename IndexType2 >
   tnlVector< RealType, tnlHost, IndexType >& operator = ( const tnlVector< RealType2, tnlHost, IndexType2 >& long_vector );

   bool operator == ( const tnlVector< RealType, tnlHost, IndexType >& long_vector ) const;

   bool operator != ( const tnlVector< RealType, tnlHost, IndexType >& long_vector ) const;

   void setValue( const RealType& v );

   RealType max() const;

   RealType min() const;

   RealType absMax() const;

   RealType absMin() const;

   RealType lpNorm( const RealType& p ) const;

   RealType sum() const;

   RealType differenceMax( const tnlVector< RealType, tnlHost, IndexType >& v ) const;

   RealType differenceMin( const tnlVector< RealType, tnlHost, IndexType >& v ) const;

   RealType differenceAbsMax( const tnlVector< RealType, tnlHost, IndexType >& v ) const;

   RealType differenceAbsMin( const tnlVector< RealType, tnlHost, IndexType >& v ) const;

   RealType differenceLpNorm( const tnlVector< RealType, tnlHost, IndexType >& v, const RealType& p ) const;

   RealType differenceSum( const tnlVector< RealType, tnlHost, IndexType >& v ) const;

   void scalarMultiplication( const RealType& alpha );

   //! Compute scalar dot product
   RealType sdot( const tnlVector< RealType, tnlHost, IndexType >& v );

   //! Compute SAXPY operation (Scalar Alpha X Pus Y ).
   void saxpy( const RealType& alpha,
                const tnlVector< RealType, tnlHost, IndexType >& x );

   //! Compute SAXMY operation (Scalar Alpha X Minus Y ).
   /*!**
    * It is not a standart BLAS function but is useful for GMRES solver.
    */
   void saxmy( const RealType& alpha,
                const tnlVector< RealType, tnlHost, IndexType >& x );

   virtual ~tnlVector();
};

template< typename RealType, typename IndexType >
ostream& operator << ( ostream& str, const tnlVector< RealType, tnlHost, IndexType >& vec );

template< typename RealType, typename IndexType >
tnlVector< RealType, tnlHost, IndexType > :: tnlVector( const tnlString& name, IndexType _size )
: tnlArrayManager< RealType, tnlHost, IndexType >( name )
{
   this -> setSize( _size );
};

template< typename RealType, typename IndexType >
tnlVector< RealType, tnlHost, IndexType > :: tnlVector( const tnlString& name, const tnlVector& v )
: tnlArrayManager< RealType, tnlHost, IndexType >( name )
{
  this -> setSize( v. getSize() );
};

template< typename RealType, typename IndexType >
tnlVector< RealType, tnlHost, IndexType > :: tnlVector( const tnlString& name, const tnlVector< RealType, tnlCuda, IndexType >& v )
: tnlArrayManager< RealType, tnlHost, IndexType >( name )
{
  this -> setSize( v. getSize() );
};

template< typename RealType, typename IndexType >
tnlVector< RealType, tnlHost, IndexType >& tnlVector< RealType, tnlHost, IndexType > :: operator =( const tnlVector< RealType, tnlHost, IndexType >& vector )
{
   tnlArrayManager< RealType, tnlHost, IndexType > :: operator = ( vector );
   return *this;
};

template< typename RealType, typename IndexType >
   template< typename RealType2, typename IndexType2 >
tnlVector< RealType, tnlHost, IndexType >& tnlVector< RealType, tnlHost, IndexType > :: operator = ( const tnlVector< RealType2, tnlHost, IndexType2 >& vector )
{
   tnlArrayManager< RealType, tnlHost, IndexType > :: operator = ( vector );
   return *this;
};

template< typename RealType, typename IndexType >
tnlVector< RealType, tnlHost, IndexType >& tnlVector< RealType, tnlHost, IndexType > :: operator =  ( const tnlVector< RealType, tnlCuda, IndexType >& cuda_vector )
{
   tnlArrayManager< RealType, tnlHost, IndexType > :: operator = ( cuda_vector );
   return *this;
};

template< typename RealType, typename IndexType >
bool tnlVector< RealType, tnlHost, IndexType > :: operator == ( const tnlVector< RealType, tnlHost, IndexType >& long_vector ) const
{
   tnlAssert( this -> getSize() > 0, );
   tnlAssert( this -> getSize() == long_vector. getSize(),
              cerr << "You try to compare two long vectors with different sizes." << endl
                   << "The first one is " << this -> getName() << " with size " << this -> getSize()
                   << " while the second one is " << long_vector. getName() << " with size " << long_vector. getSize() << "." );
   if( memcmp( this -> getData(), long_vector. getData(), this -> getSize() * sizeof( RealType ) ) == 0 )
      return true;
   return false;
};

template< typename RealType, typename IndexType >
bool tnlVector< RealType, tnlHost, IndexType > :: operator != ( const tnlVector< RealType, tnlHost, IndexType >& long_vector ) const
{
   return ! ( ( *this ) == long_vector );
};

template< typename RealType, typename IndexType >
void tnlVector< RealType, tnlHost, IndexType > :: setValue( const RealType& v )
{
   tnlAssert( this -> size != 0,
              cerr << "Vector name is " << this -> getName() );
   const IndexType n = this -> getSize();
   RealType* data1 = this -> getData();
   for( IndexType i = 0; i < n; i ++ )
      data1[ i ] = v;
};

template< typename RealType, typename IndexType >
tnlVector< RealType, tnlHost, IndexType > :: ~tnlVector()
{
};

template< typename RealType, typename IndexType >
RealType tnlVector< RealType, tnlHost, IndexType > :: max() const
{
   tnlAssert( this -> getSize() != 0,
              cerr << "Vector name is " << this -> getName() );
   RealType result = this -> getData()[ 0 ];
   const IndexType n = this -> getSize();
   const RealType* data1 = this -> getData();
   for( IndexType i = 1; i < n; i ++ )
      result = Max( result, data1[ i ] );
   return result;
};

template< typename RealType, typename IndexType >
RealType tnlVector< RealType, tnlHost, IndexType > :: min() const
{
   tnlAssert( this -> getSize() != 0,
              cerr << "Vector name is " << this -> getName() );
   RealType result = this -> getData()[ 0 ];
   const IndexType n = this -> getSize();
   const RealType* data1 = this -> getData();
   for( IndexType i = 1; i < n; i ++ )
      result = Min( result, data1[ i ] );
   return result;
};

template< typename RealType, typename IndexType >
RealType tnlVector< RealType, tnlHost, IndexType > :: absMax() const
{
   tnlAssert( this -> getSize() != 0,
              cerr << "Vector name is " << this -> getName() );
   RealType result = this -> getData()[ 0 ];
   const IndexType n = this -> getSize();
   const RealType* data1 = this -> getData();
   for( IndexType i = 1; i < n; i ++ )
      result = Max( result, ( RealType ) fabs( data1[ i ] ) );
   return result;
};

template< typename RealType, typename IndexType >
RealType tnlVector< RealType, tnlHost, IndexType > :: absMin() const
{
   tnlAssert( this -> getSize() != 0,
              cerr << "Vector name is " << this -> getName() );
   RealType result = this -> getData()[ 0 ];
   const IndexType n = this -> getSize();
   const RealType* data1 = this -> getData();
   for( IndexType i = 1; i < n; i ++ )
      result = Min( result, ( RealType ) fabs( data1[ i ] ) );
   return result;
};

template< typename RealType, typename IndexType >
RealType tnlVector< RealType, tnlHost, IndexType > :: lpNorm( const RealType& p ) const
{
   tnlAssert( this -> getSize() != 0,
              cerr << "Vector name is " << this -> getName() );
   const IndexType n = this -> getSize();
   const RealType* data1 = this -> getData();
   RealType result = pow( ( RealType ) fabs( data1[ 0 ] ), ( RealType ) p );
   for( IndexType i = 1; i < n; i ++ )
      result += pow( ( RealType ) fabs( data1[ i ] ), ( RealType ) p  );
   return pow( result, 1.0 / p );
};

template< typename RealType, typename IndexType >
RealType tnlVector< RealType, tnlHost, IndexType > :: sum() const
{
   tnlAssert( this -> getSize() != 0,
              cerr << "Vector name is " << this -> getName() );
   RealType result = this -> getData()[ 0 ];
   const IndexType n = this -> getSize();
   const RealType* data1 = this -> getData();
   for( IndexType i = 1; i < n; i ++ )
      result += data1[ i ];
   return result;
};

template< typename RealType, typename IndexType >
RealType tnlVector< RealType, tnlHost, IndexType > :: differenceMax( const tnlVector< RealType, tnlHost, IndexType >& v ) const
{
   tnlAssert( this -> getSize() != 0,
              cerr << "Vector name is " << this -> getName() );
   tnlAssert( this -> getSize() == v. getSize(),
              cerr << "Vector names are " << this -> getName() << " and " << v. getName() );

   RealType result = ( *this )[ 0 ] - v[ 0 ];
   const IndexType n = v. getSize();
   for( IndexType i = 1; i < n; i ++ )
      result = Max( result, ( *this )[ i ] - v[ i ] );
   return result;
}

template< typename RealType, typename IndexType >
RealType tnlVector< RealType, tnlHost, IndexType > :: differenceMin( const tnlVector< RealType, tnlHost, IndexType >& v ) const
{
   tnlAssert( this -> getSize() != 0,
              cerr << "Vector name is " << this -> getName() );
   tnlAssert( this -> getSize() == v. getSize(),
              cerr << "Vector names are " << this -> getName() << " and " << v. getName() );

   RealType result = ( *this )[ 0 ] - v[ 0 ];
   const IndexType n = v. getSize();
   for( IndexType i = 1; i < n; i ++ )
      result = Min( result, ( *this )[ i ] - v[ i ] );
   return result;
}

template< typename RealType, typename IndexType >
RealType tnlVector< RealType, tnlHost, IndexType > :: differenceAbsMax( const tnlVector< RealType, tnlHost, IndexType >& v ) const
{
   tnlAssert( this -> getSize() != 0,
              cerr << "Vector name is " << this -> getName() );
   tnlAssert( this -> getSize() == v. getSize(),
              cerr << "Vector names are " << this -> getName() << " and " << v. getName() );

   RealType result = ( *this )[ 0 ] - v[ 0 ];
   const IndexType n = v. getSize();
   for( IndexType i = 1; i < n; i ++ )
      result = Max( result, ( RealType ) fabs( ( *this )[ i ] - v[ i ] ) );
   return result;

}

template< typename RealType, typename IndexType >
RealType tnlVector< RealType, tnlHost, IndexType > :: differenceAbsMin( const tnlVector< RealType, tnlHost, IndexType >& v ) const
{
   tnlAssert( this -> getSize() != 0,
              cerr << "Vector name is " << this -> getName() );
   tnlAssert( this -> getSize() == v. getSize(),
              cerr << "Vector names are " << this -> getName() << " and " << v. getName() );

   RealType result = ( *this )[ 0 ] - v[ 0 ];
   const IndexType n = v. getSize();
   for( IndexType i = 1; i < n; i ++ )
      result = Min( result, ( RealType ) fabs( ( *this )[ i ] - v[ i ] ) );
   return result;
}

template< typename RealType, typename IndexType >
RealType tnlVector< RealType, tnlHost, IndexType > :: differenceLpNorm( const tnlVector< RealType, tnlHost, IndexType >& v, const RealType& p ) const
{
   tnlAssert( this -> getSize() != 0,
              cerr << "Vector name is " << this -> getName() );
   tnlAssert( this -> getSize() == v. getSize(),
              cerr << "Vector names are " << this -> getName() << " and " << v. getName() );

   const IndexType n = v. getSize();
   RealType result = pow( ( RealType ) fabs( ( *this )[ 0 ] - v[ 0 ] ), ( RealType ) p );
   for( IndexType i = 1; i < n; i ++ )
      result += pow( ( RealType ) fabs( ( *this )[ i ] - v[ i ] ), ( RealType ) p  );
   return pow( result, 1.0 / p );
}

template< typename RealType, typename IndexType >
RealType tnlVector< RealType, tnlHost, IndexType > :: differenceSum( const tnlVector< RealType, tnlHost, IndexType >& v ) const
{
   tnlAssert( this -> getSize() != 0,
              cerr << "Vector name is " << this -> getName() );
   tnlAssert( this -> getSize() == v. getSize(),
              cerr << "Vector names are " << this -> getName() << " and " << v. getName() );

   RealType result = ( *this )[ 0 ] - v[ 0 ];
   const IndexType n = this -> getSize();
   for( IndexType i = 1; i < n; i ++ )
      result += ( *this )[ i ] - v[ i ];
   return result;
};

template< typename RealType, typename IndexType >
void tnlVector< RealType, tnlHost, IndexType > :: scalarMultiplication( const RealType& alpha )
{
   tnlAssert( this -> getSize() != 0,
              cerr << "Vector name is " << this -> getName() );
   const IndexType n = this -> getSize();
   for( IndexType i = 0; i < n; i ++ )
      ( *this )[ i ] *= alpha;
}

template< typename RealType, typename IndexType >
RealType tnlVector< RealType, tnlHost, IndexType > :: sdot( const tnlVector< RealType, tnlHost, IndexType >& v )
{
   tnlAssert( this -> getSize() != 0,
              cerr << "Vector name is " << this -> getName() );
   tnlAssert( this -> getSize() == v. getSize(),
            cerr << "You try to compute SDOT of two vectors with different size." << endl
                 << "The first tnLongVector " << this -> getName() << " has size " << this -> getSize() << "." << endl
                 << "The second tnlVector " << v. getName() << " has size " << v. getSize() << "."  );
   RealType result = ( RealType ) 0;
   const IndexType n = this -> getSize();
   const RealType* data1 = this -> getData();
   const RealType* data2 = v. getData();
   for( IndexType i = 0; i < n; i ++ )
      result += data1[ i ] * data2[ i ];
   return result;
};

template< typename RealType, typename IndexType >
void tnlVector< RealType, tnlHost, IndexType > :: saxpy( const RealType& alpha,
                                                         const tnlVector< RealType, tnlHost, IndexType >& x )
{
   tnlAssert( this -> getSize() != 0,
              cerr << "Vector name is " << this -> getName() );
   tnlAssert( this -> getSize() == x. getSize(),
            cerr << "You try to compute SAXPY with vector x having different size." << endl
                 << "The y tnLongVector " << this -> getName() << " has size " << this -> getSize() << "." << endl
                 << "The x tnlVector " << x. getName() << " has size " << x. getSize() << "."  );
   const IndexType n = this -> getSize();
   RealType* data1 = this -> getData();
   const RealType* data2 = x. getData();
   for( IndexType i = 0; i < n; i ++ )
      data1[ i ] += alpha * data2[ i ];
};

template< typename RealType, typename IndexType >
void tnlVector< RealType, tnlHost, IndexType > :: saxmy( const RealType& alpha,
                                                             const tnlVector< RealType, tnlHost, IndexType >& x )
{
   tnlAssert( this -> getSize() != 0,
              cerr << "Vector name is " << this -> getName() );
   tnlAssert( this -> getSize() == x. getSize(),
            cerr << "You try to compute SAXPY with vector x having different size." << endl
                 << "The y tnLongVector " << this -> getName() << " has size " << this -> getSize() << "." << endl
                 << "The x tnlVector " << x. getName() << " has size " << x. getSize() << "."  );
   const IndexType n = this -> getSize();
   RealType* data1 = this -> getData();
   const RealType* data2 = x. getData();
   for( IndexType i = 0; i < n; i ++ )
      data1[ i ] = alpha * data2[ i ] - data1[ i ];
};



#endif /* TNLVECTORHOST_H_ */
