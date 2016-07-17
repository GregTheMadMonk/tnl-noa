/***************************************************************************
                          tnlStaticVector.h  -  description
                             -------------------
    begin                : 2006/03/04
    copyright            : (C) 2006 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef tnlStaticVectorH
#define tnlStaticVectorH

#include <core/tnlAssert.h>
#include <string.h>
#include <core/tnlFile.h>
#include <core/tnlString.h>
#include "param-types.h"

//! Aliases for the coordinates
enum { tnlX = 0, tnlY, tnlZ };

template< int Size, typename Real = double >
class tnlStaticVector
{
   public:
   typedef Real RealType;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector();

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector( const Real v[ Size ] );

   //! This sets all vector components to v
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector( const Real& v );

   //! Copy constructor
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector( const tnlStaticVector< Size, Real >& v );

   //! This is constructore of vector with Size = 2.
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector( const Real& v1,
              const Real& v2 );

   //! This is constructore of vector with Size = 3
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector( const Real& v1,
             const Real& v2,
             const Real& v3 );

   static tnlString getType();

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   const Real& operator[]( int i ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   Real& operator[]( int i );
 
   //! Returns the first coordinate
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   Real& x();

   //! Returns the first coordinate
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   const Real& x() const;

   //! Returns the second coordinate
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   Real& y();

   //! Returns the second coordinate
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   const Real& y() const;

   //! Returns the third coordinate
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   Real& z();

   //! Returns the third coordinate
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   const Real& z() const;

   //! Adding operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector& operator += ( const tnlStaticVector& v );

   //! Subtracting operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector& operator -= ( const tnlStaticVector& v );

   //! Multiplication with number
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector& operator *= ( const Real& c );

   //! Adding operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector operator + ( const tnlStaticVector& u ) const;

   //! Subtracting operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector operator - ( const tnlStaticVector& u ) const;

   //! Multiplication with number
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector operator * ( const Real& c ) const;

   //!
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector& operator = ( const tnlStaticVector& v );

   //! Scalar product
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   Real operator * ( const tnlStaticVector& u ) const;

   //! Comparison operator
   template< typename Real2 >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator == ( const tnlStaticVector< Size, Real2 >& v ) const;

   //! Comparison operator
   template< typename Real2 >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator != ( const tnlStaticVector< Size, Real2 >& v ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator < ( const tnlStaticVector& v ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator <= ( const tnlStaticVector& v ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator > ( const tnlStaticVector& v ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator >= ( const tnlStaticVector& v ) const;

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file);

   protected:
   Real data[ Size ];

};

template< int Size, typename Real >
tnlStaticVector< Size, Real > operator * ( const Real& c, const tnlStaticVector< Size, Real >& u );

template< int Size, typename Real >
ostream& operator << ( ostream& str, const tnlStaticVector< Size, Real >& v );

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
tnlStaticVector< Size, Real > :: tnlStaticVector()
{
   bzero( data, Size * sizeof( Real ) );
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
tnlStaticVector< Size, Real > :: tnlStaticVector( const Real v[ Size ] )
{
   if( Size == 1 )
      data[ 0 ] = v[ 0 ];
   if( Size == 2 )
   {
      data[ 0 ] = v[ 0 ];
      data[ 1 ] = v[ 1 ];
   }
   if( Size == 3 )
   {
      data[ 0 ] = v[ 0 ];
      data[ 1 ] = v[ 1 ];
      data[ 2 ] = v[ 2 ];
   }
   if( Size > 3 )
   {
      for( int i = 0; i < Size; i ++ )
         data[ i ] = v[ i ];
   }
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
tnlStaticVector< Size, Real > :: tnlStaticVector( const Real& v )
{
   if( Size == 1 )
      data[ 0 ] = v;
   if( Size == 2 )
   {
      data[ 0 ] = data[ 1 ] = v;
   }
   if( Size == 3 )
   {
      data[ 0 ] = data[ 1 ] = data[ 2 ] = v;
   }
   if( Size > 3 )
   {
      for( int i = 0; i < Size; i ++ )
         data[ i ] = v;
   }
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
tnlStaticVector< Size, Real > :: tnlStaticVector( const tnlStaticVector< Size, Real >& v )
{
   if( Size == 1 )
      data[ 0 ] = v[ 0 ];
   if( Size == 2 )
   {
      data[ 0 ] = v[ 0 ];
      data[ 1 ] = v[ 1 ];
   }
   if( Size == 3 )
   {
      data[ 0 ] = v[ 0 ];
      data[ 1 ] = v[ 1 ];
      data[ 2 ] = v[ 2 ];
   }
   if( Size > 3 )
   {
      for( int i = 0; i < Size; i ++ )
         data[ i ] = v[ i ];
   }
};


template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
tnlStaticVector< Size, Real > :: tnlStaticVector( const Real& v1,
                                    const Real& v2 )
{
   tnlAssert( Size == 2,
              printf( "Using this constructor does not makes sense for Size different then 2.\n") );
   data[ 0 ] = v1;
   data[ 1 ] = v2;
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
tnlStaticVector< Size, Real > :: tnlStaticVector( const Real& v1,
                                      const Real& v2,
                                      const Real& v3 )
{
   tnlAssert( Size == 3,
              printf( "Using this constructor does not makes sense for Size different then 3.\n") );
   data[ 0 ] = v1;
   data[ 1 ] = v2;
   data[ 2 ] = v3;
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
tnlString tnlStaticVector< Size, Real > :: getType()
{
   return tnlString( "tnlStaticVector< " ) +
          tnlString( Size ) +
          tnlString( ", " ) +
          ::getType< Real >() +
          tnlString( " >" );
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
const Real& tnlStaticVector< Size, Real > :: operator[]( int i ) const
{
   assert( i >= 0 && i < Size );
   return data[ i ];
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
Real& tnlStaticVector< Size, Real > :: operator[]( int i )
{
   assert( i < Size );
   return data[ i ];
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
Real& tnlStaticVector< Size, Real > :: x()
{
   tnlAssert( Size > 0, cerr << "Size = " << Size << endl; );
   if( Size < 1 )
   {
      printf( "The size of the tnlStaticVector is too small to get x coordinate.\n" );
      abort();
   }
   return data[ 0 ];
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
const Real& tnlStaticVector< Size, Real > :: x() const
{
   tnlAssert( Size > 0, cerr << "Size = " << Size << endl; );
   if( Size < 1 )
   {
      printf( "The size of the tnlStaticVector is too small to get x coordinate.\n" );
      abort();
   }
   return data[ 0 ];
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
Real& tnlStaticVector< Size, Real > :: y()
{
   tnlAssert( Size > 1, cerr << "Size = " << Size << endl; );
   if( Size < 2 )
   {
      printf( "The size of the tnlStaticVector is too small to get y coordinate.\n" );
      abort();
   }
   return data[ 1 ];
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
const Real& tnlStaticVector< Size, Real > :: y() const
{
   tnlAssert( Size > 1, cerr << "Size = " << Size << endl; );
   if( Size < 2 )
   {
      printf( "The size of the tnlStaticVector is too small to get y coordinate.\n" );
      abort();
   }
   return data[ 1 ];

};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
Real& tnlStaticVector< Size, Real > :: z()
{
   tnlAssert( Size > 2, cerr << "Size = " << Size << endl; );
   if( Size < 3 )
   {
      printf( "The size of the tnlStaticVector is too small to get z coordinate.\n" );
      abort();
   }
   return data[ 2 ];
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
const Real& tnlStaticVector< Size, Real > :: z() const
{
   tnlAssert( Size > 2, cerr << "Size = " << Size << endl; );
   if( Size < 3 )
   {
      printf( "The size of the tnlStaticVector is too small to get z coordinate.\n" );
      abort();
   }
   return data[ 2 ];
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
tnlStaticVector< Size, Real >& tnlStaticVector< Size, Real > :: operator += ( const tnlStaticVector& v )
{
   if( Size == 1 )
      data[ 0 ] += v. data[ 0 ];
   if( Size == 2 )
   {
      data[ 0 ] += v. data[ 0 ];
      data[ 1 ] += v. data[ 1 ];
   }
   if( Size == 3 )
   {
      data[ 0 ] += v. data[ 0 ];
      data[ 1 ] += v. data[ 1 ];
      data[ 2 ] += v. data[ 2 ];
   }
   if( Size > 3 )
   {
      for( int i = 0; i < Size; i ++ )
         data[ i ] += v. data[ i ];
   }
   return ( *this );
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
tnlStaticVector< Size, Real >& tnlStaticVector< Size, Real > :: operator -= ( const tnlStaticVector& v )
{
   if( Size == 1 )
      data[ 0 ] -= v. data[ 0 ];
   if( Size == 2 )
   {
      data[ 0 ] -= v. data[ 0 ];
      data[ 1 ] -= v. data[ 1 ];
   }
   if( Size == 3 )
   {
      data[ 0 ] -= v. data[ 0 ];
      data[ 1 ] -= v. data[ 1 ];
      data[ 2 ] -= v. data[ 2 ];
   }
   if( Size > 3 )
   {
      for( int i = 0; i < Size; i ++ )
         data[ i ] -= v. data[ i ];
   }
   return ( *this );
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
tnlStaticVector< Size, Real >& tnlStaticVector< Size, Real > :: operator *= ( const Real& c )
{
   if( Size == 1 )
      data[ 0 ] *= c;
   if( Size == 2 )
   {
      data[ 0 ] *= c;
      data[ 1 ] *= c;
   }
   if( Size == 3 )
   {
      data[ 0 ] *= c;
      data[ 1 ] *= c;
      data[ 2 ] *= c;
   }
   if( Size > 3 )
   {
      for( int i = 0; i < Size; i ++ )
         data[ i ] *= c;
   }
   return ( *this );
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
tnlStaticVector< Size, Real > tnlStaticVector< Size, Real > :: operator + ( const tnlStaticVector& u ) const
{
   // TODO: Leads to sigsegv
   return tnlStaticVector( * this ) += u;
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
tnlStaticVector< Size, Real > tnlStaticVector< Size, Real > :: operator - ( const tnlStaticVector& u ) const
{
   // TODO: Leads to sigsegv
   return tnlStaticVector( * this ) -= u;
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
tnlStaticVector< Size, Real > tnlStaticVector< Size, Real > :: operator * ( const Real& c ) const
{
   return tnlStaticVector( * this ) *= c;
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
tnlStaticVector< Size, Real >& tnlStaticVector< Size, Real > :: operator = ( const tnlStaticVector& v )
{
   memcpy( &data[ 0 ], &v. data[ 0 ], Size * sizeof( Real ) );
   /*int i;
   for( i = 0; i < Size; i ++ )
      data[ i ] = v. data[ i ];*/
   return *this;
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
Real tnlStaticVector< Size, Real > :: operator * ( const tnlStaticVector& u ) const
{
   if( Size == 1 )
      return data[ 0 ] * u[ 0 ];
   if( Size == 2 )
      return data[ 0 ] * u[ 0 ] +
             data[ 1 ] * u[ 1 ];
   if( Size == 3 )
      return data[ 0 ] * u[ 0 ] +
             data[ 1 ] * u[ 1 ] +
             data[ 2 ] * u[ 2 ];
   if( Size > 3 )
   {
      Real res( 0.0 );
      for( int i = 0; i < Size; i ++ )
         res += data[ i ] * u. data[ i ];
      return res;
   }
};

template< int Size, typename Real >
template< typename Real2 >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
bool tnlStaticVector< Size, Real > :: operator == ( const tnlStaticVector< Size, Real2 >& u ) const
{
   if( Size == 1 )
      return data[ 0 ] == u[ 0 ];
   if( Size == 2 )
      return data[ 0 ] == u[ 0 ] &&
             data[ 1 ] == u[ 1 ];
   if( Size == 3 )
      return data[ 0 ] == u[ 0 ] &&
             data[ 1 ] == u[ 1 ] &&
             data[ 2 ] == u[ 2 ];
   for( int i = 0; i < Size; i ++ )
      if( data[ i ] != u[ i ] )
         return false;
   return true;

};

template< int Size, typename Real >
template< typename Real2 >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
bool tnlStaticVector< Size, Real > :: operator != ( const tnlStaticVector< Size, Real2 >& u ) const
{
   if( Size == 1 )
      return data[ 0 ] != u[ 0 ];
   if( Size == 2 )
      return data[ 0 ] != u[ 0 ] ||
             data[ 1 ] != u[ 1 ];
   if( Size == 3 )
      return data[ 0 ] != u[ 0 ] ||
             data[ 1 ] != u[ 1 ] ||
             data[ 2 ] != u[ 2 ];
   for( int i = 0; i < Size; i ++ )
      if( data[ i ] != u[ i ] )
         return true;
   return false;
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
bool tnlStaticVector< Size, Real > :: operator < ( const tnlStaticVector& u ) const
{
   if( Size == 1 )
      return data[ 0 ] < u[ 0 ];
   if( Size == 2 )
      return data[ 0 ] < u[ 0 ] &&
             data[ 1 ] < u[ 1 ];
   if( Size == 3 )
      return data[ 0 ] < u[ 0 ] &&
             data[ 1 ] < u[ 1 ] &&
             data[ 2 ] < u[ 2 ];
   if( Size > 3 )
   {
      for( int i = 0; i <  Size; i ++ )
         if( data[ i ] >= u[ i ] )
            return false;
      return true;
   }
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
bool tnlStaticVector< Size, Real > :: operator <= ( const tnlStaticVector& u ) const
{
   if( Size == 1 )
      return data[ 0 ] <= u[ 0 ];
   if( Size == 2 )
      return data[ 0 ] <= u[ 0 ] &&
             data[ 1 ] <= u[ 1 ];
   if( Size == 3 )
      return data[ 0 ] <= u[ 0 ] &&
             data[ 1 ] <= u[ 1 ] &&
             data[ 2 ] <= u[ 2 ];
   for( int i = 0; i <  Size; i ++ )
      if( data[ i ] > u[ i ] )
         return false;
   return true;
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
bool tnlStaticVector< Size, Real > :: operator > ( const tnlStaticVector& u ) const
{
   if( Size == 1 )
      return data[ 0 ] > u[ 0 ];
   if( Size == 2 )
      return data[ 0 ] > u[ 0 ] &&
             data[ 1 ] > u[ 1 ];
   if( Size == 3 )
      return data[ 0 ] > u[ 0 ] &&
             data[ 1 ] > u[ 1 ] &&
             data[ 2 ] > u[ 2 ];
   for( int i = 0; i <  Size; i ++ )
      if( data[ i ] <= u[ i ] )
         return false;
   return true;
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
bool tnlStaticVector< Size, Real > :: operator >= ( const tnlStaticVector& u ) const
{
   if( Size == 1 )
      return data[ 0 ] >= u[ 0 ];
   if( Size == 2 )
      return data[ 0 ] >= u[ 0 ] &&
             data[ 1 ] >= u[ 1 ];
   if( Size == 3 )
      return data[ 0 ] >= u[ 0 ] &&
             data[ 1 ] >= u[ 1 ] &&
             data[ 2 ] >= u[ 2 ];
   for( int i = 0; i <  Size; i ++ )
      if( data[ i ] < u[ i ] )
         return false;
   return true;
}

template< int Size, typename Real >
bool tnlStaticVector< Size, Real > :: save( tnlFile& file ) const
{
   int size = Size;
#ifdef HAVE_NOT_CXX11
   if( ! file. write< int, tnlHost >( &size ) ||
       ! file. write< Real, tnlHost, int >( data, size ) )
      cerr << "Unable to write tnlStaticVector." << endl;
#else
   if( ! file. write( &size ) ||
       ! file. write( data, size ) )
      cerr << "Unable to write tnlStaticVector." << endl;
#endif
   return true;
};

template< int Size, typename Real >
bool tnlStaticVector< Size, Real > :: load( tnlFile& file)
{
   int size;
#ifdef HAVE_NOT_CXX11
   if( ! file. read< int, tnlHost >( &size ) )
#else
   if( ! file. read( &size ) )
#endif
   {
      cerr << "Unable to read tnlStaticVector." << endl;
      return false;
   }
   if( size != Size )
   {
      cerr << "You try to read tnlStaticVector with wrong size " << size
           << ". It should be " << Size << endl;
      return false;
   }
#ifdef HAVE_NOT_CXX11
   if( ! file. read< Real, tnlHost, int >( data, size ) )
#else
   if( ! file. read( data, size ) )
#endif
   {
      cerr << "Unable to read tnlStaticVector." << endl;
      return false;
   }
   return true;
}

template< int Size, typename Real >
tnlStaticVector< Size, Real > operator * ( const Real& c, const tnlStaticVector< Size, Real >& u )
{
   return u * c;
}



// TODO: remove
template< int Size, typename Real > bool Save( ostream& file, const tnlStaticVector< Size, Real >& vec )
{
   for( int i = 0; i < Size; i ++ )
      file. write( ( char* ) &vec[ i ], sizeof( Real ) );
   if( file. bad() ) return false;
   return true;
};

// TODO: remove
template< int Size, typename Real > bool Load( istream& file, tnlStaticVector< Size, Real >& vec )
{
   for( int i = 0; i < Size; i ++ )
      file. read( ( char* ) &vec[ i ], sizeof( Real ) );
   if( file. bad() ) return false;
   return true;
};

template< int Size, typename Real >
ostream& operator << ( ostream& str, const tnlStaticVector< Size, Real >& v )
{
   for( int i = 0; i < Size - 1; i ++ )
      str << v[ i ] << ", ";
   str << v[ Size - 1 ];
   return str;
};

template< typename Real >
tnlStaticVector< 3, Real > tnlVectorProduct( const tnlStaticVector< 3, Real >& u,
                                      const tnlStaticVector< 3, Real >& v )
{
   tnlStaticVector< 3, Real > p;
   p[ 0 ] = u[ 1 ] * v[ 2 ] - u[ 2 ] * v[ 1 ];
   p[ 1 ] = u[ 2 ] * v[ 0 ] - u[ 0 ] * v[ 2 ];
   p[ 2 ] = u[ 0 ] * v[ 1 ] - u[ 1 ] * v[ 0 ];
   return p;
};

template< typename Real >
Real tnlScalarProduct( const tnlStaticVector< 2, Real >& u,
                       const tnlStaticVector< 2, Real >& v )
{
   return u[ 0 ] * v[ 0 ] + u[ 1 ] * v[ 1 ];
};

template< typename Real >
Real tnlScalarProduct( const tnlStaticVector< 3, Real >& u,
                       const tnlStaticVector< 3, Real >& v )
{
   return u[ 0 ] * v[ 0 ] + u[ 1 ] * v[ 1 ] + u[ 2 ] * v[ 2 ];
};

template< typename Real >
Real tnlTriangleArea( const tnlStaticVector< 2, Real >& a,
                      const tnlStaticVector< 2, Real >& b,
                      const tnlStaticVector< 2, Real >& c )
{
   tnlStaticVector< 3, Real > u1, u2;
   u1. x() = b. x() - a. x();
   u1. y() = b. y() - a. y();
   u1. z() = 0.0;
   u2. x() = c. x() - a. x();
   u2. y() = c. y() - a. y();
   u2. z() = 0;

   const tnlStaticVector< 3, Real > v = tnlVectorProduct( u1, u2 );
   return 0.5 * sqrt( tnlScalarProduct( v, v ) );
};

#endif
