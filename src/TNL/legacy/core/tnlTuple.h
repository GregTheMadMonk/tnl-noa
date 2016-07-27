/***************************************************************************
                          StaticVector.h  -  description
                             -------------------
    begin                : 2006/03/04
    copyright            : (C) 2006 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef StaticVectorH
#define StaticVectorH

#include <TNL/Assert.h>
#include <string.h>
#include <TNL/File.h>
#include <TNL/String.h>
#include "param-types.h"

//! Aliases for the coordinates
enum { tnlX = 0, tnlY, tnlZ };

template< int Size, typename Real = double >
class StaticVector
{
   public:
   typedef Real RealType;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   StaticVector();

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   StaticVector( const Real v[ Size ] );

   //! This sets all vector components to v
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   StaticVector( const Real& v );

   //! Copy constructor
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   StaticVector( const StaticVector< Size, Real >& v );

   //! This is constructore of vector with Size = 2.
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   StaticVector( const Real& v1,
              const Real& v2 );

   //! This is constructore of vector with Size = 3
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   StaticVector( const Real& v1,
             const Real& v2,
             const Real& v3 );

   static String getType();

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
   StaticVector& operator += ( const StaticVector& v );

   //! Subtracting operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   StaticVector& operator -= ( const StaticVector& v );

   //! Multiplication with number
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   StaticVector& operator *= ( const Real& c );

   //! Adding operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   StaticVector operator + ( const StaticVector& u ) const;

   //! Subtracting operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   StaticVector operator - ( const StaticVector& u ) const;

   //! Multiplication with number
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   StaticVector operator * ( const Real& c ) const;

   //!
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   StaticVector& operator = ( const StaticVector& v );

   //! Scalar product
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   Real operator * ( const StaticVector& u ) const;

   //! Comparison operator
   template< typename Real2 >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator == ( const StaticVector< Size, Real2 >& v ) const;

   //! Comparison operator
   template< typename Real2 >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator != ( const StaticVector< Size, Real2 >& v ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator < ( const StaticVector& v ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator <= ( const StaticVector& v ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator > ( const StaticVector& v ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator >= ( const StaticVector& v ) const;

   bool save( File& file ) const;

   bool load( File& file);

   protected:
   Real data[ Size ];

};

template< int Size, typename Real >
StaticVector< Size, Real > operator * ( const Real& c, const StaticVector< Size, Real >& u );

template< int Size, typename Real >
ostream& operator << ( std::ostream& str, const StaticVector< Size, Real >& v );

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
StaticVector< Size, Real > :: StaticVector()
{
   bzero( data, Size * sizeof( Real ) );
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
StaticVector< Size, Real > :: StaticVector( const Real v[ Size ] )
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
StaticVector< Size, Real > :: StaticVector( const Real& v )
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
StaticVector< Size, Real > :: StaticVector( const StaticVector< Size, Real >& v )
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
StaticVector< Size, Real > :: StaticVector( const Real& v1,
                                    const Real& v2 )
{
   Assert( Size == 2,
              printf( "Using this constructor does not makes sense for Size different then 2.\n") );
   data[ 0 ] = v1;
   data[ 1 ] = v2;
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
StaticVector< Size, Real > :: StaticVector( const Real& v1,
                                      const Real& v2,
                                      const Real& v3 )
{
   Assert( Size == 3,
              printf( "Using this constructor does not makes sense for Size different then 3.\n") );
   data[ 0 ] = v1;
   data[ 1 ] = v2;
   data[ 2 ] = v3;
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
String StaticVector< Size, Real > :: getType()
{
   return String( "StaticVector< " ) +
          String( Size ) +
          String( ", " ) +
         TNL::getType< Real >() +
          String( " >" );
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
const Real& StaticVector< Size, Real > :: operator[]( int i ) const
{
   assert( i >= 0 && i < Size );
   return data[ i ];
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
Real& StaticVector< Size, Real > :: operator[]( int i )
{
   assert( i < Size );
   return data[ i ];
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
Real& StaticVector< Size, Real > :: x()
{
   Assert( Size > 0, std::cerr << "Size = " << Size << std::endl; );
   if( Size < 1 )
   {
      printf( "The size of the StaticVector is too small to get x coordinate.\n" );
      abort();
   }
   return data[ 0 ];
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
const Real& StaticVector< Size, Real > :: x() const
{
   Assert( Size > 0, std::cerr << "Size = " << Size << std::endl; );
   if( Size < 1 )
   {
      printf( "The size of the StaticVector is too small to get x coordinate.\n" );
      abort();
   }
   return data[ 0 ];
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
Real& StaticVector< Size, Real > :: y()
{
   Assert( Size > 1, std::cerr << "Size = " << Size << std::endl; );
   if( Size < 2 )
   {
      printf( "The size of the StaticVector is too small to get y coordinate.\n" );
      abort();
   }
   return data[ 1 ];
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
const Real& StaticVector< Size, Real > :: y() const
{
   Assert( Size > 1, std::cerr << "Size = " << Size << std::endl; );
   if( Size < 2 )
   {
      printf( "The size of the StaticVector is too small to get y coordinate.\n" );
      abort();
   }
   return data[ 1 ];

};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
Real& StaticVector< Size, Real > :: z()
{
   Assert( Size > 2, std::cerr << "Size = " << Size << std::endl; );
   if( Size < 3 )
   {
      printf( "The size of the StaticVector is too small to get z coordinate.\n" );
      abort();
   }
   return data[ 2 ];
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
const Real& StaticVector< Size, Real > :: z() const
{
   Assert( Size > 2, std::cerr << "Size = " << Size << std::endl; );
   if( Size < 3 )
   {
      printf( "The size of the StaticVector is too small to get z coordinate.\n" );
      abort();
   }
   return data[ 2 ];
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
StaticVector< Size, Real >& StaticVector< Size, Real > :: operator += ( const StaticVector& v )
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
StaticVector< Size, Real >& StaticVector< Size, Real > :: operator -= ( const StaticVector& v )
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
StaticVector< Size, Real >& StaticVector< Size, Real > :: operator *= ( const Real& c )
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
StaticVector< Size, Real > StaticVector< Size, Real > :: operator + ( const StaticVector& u ) const
{
   // TODO: Leads to sigsegv
   return StaticVector( * this ) += u;
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
StaticVector< Size, Real > StaticVector< Size, Real > :: operator - ( const StaticVector& u ) const
{
   // TODO: Leads to sigsegv
   return StaticVector( * this ) -= u;
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
StaticVector< Size, Real > StaticVector< Size, Real > :: operator * ( const Real& c ) const
{
   return StaticVector( * this ) *= c;
};

template< int Size, typename Real >
#ifdef HAVE_CUDA
   __host__ __device__
#endif
StaticVector< Size, Real >& StaticVector< Size, Real > :: operator = ( const StaticVector& v )
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
Real StaticVector< Size, Real > :: operator * ( const StaticVector& u ) const
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
bool StaticVector< Size, Real > :: operator == ( const StaticVector< Size, Real2 >& u ) const
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
bool StaticVector< Size, Real > :: operator != ( const StaticVector< Size, Real2 >& u ) const
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
bool StaticVector< Size, Real > :: operator < ( const StaticVector& u ) const
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
bool StaticVector< Size, Real > :: operator <= ( const StaticVector& u ) const
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
bool StaticVector< Size, Real > :: operator > ( const StaticVector& u ) const
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
bool StaticVector< Size, Real > :: operator >= ( const StaticVector& u ) const
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
bool StaticVector< Size, Real > :: save( File& file ) const
{
   int size = Size;
#ifdef HAVE_NOT_CXX11
   if( ! file. write< int, Devices::Host >( &size ) ||
       ! file. write< Real, Devices::Host, int >( data, size ) )
      std::cerr << "Unable to write StaticVector." << std::endl;
#else
   if( ! file. write( &size ) ||
       ! file. write( data, size ) )
      std::cerr << "Unable to write StaticVector." << std::endl;
#endif
   return true;
};

template< int Size, typename Real >
bool StaticVector< Size, Real > :: load( File& file)
{
   int size;
#ifdef HAVE_NOT_CXX11
   if( ! file. read< int, Devices::Host >( &size ) )
#else
   if( ! file. read( &size ) )
#endif
   {
      std::cerr << "Unable to read StaticVector." << std::endl;
      return false;
   }
   if( size != Size )
   {
      std::cerr << "You try to read StaticVector with wrong size " << size
           << ". It should be " << Size << std::endl;
      return false;
   }
#ifdef HAVE_NOT_CXX11
   if( ! file. read< Real, Devices::Host, int >( data, size ) )
#else
   if( ! file. read( data, size ) )
#endif
   {
      std::cerr << "Unable to read StaticVector." << std::endl;
      return false;
   }
   return true;
}

template< int Size, typename Real >
StaticVector< Size, Real > operator * ( const Real& c, const StaticVector< Size, Real >& u )
{
   return u * c;
}



// TODO: remove
template< int Size, typename Real > bool Save( std::ostream& file, const StaticVector< Size, Real >& vec )
{
   for( int i = 0; i < Size; i ++ )
      file. write( ( char* ) &vec[ i ], sizeof( Real ) );
   if( file. bad() ) return false;
   return true;
};

// TODO: remove
template< int Size, typename Real > bool Load( std::istream& file, StaticVector< Size, Real >& vec )
{
   for( int i = 0; i < Size; i ++ )
      file. read( ( char* ) &vec[ i ], sizeof( Real ) );
   if( file. bad() ) return false;
   return true;
};

template< int Size, typename Real >
ostream& operator << ( std::ostream& str, const StaticVector< Size, Real >& v )
{
   for( int i = 0; i < Size - 1; i ++ )
      str << v[ i ] << ", ";
   str << v[ Size - 1 ];
   return str;
};

template< typename Real >
StaticVector< 3, Real > VectorProduct( const StaticVector< 3, Real >& u,
                                      const StaticVector< 3, Real >& v )
{
   StaticVector< 3, Real > p;
   p[ 0 ] = u[ 1 ] * v[ 2 ] - u[ 2 ] * v[ 1 ];
   p[ 1 ] = u[ 2 ] * v[ 0 ] - u[ 0 ] * v[ 2 ];
   p[ 2 ] = u[ 0 ] * v[ 1 ] - u[ 1 ] * v[ 0 ];
   return p;
};

template< typename Real >
Real tnlScalarProduct( const StaticVector< 2, Real >& u,
                       const StaticVector< 2, Real >& v )
{
   return u[ 0 ] * v[ 0 ] + u[ 1 ] * v[ 1 ];
};

template< typename Real >
Real tnlScalarProduct( const StaticVector< 3, Real >& u,
                       const StaticVector< 3, Real >& v )
{
   return u[ 0 ] * v[ 0 ] + u[ 1 ] * v[ 1 ] + u[ 2 ] * v[ 2 ];
};

template< typename Real >
Real tnlTriangleArea( const StaticVector< 2, Real >& a,
                      const StaticVector< 2, Real >& b,
                      const StaticVector< 2, Real >& c )
{
   StaticVector< 3, Real > u1, u2;
   u1. x() = b. x() - a. x();
   u1. y() = b. y() - a. y();
   u1. z() = 0.0;
   u2. x() = c. x() - a. x();
   u2. y() = c. y() - a. y();
   u2. z() = 0;

   const StaticVector< 3, Real > v = VectorProduct( u1, u2 );
   return 0.5 * ::sqrt( tnlScalarProduct( v, v ) );
};

#endif
