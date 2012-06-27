/***************************************************************************
                          tnlTuple.h  -  description
                             -------------------
    begin                : 2006/03/04
    copyright            : (C) 2006 by Tomas Oberhuber
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

#ifndef tnlTupleH
#define tnlTupleH

#include <core/tnlAssert.h>
#include <string.h>
#include <core/tnlFile.h>
#include "param-types.h"

//! Aliases for the coordinates
enum { tnlX = 0, tnlY, tnlZ };

template< int Size, typename Real = double >
class tnlTuple
{
   public:

   tnlTuple();

   tnlTuple( const Real v[ Size ] );

   //! This sets all vector components to v
   tnlTuple( const Real& v );

   //! Copy constructor
   tnlTuple( const tnlTuple< Size, Real >& v );

   //! This is constructore of vector with Size = 2.
   tnlTuple( const Real& v1,
              const Real& v2 );

   //! This is constructore of vector with Size = 3.
   tnlTuple( const Real& v1,
              const Real& v2,
              const Real& v3 );

   const Real& operator[]( int i ) const;

   Real& operator[]( int i );
   
   //! Returns the first coordinate
   Real& x();

   //! Returns the first coordinate
   const Real& x() const;

   //! Returns the second coordinate
   Real& y();

   //! Returns the second coordinate
   const Real& y() const;

   //! Returns the third coordinate
   Real& z();

   //! Returns the third coordinate
   const Real& z() const;

   //! Adding operator
   tnlTuple& operator += ( const tnlTuple& v );

   //! Subtracting operator
   tnlTuple& operator -= ( const tnlTuple& v );

   //! Multiplication with number
   tnlTuple& operator *= ( const Real& c );

   //! Adding operator
   tnlTuple operator + ( const tnlTuple& u ) const;

   //! Subtracting operator
   tnlTuple operator - ( const tnlTuple& u ) const;

   //! Multiplication with number
   tnlTuple operator * ( const Real& c ) const;

   //! 
   tnlTuple& operator = ( const tnlTuple& v );

   //! Scalar product
   Real operator * ( const tnlTuple& u ) const;

   //! Comparison operator
   template< typename Real2 >
   bool operator == ( const tnlTuple< Size, Real2 >& v ) const;

   //! Comparison operator
   template< typename Real2 >
   bool operator != ( const tnlTuple< Size, Real2 >& v ) const;

   bool operator < ( const tnlTuple& v ) const;

   bool operator <= ( const tnlTuple& v ) const;

   bool operator > ( const tnlTuple& v ) const;

   bool operator >= ( const tnlTuple& v ) const;

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file);

   protected:
   Real data[ Size ];

};

template< int Size, typename Real >
tnlTuple< Size, Real > operator * ( const Real& c, const tnlTuple< Size, Real >& u );

template< int Size, typename Real >
ostream& operator << ( ostream& str, const tnlTuple< Size, Real >& v );

template< int Size, typename Real >
tnlTuple< Size, Real > :: tnlTuple()
{
   bzero( data, Size * sizeof( Real ) );
};

template< int Size, typename Real >
tnlTuple< Size, Real > :: tnlTuple( const Real v[ Size ] )
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
tnlTuple< Size, Real > :: tnlTuple( const Real& v )
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
tnlTuple< Size, Real > :: tnlTuple( const tnlTuple< Size, Real >& v )
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
tnlTuple< Size, Real > :: tnlTuple( const Real& v1,
                                      const Real& v2 )
{
   tnlAssert( Size == 2,
              cerr << "Using this constructor does not makes sense for Size different then 2.")
   data[ 0 ] = v1;
   data[ 1 ] = v2;
}

template< int Size, typename Real >
tnlTuple< Size, Real > :: tnlTuple( const Real& v1,
                                      const Real& v2,
                                      const Real& v3 )
{
   tnlAssert( Size == 3,
              cerr << "Using this constructor does not makes sense for Size different then 3.")
   data[ 0 ] = v1;
   data[ 1 ] = v2;
   data[ 2 ] = v3;
}

template< int Size, typename Real >
const Real& tnlTuple< Size, Real > :: operator[]( int i ) const
{
   assert( i >= 0 && i < Size );
   return data[ i ];
};

template< int Size, typename Real >
Real& tnlTuple< Size, Real > :: operator[]( int i )
{
   assert( i < Size );
   return data[ i ];
};

template< int Size, typename Real >
Real& tnlTuple< Size, Real > :: x()
{
   tnlAssert( Size > 0, cerr << "Size = " << Size << endl; );
   if( Size < 1 )
   {
      cerr << "The size of the tnlTuple is too small to get x coordinate." << endl;
      abort();
   }
   return data[ 0 ];
};

template< int Size, typename Real >
const Real& tnlTuple< Size, Real > :: x() const
{
   tnlAssert( Size > 0, cerr << "Size = " << Size << endl; );
   if( Size < 1 )
   {
      cerr << "The size of the tnlTuple is too small to get x coordinate." << endl;
      abort();
   }
   return data[ 0 ];
};

template< int Size, typename Real >
Real& tnlTuple< Size, Real > :: y()
{
   tnlAssert( Size > 1, cerr << "Size = " << Size << endl; );
   if( Size < 2 )
   {
      cerr << "The size of the tnlTuple is too small to get y coordinate." << endl;
      abort();
   }
   return data[ 1 ];
};

template< int Size, typename Real >
const Real& tnlTuple< Size, Real > :: y() const
{
   tnlAssert( Size > 1, cerr << "Size = " << Size << endl; );
   if( Size < 2 )
   {
      cerr << "The size of the tnlTuple is too small to get y coordinate." << endl;
      abort();
   }
   return data[ 1 ];

};

template< int Size, typename Real >
Real& tnlTuple< Size, Real > :: z()
{
   tnlAssert( Size > 2, cerr << "Size = " << Size << endl; );
   if( Size < 3 )
   {
      cerr << "The size of the tnlTuple is too small to get z coordinate." << endl;
      abort();
   }
   return data[ 2 ];
};

template< int Size, typename Real >
const Real& tnlTuple< Size, Real > :: z() const
{
   tnlAssert( Size > 2, cerr << "Size = " << Size << endl; );
   if( Size < 3 )
   {
      cerr << "The size of the tnlTuple is too small to get z coordinate." << endl;
      abort();
   }
   return data[ 2 ];
};

template< int Size, typename Real >
tnlTuple< Size, Real >& tnlTuple< Size, Real > :: operator += ( const tnlTuple& v )
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
tnlTuple< Size, Real >& tnlTuple< Size, Real > :: operator -= ( const tnlTuple& v )
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
tnlTuple< Size, Real >& tnlTuple< Size, Real > :: operator *= ( const Real& c )
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
tnlTuple< Size, Real > tnlTuple< Size, Real > :: operator + ( const tnlTuple& u ) const
{
   // TODO: Leads to sigsegv
   return tnlTuple( * this ) += u;
};

template< int Size, typename Real >
tnlTuple< Size, Real > tnlTuple< Size, Real > :: operator - ( const tnlTuple& u ) const
{
   // TODO: Leads to sigsegv
   return tnlTuple( * this ) -= u;
};

template< int Size, typename Real >
tnlTuple< Size, Real > tnlTuple< Size, Real > :: operator * ( const Real& c ) const
{
   return tnlTuple( * this ) *= c;
};

template< int Size, typename Real >
tnlTuple< Size, Real >& tnlTuple< Size, Real > :: operator = ( const tnlTuple& v )
{
   memcpy( &data[ 0 ], &v. data[ 0 ], Size * sizeof( Real ) );
   /*int i;
   for( i = 0; i < Size; i ++ )
      data[ i ] = v. data[ i ];*/
   return *this;
};

template< int Size, typename Real >
Real tnlTuple< Size, Real > :: operator * ( const tnlTuple& u ) const
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
bool tnlTuple< Size, Real > :: operator == ( const tnlTuple< Size, Real2 >& u ) const
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
bool tnlTuple< Size, Real > :: operator != ( const tnlTuple< Size, Real2 >& u ) const
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
bool tnlTuple< Size, Real > :: operator < ( const tnlTuple& u ) const
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
bool tnlTuple< Size, Real > :: operator <= ( const tnlTuple& u ) const
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
bool tnlTuple< Size, Real > :: operator > ( const tnlTuple& u ) const
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
bool tnlTuple< Size, Real > :: operator >= ( const tnlTuple& u ) const
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
bool tnlTuple< Size, Real > :: save( tnlFile& file ) const
{
   int size = Size;
   if( ! file. write( &size, 1 ) ||
       ! file. write( data, size ) )
      cerr << "Unable to write tnlTuple." << endl;
   return true;
};

template< int Size, typename Real >
bool tnlTuple< Size, Real > :: load( tnlFile& file)
{
   int size;
   if( ! file. read( &size, 1 ) )
   {
      cerr << "Unable to read tnlTuple." << endl;
      return false;
   }
   if( size != Size )
   {
      cerr << "You try to read tnlTuple with wrong size " << size
           << ". It should be " << Size << endl;
      return false;
   }
   if( ! file. read( data, size ) )
   {
      cerr << "Unable to read tnlTuple." << endl;
      return false;
   }
   return true;
}

template< int Size, typename Real >
tnlTuple< Size, Real > operator * ( const Real& c, const tnlTuple< Size, Real >& u )
{
   return u * c;
}



// TODO: remove
template< int Size, typename Real > bool Save( ostream& file, const tnlTuple< Size, Real >& vec )
{
   for( int i = 0; i < Size; i ++ )
      file. write( ( char* ) &vec[ i ], sizeof( Real ) );
   if( file. bad() ) return false;
   return true;
};

// TODO: remove
template< int Size, typename Real > bool Load( istream& file, tnlTuple< Size, Real >& vec )
{
   for( int i = 0; i < Size; i ++ )
      file. read( ( char* ) &vec[ i ], sizeof( Real ) );
   if( file. bad() ) return false;
   return true;
};

template< int Size, typename Real >
ostream& operator << ( ostream& str, const tnlTuple< Size, Real >& v )
{
   for( int i = 0; i < Size - 1; i ++ )
      str << v[ i ] << ", ";
   str << v[ Size - 1 ];
   return str;
};

template< int Size, typename Real > tnlString GetParameterType( const tnlTuple< Size, Real >& )
{ 
   return tnlString( "tnlTuple< " ) +
          tnlString( Size ) +
          tnlString( ", " ) +
          tnlString( GetParameterType( Real( 0 ) ) ) +
          tnlString( " >" );
};

#endif
