/***************************************************************************
                          mfuncs.h  -  description
                             -------------------
    begin                : 2005/07/05
    copyright            : (C) 2005 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <cmath>
#include <stdlib.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {

template< typename Type1, typename Type2 >
__cuda_callable__
Type1 min( const Type1& a, const Type2& b )
{
   return a < b ? a : b;
};

template< typename Type1, typename Type2 >
__cuda_callable__
Type1 max( const Type1& a, const Type2& b )
{
   return a > b ? a : b;
};

template< typename Type >
__cuda_callable__
void swap( Type& a, Type& b )
{
   Type tmp( a );
   a = b;
   b = tmp;
};

template< class T >
__cuda_callable__
T sign( const T& a )
{
   if( a < ( T ) 0 ) return -1;
   if( a == ( T ) 0 ) return 0;
   return 1;
};

template< class T >
__cuda_callable__
T abs( const T& n )
{
   if( n < ( T ) 0 )
      return -n;
   return n;
};

__cuda_callable__
inline int abs( const int& n )
{
   return ::abs( n );
};

__cuda_callable__
inline float abs( const float& f )
{
   return ::fabs( f );
};

__cuda_callable__
inline double abs( const double& d )
{
   return ::fabs( d );
};

template< typename Real >
__cuda_callable__
bool isSmall( const Real& v,
              const Real& tolerance = 1.0e-5 )
{
   return ( -tolerance <= v && v <= tolerance );
}

__cuda_callable__
inline int roundUpDivision( const int num, const int div )
{
   return num / div + ( num % div != 0 );
}

__cuda_callable__
inline int roundToMultiple( int number, int multiple )
{
   return multiple*( number/ multiple + ( number % multiple != 0 ) );
}

__cuda_callable__
inline bool isPow2( int x )
{
   return ( ( x & ( x - 1 ) ) == 0 );
}

__cuda_callable__
inline bool isPow2( long int x )
{
   return ( ( x & ( x - 1 ) ) == 0 );
}

} // namespace TNL

