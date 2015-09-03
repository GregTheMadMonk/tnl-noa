/***************************************************************************
                          mfuncs.h  -  description
                             -------------------
    begin                : 2005/07/05
    copyright            : (C) 2005 by Tomas Oberhuber
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

#ifndef mfuncsH
#define mfuncsH

#include <math.h>
#include <stdlib.h>
#include <core/tnlCuda.h>

template< typename Type1, typename Type2 >
__cuda_callable__
Type1 Min( const Type1& a, const Type2& b )
{
   return a < b ? a : b;
};

template< typename Type1, typename Type2 >
__cuda_callable__
Type1 Max( const Type1& a, const Type2& b )
{
   return a > b ? a : b;
};

template< typename Type >
__cuda_callable__
void Swap( Type& a, Type& b )
{
   Type tmp( a );
   a = b;
   b = tmp;
};

template< class T >
__cuda_callable__
T Sign( const T& a )
{
   if( a < ( T ) 0 ) return -1;
   if( a == ( T ) 0 ) return 0;
   return 1;
};

template< class T >
__cuda_callable__
T tnlAbs( const T& n )
{
   if( n < ( T ) 0 )
      return -n;
   return n;
};

inline int tnlAbs( const int& n )
{
   return abs( n );
};

inline float tnlAbs( const float& f )
{
   return fabs( f );
};

inline double tnlAbs( const double& d )
{
   return fabs( d );
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
   return ( x & ( x - 1 ) == 0 );
}

__cuda_callable__
inline bool isPow2( long int x )
{
   return ( x & ( x - 1 ) == 0 );
}


#endif
