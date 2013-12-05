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

template< typename Type1, typename Type2 > Type1 Min( const Type1& a, const Type2& b )
{
   return a < b ? a : b;
};

template< typename Type1, typename Type2 > Type1 Max( const Type1& a, const Type2& b )
{
   return a > b ? a : b;
};

template< typename Type > void Swap( Type& a, Type& b )
{
   Type tmp( a );
   a = b;
   b = tmp;
};

template< class T > T Sign( const T& a )
{
   if( a < ( T ) 0 ) return -1;
   if( a == ( T ) 0 ) return 0;
   return 1;
};

template< class T >
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
bool isSmall( const Real& v,
              const Real& tolerance = 1.0e-5 )
{
   return ( -tolerance <= v && v <= tolerance );
}
/*template< typename T >
void swap( T& a, T& b)
{
   T aux;
   aux = a;
   a = b;
   b = aux;
}*/


#endif
