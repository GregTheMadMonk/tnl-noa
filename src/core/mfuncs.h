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

template< class T > T Min( const T& a, const T& b )
{
   return a < b ? a : b;
};

template< class T > T Max( const T& a, const T& b )
{
   return a > b ? a : b;
};

template< class T > void Swap( T& a, T& b )
{
   T tmp( a );
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

#endif
