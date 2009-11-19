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
   if( a < 0 ) return -1;
   if( a == 0 ) return 0;
   return 1;
}

#endif
