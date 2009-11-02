/***************************************************************************
                          norms.h  -  description
                             -------------------
    begin                : 2007/07/05
    copyright            : (C) 2007 by Tomá¹ Oberhuber
    email                : oberhuber@seznam.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef normsH
#define normsH

#include "mGrid2D.h"

template< class T > T GetL1Norm( const mGrid2D< T >& u )
{
   long int size = u. GetSize();
   const T* _u = u. Data(); 
   long int i;
   T result( 0 );
   for( i = 0; i < size; i ++ )
      result += fabs( _u[ i ] );
   result *= u. GetHx() * u. GetHy();
   return result;
};

template< class T > T GetL2Norm( const mGrid2D< T >& u )
{
   long int size = u. GetSize();
   const T* _u = u. Data(); 
   long int i;
   T result( 0 );
   for( i = 0; i < size; i ++ )
      result += _u[ i ] * _u[ i ];
   result *= u. GetHx() * u. GetHy();
   return sqrt( result );
};

template< class T > T GetMaxNorm( const mGrid2D< T >& u )
{
   long int size = u. GetSize();
   const T* _u = u. Data(); 
   long int i;
   T result( 0 );
   for( i = 0; i < size; i ++ )
      result = Max( result, fabs( _u[ i ] ) );
   return result;
};

template< class T > T GetDiffL1Norm( const mGrid2D< T >& u1,
                                     const mGrid2D< T >& u2 )
{
   assert( u1. GetSize() == u2. GetSize() );
   assert( u1. GetHx() == u2. GetHx() );
   assert( u1. GetHy() == u2. GetHy() );
   long int size = u1. GetSize();
   const T* _u1 = u1. Data(); 
   const T* _u2 = u2. Data(); 
   long int i;
   T result( 0 );
   for( i = 0; i < size; i ++ )
   {
      T diff = _u1[ i ] - _u2[ i ];
      result += fabs( diff );
   }
   result *= u1. GetHx() * u1. GetHy();
   return result;
};

template< class T > T GetDiffL2Norm( const mGrid2D< T >& u1,
                                     const mGrid2D< T >& u2 )
{
   assert( u1. GetSize() == u2. GetSize() );
   assert( u1. GetHx() == u2. GetHx() );
   assert( u1. GetHy() == u2. GetHy() );
   long int size = u1. GetSize();
   const T* _u1 = u1. Data(); 
   const T* _u2 = u2. Data(); 
   long int i;
   T result( 0 );
   for( i = 0; i < size; i ++ )
   {
      T diff = _u1[ i ] - _u2[ i ];
      result += diff * diff;
   }
   result *= u1. GetHx() * u1. GetHy();
   return sqrt( result );
};

template< class T > T GetDiffMaxNorm( const mGrid2D< T >& u1,
                                     const mGrid2D< T >& u2 )
{
   assert( u1. GetSize() == u2. GetSize() );
   long int size = u1. GetSize();
   const T* _u1 = u1. Data(); 
   const T* _u2 = u2. Data(); 
   long int i;
   T result( 0 );
   for( i = 0; i < size; i ++ )
   {
      T diff = _u1[ i ] - _u2[ i ];
      result = Max( result, fabs( diff ) );
   }
   return result;
};

#endif
