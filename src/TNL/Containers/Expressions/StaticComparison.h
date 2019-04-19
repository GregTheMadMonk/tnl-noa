/***************************************************************************
                          StaticComparison.h  -  description
                             -------------------
    begin                : Apr 19, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Assert.h>

namespace TNL {
   namespace Containers {
      namespace Expressions {

template< typename T1,
          typename T2 >
bool StaticComparisonEQ( const T1& a, const T2& b )
{
   TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );
   for( int i = 0; i < a.getSize(); i++ )
      if( a[ i ] != b[ i ] )
         return false;
   return true;
}

template< typename T1,
          typename T2 >
bool StaticComparisonNE( const T1& a, const T2& b )
{
   return ! StaticComparisonEQ( a, b );
}

template< typename T1,
          typename T2 >
bool StaticComparisonGT( const T1& a, const T2& b )
{
   TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );
   for( int i = 0; i < a.getSize(); i++ )
      if( a[ i ] <= b[ i ] )
         return false;
   return true;
}

template< typename T1,
          typename T2 >
bool StaticComparisonLE( const T1& a, const T2& b )
{
   return ! StaticComparisonGT( a, b );
}

template< typename T1,
          typename T2 >
bool StaticComparisonLT( const T1& a, const T2& b )
{
   TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );
   for( int i = 0; i < a.getSize(); i++ )
      if( a[ i ] >= b[ i ] )
         return false;
   return true;
}

template< typename T1,
          typename T2 >
bool StaticComparisonGE( const T1& a, const T2& b )
{
   return ! StaticComparisonLT( a, b );
}

      } //namespace Expressions
   } // namespace Containers
} // namespace TNL