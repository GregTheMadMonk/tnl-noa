/***************************************************************************
                          StaticComparison.h  -  description
                             -------------------
    begin                : Jul 3, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Assert.h>
#include <TNL/Containers/Algorithms/Reduction.h>

namespace TNL {
   namespace Containers {
      namespace Expressions {

template< typename T1,
          typename T2,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value,
          ExpressionVariableType T2Type = ExpressionVariableTypeGetter< T2 >::value >
struct StaticComparison
{
};

/////
// Static comparison of vector expressions
template< typename T1,
          typename T2 >
struct StaticComparison< T1, T2, VectorVariable, VectorVariable >
{

   __cuda_callable__
   static bool EQ( const T1& a, const T2& b )
   {
      TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );
      for( int i = 0; i < a.getSize(); i++ )
         if( a[ i ] != b[ i ] )
            return false;
      return true;
   }

   __cuda_callable__
   static bool NE( const T1& a, const T2& b )
   {
      return ! EQ( a, b );
   }

   __cuda_callable__
   static bool GT( const T1& a, const T2& b )
   {
      TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );
      for( int i = 0; i < a.getSize(); i++ )
         if( a[ i ] <= b[ i ] )
            return false;
      return true;
   }

   __cuda_callable__
   static bool GE( const T1& a, const T2& b )
   {
      TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );
      for( int i = 0; i < a.getSize(); i++ )
         if( a[ i ] < b[ i ] )
            return false;
      return true;
   }

   __cuda_callable__
   static bool LT( const T1& a, const T2& b )
   {
      TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );
      for( int i = 0; i < a.getSize(); i++ )
         if( a[ i ] >= b[ i ] )
            return false;
      return true;
   }

   __cuda_callable__
   static bool LE( const T1& a, const T2& b )
   {
      TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );
      for( int i = 0; i < a.getSize(); i++ )
         if( a[ i ] > b[ i ] )
            return false;
      return true;
   }
};

/////
// Static comparison of number and vector expressions
template< typename T1,
          typename T2 >
struct StaticComparison< T1, T2, ArithmeticVariable, VectorVariable >
{

   __cuda_callable__
   static bool EQ( const T1& a, const T2& b )
   {
      for( int i = 0; i < b.getSize(); i++ )
         if( a != b[ i ] )
            return false;
      return true;
   }

   __cuda_callable__
   static bool NE( const T1& a, const T2& b )
   {
      return ! EQ( a, b );
   }

   __cuda_callable__
   static bool GT( const T1& a, const T2& b )
   {
      for( int i = 0; i < b.getSize(); i++ )
         if( a <= b[ i ] )
            return false;
      return true;
   }

   __cuda_callable__
   static bool GE( const T1& a, const T2& b )
   {
      for( int i = 0; i < b.getSize(); i++ )
         if( a < b[ i ] )
            return false;
      return true;
   }

   __cuda_callable__
   static bool LT( const T1& a, const T2& b )
   {
      for( int i = 0; i < b.getSize(); i++ )
         if( a >= b[ i ] )
            return false;
      return true;
   }

   __cuda_callable__
   static bool LE( const T1& a, const T2& b )
   {
      for( int i = 0; i < b.getSize(); i++ )
         if( a > b[ i ] )
            return false;
      return true;
   }
};

/////
// Static comparison of vector expressions and number
template< typename T1,
          typename T2 >
struct StaticComparison< T1, T2, VectorVariable, ArithmeticVariable >
{

   __cuda_callable__
   static bool EQ( const T1& a, const T2& b )
   {
      for( int i = 0; i < a.getSize(); i++ )
         if( a[ i ] != b )
            return false;
      return true;
   }

   __cuda_callable__
   static bool NE( const T1& a, const T2& b )
   {
      return ! EQ( a, b );
   }

   __cuda_callable__
   static bool GT( const T1& a, const T2& b )
   {
      for( int i = 0; i < a.getSize(); i++ )
         if( a[ i ] <= b )
            return false;
      return true;
   }

   __cuda_callable__
   static bool GE( const T1& a, const T2& b )
   {
      for( int i = 0; i < a.getSize(); i++ )
         if( a[ i ] < b )
            return false;
      return true;
   }

   __cuda_callable__
   static bool LT( const T1& a, const T2& b )
   {
      for( int i = 0; i < a.getSize(); i++ )
         if( a[ i ] >= b )
            return false;
      return true;
   }

   __cuda_callable__
   static bool LE( const T1& a, const T2& b )
   {
      for( int i = 0; i < a.getSize(); i++ )
         if( a[ i ] > b )
            return false;
      return true;
   }
};

      } //namespace Expressions
   } // namespace Containers
} // namespace TNL
