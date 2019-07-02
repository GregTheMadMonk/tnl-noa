/***************************************************************************
                          Comparison.h  -  description
                             -------------------
    begin                : Apr 19, 2019
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
      static_assert( a.getSize() == b.getSize(), "Sizes of expressions to be compared do not fit." );
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
      static_assert( a.getSize() == b.getSize(), "Sizes of expressions to be compared do not fit." );
      for( int i = 0; i < a.getSize(); i++ )
         if( a[ i ] <= b[ i ] )
            return false;
      return true;
   }

   __cuda_callable__
   static bool LE( const T1& a, const T2& b )
   {
      return ! GT( a, b );
   }

   __cuda_callable__
   static bool LT( const T1& a, const T2& b )
   {
      static_assert( a.getSize() == b.getSize(), "Sizes of expressions to be compared do not fit." );
      for( int i = 0; i < a.getSize(); i++ )
         if( a[ i ] >= b[ i ] )
            return false;
      return true;
   }

   __cuda_callable__
   static bool GE( const T1& a, const T2& b )
   {
      return ! LT( a, b );
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
      for( int i = 0; i < a.getSize(); i++ )
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
      for( int i = 0; i < a.getSize(); i++ )
         if( a <= b[ i ] )
            return false;
      return true;
   }

   __cuda_callable__
   static bool LE( const T1& a, const T2& b )
   {
      return ! GT( a, b );
   }

   __cuda_callable__
   static bool LT( const T1& a, const T2& b )
   {
      for( int i = 0; i < a.getSize(); i++ )
         if( a >= b[ i ] )
            return false;
      return true;
   }

   __cuda_callable__
   static bool GE( const T1& a, const T2& b )
   {
      return ! LT( a, b );
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
   static bool LE( const T1& a, const T2& b )
   {
      return ! GT( a, b );
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
   static bool GE( const T1& a, const T2& b )
   {
      return ! LT( a, b );
   }
};


////
// Non-static comparison
template< typename T1,
          typename T2 >
__cuda_callable__
bool ComparisonEQ( const T1& a, const T2& b )
{
   if( a.getSize() != b.getSize() )
      return false;
   if( a.getSize() == 0 )
      return true;

   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a[ i ] == b[ i ] ); };
   auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
   return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );
}

template< typename T1,
          typename T2 >
__cuda_callable__
bool ComparisonNE( const T1& a, const T2& b )
{
   return ! ComparisonEQ( a, b );
}

template< typename T1,
          typename T2 >
__cuda_callable__
bool ComparisonGT( const T1& a, const T2& b )
{
   TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );

   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a[ i ] > b[ i ] ); };
   auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
   return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );
}

template< typename T1,
          typename T2 >
__cuda_callable__
bool ComparisonLE( const T1& a, const T2& b )
{
   return ! ComparisonGT( a, b );
}

template< typename T1,
          typename T2 >
__cuda_callable__
bool ComparisonLT( const T1& a, const T2& b )
{
   TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );

   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a[ i ] < b[ i ] ); };
   auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
   return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );
}

template< typename T1,
          typename T2 >
__cuda_callable__
bool ComparisonGE( const T1& a, const T2& b )
{
   return ! ComparisonLT( a, b );
}

      } //namespace Expressions
   } // namespace Containers
} // namespace TNL
