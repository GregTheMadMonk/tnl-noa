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

////
// Non-static comparison
template< typename T1,
          typename T2,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value,
          ExpressionVariableType T2Type = ExpressionVariableTypeGetter< T2 >::value >
struct Comparison
{
};

/////
// Comparison of two vector expressions
template< typename T1,
          typename T2 >
struct Comparison< T1, T2, VectorVariable, VectorVariable >
{


   bool EQ( const T1& a, const T2& b )
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

   bool NE( const T1& a, const T2& b )
   {
      return ! EQ( a, b );
   }

   bool GT( const T1& a, const T2& b )
   {
      TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );

      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a[ i ] > b[ i ] ); };
      auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
      auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
      return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );
   }

   bool LE( const T1& a, const T2& b )
   {
      return ! GT( a, b );
   }

   bool LT( const T1& a, const T2& b )
   {
      TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );

      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a[ i ] < b[ i ] ); };
      auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
      auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
      return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );
   }

   bool GE( const T1& a, const T2& b )
   {
      return ! LT( a, b );
   }
};

/////
// Comparison of number and vector expression
template< typename T1,
          typename T2 >
struct Comparison< T1, T2, ArithmeticVariable, VectorVariable >
{


   bool EQ( const T1& a, const T2& b )
   {
      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a == b[ i ] ); };
      auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
      auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
      return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );
   }

   bool NE( const T1& a, const T2& b )
   {
      return ! EQ( a, b );
   }

   bool GT( const T1& a, const T2& b )
   {
      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a > b[ i ] ); };
      auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
      auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
      return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );
   }

   bool LE( const T1& a, const T2& b )
   {
      return ! GT( a, b );
   }

   bool LT( const T1& a, const T2& b )
   {
      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a < b[ i ] ); };
      auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
      auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
      return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );
   }

   bool GE( const T1& a, const T2& b )
   {
      return ! LT( a, b );
   }
};

/////
// Comparison of vector expressions and number
template< typename T1,
          typename T2 >
struct Comparison< T1, T2, VectorVariable, ArithmeticVariable >
{


   bool EQ( const T1& a, const T2& b )
   {
      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a[ i ] == b ); };
      auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
      auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
      return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );
   }

   bool NE( const T1& a, const T2& b )
   {
      return ! EQ( a, b );
   }

   bool GT( const T1& a, const T2& b )
   {
      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a[ i ] > b ); };
      auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
      auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
      return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );
   }

   bool LE( const T1& a, const T2& b )
   {
      return ! GT( a, b );
   }

   bool LT( const T1& a, const T2& b )
   {
      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return  ( a[ i ] < b ); };
      auto reduction = [=] __cuda_callable__ ( bool& a, const bool& b ) { a &= b; };
      auto volatileReduction = [=] __cuda_callable__ ( volatile bool& a, volatile bool& b ) { a &= b; };
      return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );
   }

   bool GE( const T1& a, const T2& b )
   {
      return ! LT( a, b );
   }
};


      } //namespace Expressions
   } // namespace Containers
} // namespace TNL
