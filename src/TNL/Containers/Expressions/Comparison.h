/***************************************************************************
                          Comparison.h  -  description
                             -------------------
    begin                : Apr 19, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <stdexcept>

#include <TNL/Assert.h>
#include <TNL/Containers/Expressions/ExpressionVariableType.h>
#include <TNL/Containers/Algorithms/Reduction.h>
#include <TNL/Containers/Algorithms/ArrayOperations.h>

namespace TNL {
namespace Containers {
namespace Expressions {

////
// Non-static comparison
template< typename T1,
          typename T2,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value,
          ExpressionVariableType T2Type = ExpressionVariableTypeGetter< T2 >::value >
struct Comparison;

template< typename T1,
          typename T2,
          bool BothAreVectors = IsArrayType< T1 >::value && IsArrayType< T2 >::value >
struct VectorComparison;

// If both operands are vectors we compare them using array operations.
// It allows to compare vectors on different devices
template< typename T1, typename T2 >
struct VectorComparison< T1, T2, true >
{
   static bool EQ( const T1& a, const T2& b )
   {
      if( a.getSize() != b.getSize() )
         return false;
      if( a.getSize() == 0 )
         return true;
      return Algorithms::ArrayOperations< typename T1::DeviceType, typename T2::DeviceType >::compare( a.getData(), b.getData(), a.getSize() );
   }
};

// If both operands are not vectors we compare them parallel reduction
template< typename T1, typename T2 >
struct VectorComparison< T1, T2, false >
{
   static bool EQ( const T1& a, const T2& b )
   {
      if( ! std::is_same< typename T1::DeviceType, typename T2::DeviceType >::value )
         throw std::runtime_error( "Cannot compare two expressions with different DeviceType." );

      if( a.getSize() != b.getSize() )
         return false;

      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return a[ i ] == b[ i ]; };
      return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), std::logical_and<>{}, fetch, true );
   }
};

/////
// Comparison of two vector expressions
template< typename T1,
          typename T2 >
struct Comparison< T1, T2, VectorExpressionVariable, VectorExpressionVariable >
{
   static bool EQ( const T1& a, const T2& b )
   {
      return VectorComparison< T1, T2 >::EQ( a, b );
   }

   static bool NE( const T1& a, const T2& b )
   {
      return ! EQ( a, b );
   }

   static bool GT( const T1& a, const T2& b )
   {
      if( ! std::is_same< typename T1::DeviceType, typename T2::DeviceType >::value )
         throw std::runtime_error( "Cannot compare two expressions with different DeviceType." );
      TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );

      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return a[ i ] > b[ i ]; };
      return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), std::logical_and<>{}, fetch, true );
   }

   static bool GE( const T1& a, const T2& b )
   {
      if( ! std::is_same< typename T1::DeviceType, typename T2::DeviceType >::value )
         throw std::runtime_error( "Cannot compare two expressions with different DeviceType." );
      TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );

      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return a[ i ] >= b[ i ]; };
      return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), std::logical_and<>{}, fetch, true );
   }

   static bool LT( const T1& a, const T2& b )
   {
      if( ! std::is_same< typename T1::DeviceType, typename T2::DeviceType >::value )
         throw std::runtime_error( "Cannot compare two expressions with different DeviceType." );
      TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );

      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return a[ i ] < b[ i ]; };
      return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), std::logical_and<>{}, fetch, true );
   }

   static bool LE( const T1& a, const T2& b )
   {
      if( ! std::is_same< typename T1::DeviceType, typename T2::DeviceType >::value )
         throw std::runtime_error( "Cannot compare two expressions with different DeviceType." );
      TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );

      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return a[ i ] <= b[ i ]; };
      return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), std::logical_and<>{}, fetch, true );
   }
};

/////
// Comparison of number and vector expression
template< typename T1,
          typename T2 >
struct Comparison< T1, T2, ArithmeticVariable, VectorExpressionVariable >
{
   static bool EQ( const T1& a, const T2& b )
   {
      using DeviceType = typename T2::DeviceType;
      using IndexType = typename T2::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return a == b[ i ]; };
      return Algorithms::Reduction< DeviceType >::reduce( b.getSize(), std::logical_and<>{}, fetch, true );
   }

   static bool NE( const T1& a, const T2& b )
   {
      return ! EQ( a, b );
   }

   static bool GT( const T1& a, const T2& b )
   {
      using DeviceType = typename T2::DeviceType;
      using IndexType = typename T2::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return a > b[ i ]; };
      return Algorithms::Reduction< DeviceType >::reduce( b.getSize(), std::logical_and<>{}, fetch, true );
   }

   static bool GE( const T1& a, const T2& b )
   {
      using DeviceType = typename T2::DeviceType;
      using IndexType = typename T2::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return a >= b[ i ]; };
      return Algorithms::Reduction< DeviceType >::reduce( b.getSize(), std::logical_and<>{}, fetch, true );
   }

   static bool LT( const T1& a, const T2& b )
   {
      using DeviceType = typename T2::DeviceType;
      using IndexType = typename T2::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return a < b[ i ]; };
      return Algorithms::Reduction< DeviceType >::reduce( b.getSize(), std::logical_and<>{}, fetch, true );
   }

   static bool LE( const T1& a, const T2& b )
   {
      using DeviceType = typename T2::DeviceType;
      using IndexType = typename T2::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return a <= b[ i ]; };
      return Algorithms::Reduction< DeviceType >::reduce( b.getSize(), std::logical_and<>{}, fetch, true );
   }
};

/////
// Comparison of vector expressions and number
template< typename T1,
          typename T2 >
struct Comparison< T1, T2, VectorExpressionVariable, ArithmeticVariable >
{
   static bool EQ( const T1& a, const T2& b )
   {
      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return a[ i ] == b; };
      return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), std::logical_and<>{}, fetch, true );
   }

   static bool NE( const T1& a, const T2& b )
   {
      return ! EQ( a, b );
   }

   static bool GT( const T1& a, const T2& b )
   {
      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return a[ i ] > b; };
      return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), std::logical_and<>{}, fetch, true );
   }

   static bool GE( const T1& a, const T2& b )
   {
      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return a[ i ] >= b; };
      return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), std::logical_and<>{}, fetch, true );
   }

   static bool LT( const T1& a, const T2& b )
   {
      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return a[ i ] < b; };
      return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), std::logical_and<>{}, fetch, true );
   }

   static bool LE( const T1& a, const T2& b )
   {
      using DeviceType = typename T1::DeviceType;
      using IndexType = typename T1::IndexType;

      auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool { return a[ i ] <= b; };
      return Algorithms::Reduction< DeviceType >::reduce( a.getSize(), std::logical_and<>{}, fetch, true );
   }
};

} // namespace Expressions
} // namespace Containers
} // namespace TNL
