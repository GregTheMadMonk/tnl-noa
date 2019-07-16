/***************************************************************************
                          VerticalOperations.h  -  description
                             -------------------
    begin                : May 1, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Assert.h>
#include <TNL/Containers/Algorithms/Reduction.h>

////
// By vertical operations we mean those applied across vector elements or
// vector expression elements. It means for example minim/maximum of all
// vector elements etc.
namespace TNL {
   namespace Containers {
      namespace Expressions {

////
// Vertical operations
template< typename Expression >
auto ExpressionMin( const Expression& expression ) -> typename std::remove_reference< decltype( expression[ 0 ] ) >::type
{
   using ResultType = typename std::remove_cv< typename std::remove_reference< decltype( expression[ 0 ] ) >::type >::type;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  expression[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a = a < b ? a : b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a = a < b ? a : b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, std::numeric_limits< ResultType >::max() );
}

template< typename Expression >
auto ExpressionArgMin( const Expression& expression, typename Expression::IndexType& arg ) -> typename std::remove_reference< decltype( expression[ 0 ] ) >::type
{
   using ResultType = typename std::remove_cv< typename std::remove_reference< decltype( expression[ 0 ] ) >::type >::type;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  expression[ i ]; };
   auto reduction = [=] __cuda_callable__ ( IndexType& aIdx, const IndexType& bIdx, ResultType& a, const ResultType& b ) {
      if( a > b ) {
         a = b;
         aIdx = bIdx;
      }
      else if( a == b && bIdx < aIdx )
         aIdx = bIdx;

   };
   auto volatileReduction = [=] __cuda_callable__ ( volatile IndexType& aIdx, volatile IndexType& bIdx, volatile ResultType& a, volatile ResultType& b ) {
      if( a > b ) {
         a = b;
         aIdx = bIdx;
      }
      else if( a == b && bIdx < aIdx )
         aIdx = bIdx;

   };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduceWithArgument( expression.getSize(), arg, reduction, volatileReduction, fetch, std::numeric_limits< ResultType >::max() );
}

template< typename Expression >
auto ExpressionMax( const Expression& expression ) -> typename std::remove_reference< decltype( expression[ 0 ] ) >::type
{
   using ResultType = typename std::remove_cv< typename std::remove_reference< decltype( expression[ 0 ] ) >::type >::type;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  expression[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a = a > b ? a : b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a = a > b ? a : b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, std::numeric_limits< ResultType >::lowest() );
}

template< typename Expression >
auto ExpressionArgMax( const Expression& expression, typename Expression::IndexType& arg ) -> typename std::remove_reference< decltype( expression[ 0 ] ) >::type
{
   using ResultType = typename std::remove_cv< typename std::remove_reference< decltype( expression[ 0 ] ) >::type >::type;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  expression[ i ]; };
   auto reduction = [=] __cuda_callable__ ( IndexType& aIdx, const IndexType& bIdx, ResultType& a, const ResultType& b ) {
      if( a < b ) {
         a = b;
         aIdx = bIdx;
      }
      else if( a == b && bIdx < aIdx )
         aIdx = bIdx;
   };
   auto volatileReduction = [=] __cuda_callable__ ( volatile IndexType& aIdx, volatile IndexType& bIdx, volatile ResultType& a, volatile ResultType& b ) {
      if( a < b ) {
         a = b;
         aIdx = bIdx;
      }
      else if( a == b && bIdx < aIdx )
         aIdx = bIdx;
   };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduceWithArgument( expression.getSize(), arg, reduction, volatileReduction, fetch, std::numeric_limits< ResultType >::lowest() );
}

template< typename Expression >
auto ExpressionSum( const Expression& expression ) -> 
   typename std::conditional<
      std::is_same< typename std::remove_cv< typename std::remove_reference< decltype( expression[ 0 ] ) >::type >::type, bool >::value,
      typename Expression::IndexType,
      typename std::remove_reference< decltype( expression[ 0 ] ) >::type
   >::type
{
   using ResultTypeBase = typename std::remove_cv< typename std::remove_reference< decltype( expression[ 0 ] ) >::type >::type;
   using IndexType = typename Expression::IndexType;
   using ResultType = typename std::conditional< std::is_same< ResultTypeBase, bool >::value, IndexType, ResultTypeBase >::type;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  expression[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a += b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a += b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, ( ResultType ) 0.0 );
}

template< typename Expression, typename Real >
auto ExpressionLpNorm( const Expression& expression, const Real& p ) -> 
   typename std::conditional<
      std::is_same< typename std::remove_cv< typename std::remove_reference< decltype( expression[ 0 ] ) >::type >::type, bool >::value,
      double, // TODO: Solve this some better way
      typename std::remove_reference< decltype( expression[ 0 ] ) >::type
   >::type
{
   using ResultTypeBase = typename std::remove_cv< typename std::remove_reference< decltype( expression[ 0 ] ) >::type >::type;
   using IndexType = typename Expression::IndexType;
   using ResultType = typename std::conditional< std::is_same< ResultTypeBase, bool >::value, double, ResultTypeBase >::type;

   if( p == ( Real ) 1.0 )
   {
      auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  TNL::abs( expression[ i ] ); };
      auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a += b; };
      auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a += b; };
      return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, ( ResultType ) 0.0 );
   }
   if( p == ( Real ) 2.0 )
   {
      auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  expression[ i ] * expression[ i ]; };
      auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a += b; };
      auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a += b; };
      return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, ( ResultType ) 0.0 );
   }
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  TNL::pow( expression[ i ], p ); };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a += b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a += b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, ( ResultType ) 0.0 );
}

template< typename Expression >
auto ExpressionProduct( const Expression& expression ) -> typename std::remove_reference< decltype( expression[ 0 ] ) >::type
{
   using ResultType = typename std::remove_cv< typename std::remove_reference< decltype( expression[ 0 ] ) >::type >::type;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  expression[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a *= b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a *= b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, ( ResultType ) 1.0 );
}

template< typename Expression >
bool ExpressionLogicalAnd( const Expression& expression )
{
   using ResultType = bool;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  expression[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a = a && b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a = a && b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, true );
}

template< typename Expression >
bool ExpressionLogicalOr( const Expression& expression )
{
   using ResultType = bool;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  expression[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a = a || b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a = a || b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, ( ResultType ) 0  );
}

template< typename Expression >
auto ExpressionBinaryAnd( const Expression& expression ) -> typename std::remove_reference< decltype( expression[ 0 ] ) >::type
{
   using ResultType = typename std::remove_cv< typename std::remove_reference< decltype( expression[ 0 ] ) >::type >::type;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  expression[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a = a & b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a = a & b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, std::numeric_limits< ResultType >::max() );
}

template< typename Expression >
auto ExpressionBinaryOr( const Expression& expression ) -> typename std::remove_reference< decltype( expression[ 0 ] ) >::type
{
   using ResultType = typename std::remove_cv< typename std::remove_reference< decltype( expression[ 0 ] ) >::type >::type;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  expression[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a = a | b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a = a | b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, ( ResultType ) 0 );
}

      } //namespace Expressions
   } // namespace Containers
} // namespace TNL
