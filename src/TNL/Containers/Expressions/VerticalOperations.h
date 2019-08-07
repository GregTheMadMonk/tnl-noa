/***************************************************************************
                          VerticalOperations.h  -  description
                             -------------------
    begin                : May 1, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <limits>
#include <type_traits>

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
auto ExpressionMin( const Expression& expression ) -> std::decay_t< decltype( expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] ) >;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return expression[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a = a < b ? a : b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a = a < b ? a : b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, std::numeric_limits< ResultType >::max() );
}

template< typename Expression >
auto ExpressionArgMin( const Expression& expression, typename Expression::IndexType& arg ) -> std::decay_t< decltype( expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] ) >;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return expression[ i ]; };
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
auto ExpressionMax( const Expression& expression ) -> std::decay_t< decltype( expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] ) >;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return expression[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a = a > b ? a : b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a = a > b ? a : b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, std::numeric_limits< ResultType >::lowest() );
}

template< typename Expression >
auto ExpressionArgMax( const Expression& expression, typename Expression::IndexType& arg ) -> std::decay_t< decltype( expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] ) >;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return expression[ i ]; };
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
auto ExpressionSum( const Expression& expression ) -> std::decay_t< decltype( expression[0] + expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] + expression[0] ) >;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return expression[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a += b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a += b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, (ResultType) 0 );
}

template< typename Expression >
auto ExpressionL1Norm( const Expression& expression ) -> std::decay_t< decltype( expression[0] + expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] + expression[0] ) >;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return TNL::abs( expression[ i ] ); };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a += b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a += b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, (ResultType) 0 );
}

template< typename Expression >
auto ExpressionL2Norm( const Expression& expression ) -> std::decay_t< decltype( expression[0] * expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] * expression[0] ) >;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return expression[ i ] * expression[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a += b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a += b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, (ResultType) 0 );
}

template< typename Expression, typename Real >
auto ExpressionLpNorm( const Expression& expression, const Real& p ) -> std::decay_t< decltype( TNL::pow( expression[0], p ) ) >
{
   using ResultType = std::decay_t< decltype( TNL::pow( expression[0], p ) ) >;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return TNL::pow( TNL::abs( expression[ i ] ), p ); };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a += b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a += b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, (ResultType) 0 );
}

template< typename Expression >
auto ExpressionProduct( const Expression& expression ) -> std::decay_t< decltype( expression[0] * expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] * expression[0] ) >;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return expression[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a *= b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a *= b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, (ResultType) 1 );
}

template< typename Expression >
auto ExpressionLogicalAnd( const Expression& expression ) -> std::decay_t< decltype( expression[0] && expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] && expression[0] ) >;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return expression[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a = a && b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a = a && b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, std::numeric_limits< ResultType >::max() );
}

template< typename Expression >
auto ExpressionLogicalOr( const Expression& expression ) -> std::decay_t< decltype( expression[0] || expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] || expression[0] ) >;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return expression[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a = a || b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a = a || b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, (ResultType) 0 );
}

template< typename Expression >
auto ExpressionBinaryAnd( const Expression& expression ) -> std::decay_t< decltype( expression[0] & expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] & expression[0] ) >;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return expression[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a = a & b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a = a & b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, std::numeric_limits< ResultType >::max() );
}

template< typename Expression >
auto ExpressionBinaryOr( const Expression& expression ) -> std::decay_t< decltype( expression[0] | expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] | expression[0] ) >;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return expression[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a = a | b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a = a | b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, (ResultType) 0 );
}

} // namespace Expressions
} // namespace Containers
} // namespace TNL
