/***************************************************************************
                          VerticalOperations.h  -  description
                             -------------------
    begin                : May 1, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <limits>
#include <type_traits>

#include <TNL/Algorithms/Reduction.h>

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

   const auto view = expression.getConstView();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return view[ i ]; };
   auto reduction = [] __cuda_callable__ ( const ResultType& a, const ResultType& b ) { return TNL::min( a, b ); };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, fetch, std::numeric_limits< ResultType >::max() );
}

template< typename Expression >
auto ExpressionArgMin( const Expression& expression )
-> std::pair< std::decay_t< decltype( expression[0] ) >, typename Expression::IndexType >
{
   using ResultType = std::decay_t< decltype( expression[0] ) >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return view[ i ]; };
   auto reduction = [] __cuda_callable__ ( ResultType& a, const ResultType& b, IndexType& aIdx, const IndexType& bIdx ) {
      if( a > b ) {
         a = b;
         aIdx = bIdx;
      }
      else if( a == b && bIdx < aIdx )
         aIdx = bIdx;
   };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduceWithArgument( expression.getSize(), reduction, fetch, std::numeric_limits< ResultType >::max() );
}

template< typename Expression >
auto ExpressionMax( const Expression& expression ) -> std::decay_t< decltype( expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] ) >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return view[ i ]; };
   auto reduction = [] __cuda_callable__ ( const ResultType& a, const ResultType& b ) { return TNL::max( a, b ); };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, fetch, std::numeric_limits< ResultType >::lowest() );
}

template< typename Expression >
auto ExpressionArgMax( const Expression& expression )
-> std::pair< std::decay_t< decltype( expression[0] ) >, typename Expression::IndexType >
{
   using ResultType = std::decay_t< decltype( expression[0] ) >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return view[ i ]; };
   auto reduction = [] __cuda_callable__ ( ResultType& a, const ResultType& b, IndexType& aIdx, const IndexType& bIdx ) {
      if( a < b ) {
         a = b;
         aIdx = bIdx;
      }
      else if( a == b && bIdx < aIdx )
         aIdx = bIdx;
   };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduceWithArgument( expression.getSize(), reduction, fetch, std::numeric_limits< ResultType >::lowest() );
}

template< typename Expression >
auto ExpressionSum( const Expression& expression ) -> std::decay_t< decltype( expression[0] + expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] + expression[0] ) >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return view[ i ]; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), std::plus<>{}, fetch, (ResultType) 0 );
}

template< typename Expression >
auto ExpressionProduct( const Expression& expression ) -> std::decay_t< decltype( expression[0] * expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] * expression[0] ) >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return view[ i ]; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), std::multiplies<>{}, fetch, (ResultType) 1 );
}

template< typename Expression >
auto ExpressionLogicalAnd( const Expression& expression ) -> std::decay_t< decltype( expression[0] && expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] && expression[0] ) >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return view[ i ]; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), std::logical_and<>{}, fetch, std::numeric_limits< ResultType >::max() );
}

template< typename Expression >
auto ExpressionLogicalOr( const Expression& expression ) -> std::decay_t< decltype( expression[0] || expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] || expression[0] ) >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return view[ i ]; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), std::logical_or<>{}, fetch, (ResultType) 0 );
}

template< typename Expression >
auto ExpressionBinaryAnd( const Expression& expression ) -> std::decay_t< decltype( expression[0] & expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] & expression[0] ) >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return view[ i ]; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), std::bit_and<>{}, fetch, std::numeric_limits< ResultType >::max() );
}

template< typename Expression >
auto ExpressionBinaryOr( const Expression& expression ) -> std::decay_t< decltype( expression[0] | expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] | expression[0] ) >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return view[ i ]; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), std::bit_or<>{}, fetch, (ResultType) 0 );
}

} // namespace Expressions
} // namespace Containers
} // namespace TNL
