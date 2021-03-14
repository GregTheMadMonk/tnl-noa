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
#include <TNL/Containers/Expressions/TypeTraits.h>

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
auto ExpressionMin( const Expression& expression )
-> RemoveET< std::decay_t< decltype( expression[0] ) > >
{
   using ResultType = RemoveET< std::decay_t< decltype( expression[0] ) > >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return view[ i ]; };
   auto reduction = [] __cuda_callable__ ( const ResultType& a, const ResultType& b )
   {
      // use argument-dependent lookup and make TNL::min available for unqualified calls
      using TNL::min;
      return min( a, b );
   };
   static_assert( std::numeric_limits< ResultType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's result type" );
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( ( IndexType ) 0, expression.getSize(), fetch, reduction, std::numeric_limits< ResultType >::max() );
}

template< typename Expression >
auto ExpressionArgMin( const Expression& expression )
-> RemoveET< std::pair< std::decay_t< decltype( expression[0] ) >, typename Expression::IndexType > >
{
   using ResultType = RemoveET< std::decay_t< decltype( expression[0] ) > >;
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
   static_assert( std::numeric_limits< ResultType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's result type" );
   return Algorithms::Reduction< typename Expression::DeviceType >::reduceWithArgument( ( IndexType ) 0, expression.getSize(), fetch, reduction, std::numeric_limits< ResultType >::max() );
}

template< typename Expression >
auto ExpressionMax( const Expression& expression )
-> RemoveET< std::decay_t< decltype( expression[0] ) > >
{
   using ResultType = RemoveET< std::decay_t< decltype( expression[0] ) > >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return view[ i ]; };
   auto reduction = [] __cuda_callable__ ( const ResultType& a, const ResultType& b )
   {
      // use argument-dependent lookup and make TNL::max available for unqualified calls
      using TNL::max;
      return max( a, b );
   };
   static_assert( std::numeric_limits< ResultType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's result type" );
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( ( IndexType ) 0, expression.getSize(), fetch, reduction, std::numeric_limits< ResultType >::lowest() );
}

template< typename Expression >
auto ExpressionArgMax( const Expression& expression )
-> RemoveET< std::pair< std::decay_t< decltype( expression[0] ) >, typename Expression::IndexType > >
{
   using ResultType = RemoveET< std::decay_t< decltype( expression[0] ) > >;
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
   static_assert( std::numeric_limits< ResultType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's result type" );
   return Algorithms::Reduction< typename Expression::DeviceType >::reduceWithArgument( ( IndexType ) 0, expression.getSize(), fetch, reduction, std::numeric_limits< ResultType >::lowest() );
}

template< typename Expression >
auto ExpressionSum( const Expression& expression )
-> RemoveET< std::decay_t< decltype( expression[0] + expression[0] ) > >
{
   using ResultType = RemoveET< std::decay_t< decltype( expression[0] + expression[0] ) > >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return view[ i ]; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( ( IndexType ) 0, expression.getSize(), fetch, std::plus<>{}, (ResultType) 0 );
}

template< typename Expression >
auto ExpressionProduct( const Expression& expression )
-> RemoveET< std::decay_t< decltype( expression[0] * expression[0] ) > >
{
   using ResultType = RemoveET< std::decay_t< decltype( expression[0] * expression[0] ) > >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return view[ i ]; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( ( IndexType ) 0, expression.getSize(), fetch, std::multiplies<>{}, (ResultType) 1 );
}

template< typename Expression >
auto ExpressionLogicalAnd( const Expression& expression )
-> RemoveET< std::decay_t< decltype( expression[0] && expression[0] ) > >
{
   using ResultType = RemoveET< std::decay_t< decltype( expression[0] && expression[0] ) > >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return view[ i ]; };
   static_assert( std::numeric_limits< ResultType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's result type" );
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( ( IndexType ) 0, expression.getSize(), fetch, std::logical_and<>{}, std::numeric_limits< ResultType >::max() );
}

template< typename Expression >
auto ExpressionLogicalOr( const Expression& expression )
-> RemoveET< std::decay_t< decltype( expression[0] || expression[0] ) > >
{
   using ResultType = RemoveET< std::decay_t< decltype( expression[0] || expression[0] ) > >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return view[ i ]; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( ( IndexType ) 0, expression.getSize(), fetch, std::logical_or<>{}, (ResultType) 0 );
}

template< typename Expression >
auto ExpressionBinaryAnd( const Expression& expression )
-> RemoveET< std::decay_t< decltype( expression[0] & expression[0] ) > >
{
   using ResultType = RemoveET< std::decay_t< decltype( expression[0] & expression[0] ) > >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return view[ i ]; };
   static_assert( std::numeric_limits< ResultType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's result type" );
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( ( IndexType ) 0, expression.getSize(), fetch, std::bit_and<>{}, std::numeric_limits< ResultType >::max() );
}

template< typename Expression >
auto ExpressionBinaryOr( const Expression& expression )
-> RemoveET< std::decay_t< decltype( expression[0] | expression[0] ) > >
{
   using ResultType = RemoveET< std::decay_t< decltype( expression[0] | expression[0] ) > >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return view[ i ]; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( ( IndexType ) 0, expression.getSize(), fetch, std::bit_or<>{}, (ResultType) 0 );
}

} // namespace Expressions
} // namespace Containers
} // namespace TNL
