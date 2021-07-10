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

#include <TNL/Functional.h>
#include <TNL/Algorithms/reduce.h>
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
   return Algorithms::reduce< typename Expression::DeviceType >( ( IndexType ) 0, expression.getSize(), view, TNL::Min{}, TNL::Min::template getIdempotent< ResultType >() );
}

template< typename Expression >
auto ExpressionArgMin( const Expression& expression )
-> RemoveET< std::pair< std::decay_t< decltype( expression[0] ) >, typename Expression::IndexType > >
{
   using ResultType = RemoveET< std::decay_t< decltype( expression[0] ) > >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   return Algorithms::reduceWithArgument< typename Expression::DeviceType >( ( IndexType ) 0, expression.getSize(), view, TNL::MinWithArg{}, TNL::MinWithArg::template getIdempotent< ResultType >() );
}

template< typename Expression >
auto ExpressionMax( const Expression& expression )
-> RemoveET< std::decay_t< decltype( expression[0] ) > >
{
   using ResultType = RemoveET< std::decay_t< decltype( expression[0] ) > >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   return Algorithms::reduce< typename Expression::DeviceType >( ( IndexType ) 0, expression.getSize(), view, TNL::Max{}, TNL::Max::template getIdempotent< ResultType >() );
}

template< typename Expression >
auto ExpressionArgMax( const Expression& expression )
-> RemoveET< std::pair< std::decay_t< decltype( expression[0] ) >, typename Expression::IndexType > >
{
   using ResultType = RemoveET< std::decay_t< decltype( expression[0] ) > >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   return Algorithms::reduceWithArgument< typename Expression::DeviceType >( ( IndexType ) 0, expression.getSize(), view, TNL::MaxWithArg{}, TNL::MaxWithArg::template getIdempotent< ResultType >() );
}

template< typename Expression >
auto ExpressionSum( const Expression& expression )
-> RemoveET< std::decay_t< decltype( expression[0] + expression[0] ) > >
{
   using ResultType = RemoveET< std::decay_t< decltype( expression[0] + expression[0] ) > >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   return Algorithms::reduce< typename Expression::DeviceType >( ( IndexType ) 0, expression.getSize(), view, TNL::Plus{}, TNL::Plus::template getIdempotent< ResultType >() );
}

template< typename Expression >
auto ExpressionProduct( const Expression& expression )
-> RemoveET< std::decay_t< decltype( expression[0] * expression[0] ) > >
{
   using ResultType = RemoveET< std::decay_t< decltype( expression[0] * expression[0] ) > >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   return Algorithms::reduce< typename Expression::DeviceType >( ( IndexType ) 0, expression.getSize(), view, TNL::Multiplies{}, TNL::Multiplies::template getIdempotent< ResultType >() );
}

template< typename Expression >
auto ExpressionLogicalAnd( const Expression& expression )
-> RemoveET< std::decay_t< decltype( expression[0] && expression[0] ) > >
{
   using ResultType = RemoveET< std::decay_t< decltype( expression[0] && expression[0] ) > >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   return Algorithms::reduce< typename Expression::DeviceType >( ( IndexType ) 0, expression.getSize(), view, TNL::LogicalAnd{}, TNL::LogicalAnd::template getIdempotent< ResultType >() );
}

template< typename Expression >
auto ExpressionLogicalOr( const Expression& expression )
-> RemoveET< std::decay_t< decltype( expression[0] || expression[0] ) > >
{
   using ResultType = RemoveET< std::decay_t< decltype( expression[0] || expression[0] ) > >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   return Algorithms::reduce< typename Expression::DeviceType >( ( IndexType ) 0, expression.getSize(), view, TNL::LogicalOr{}, TNL::LogicalOr::template getIdempotent< ResultType >() );
}

template< typename Expression >
auto ExpressionBinaryAnd( const Expression& expression )
-> RemoveET< std::decay_t< decltype( expression[0] & expression[0] ) > >
{
   using ResultType = RemoveET< std::decay_t< decltype( expression[0] & expression[0] ) > >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   return Algorithms::reduce< typename Expression::DeviceType >( ( IndexType ) 0, expression.getSize(), view, TNL::BitAnd{}, TNL::BitAnd::template getIdempotent< ResultType >() );
}

template< typename Expression >
auto ExpressionBinaryOr( const Expression& expression )
-> RemoveET< std::decay_t< decltype( expression[0] | expression[0] ) > >
{
   using ResultType = RemoveET< std::decay_t< decltype( expression[0] | expression[0] ) > >;
   using IndexType = typename Expression::IndexType;

   const auto view = expression.getConstView();
   return Algorithms::reduce< typename Expression::DeviceType >( ( IndexType ) 0, expression.getSize(), view, TNL::BitOr{}, TNL::BitOr::template getIdempotent< ResultType >() );
}

} // namespace Expressions
} // namespace Containers
} // namespace TNL
