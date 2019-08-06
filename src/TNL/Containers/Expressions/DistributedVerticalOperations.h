/***************************************************************************
                          DistributedVerticalOperations.h  -  description
                             -------------------
    begin                : Aug 6, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Expressions/VerticalOperations.h>
#include <TNL/Communicators/MpiDefs.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Containers {
namespace Expressions {

////
// Vertical operations
template< typename Expression >
auto DistributedExpressionMin( const Expression& expression ) -> std::remove_reference_t< decltype( expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] ) >;
   using CommunicatorType = typename Expression::CommunicatorType;

   ResultType result = std::numeric_limits< ResultType >::max();
   if( expression.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const ResultType localResult = ExpressionMin( expression.getConstLocalView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MIN, expression.getCommunicationGroup() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionArgMin( const Expression& expression, typename Expression::IndexType& arg ) -> std::remove_reference_t< decltype( expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] ) >;
   throw Exceptions::NotImplementedError("DistributedExpressionArgMin is not implemented yet");
}

template< typename Expression >
auto DistributedExpressionMax( const Expression& expression ) -> std::remove_reference_t< decltype( expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] ) >;
   using CommunicatorType = typename Expression::CommunicatorType;

   ResultType result = std::numeric_limits< ResultType >::min();
   if( expression.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const ResultType localResult = ExpressionMax( expression.getConstLocalView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_MAX, expression.getCommunicationGroup() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionArgMax( const Expression& expression, typename Expression::IndexType& arg ) -> std::remove_reference_t< decltype( expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] ) >;
   throw Exceptions::NotImplementedError("DistributedExpressionArgMax is not implemented yet");
}

template< typename Expression >
auto DistributedExpressionSum( const Expression& expression ) ->
   typename std::conditional<
      std::is_same< std::decay_t< decltype( expression[0] ) >, bool >::value,
      typename Expression::IndexType,
      std::remove_reference_t< decltype( expression[0] ) >
   >::type
{
   using ResultTypeBase = std::decay_t< decltype( expression[0] ) >;
   using IndexType = typename Expression::IndexType;
   using ResultType = typename std::conditional< std::is_same< ResultTypeBase, bool >::value, IndexType, ResultTypeBase >::type;
   using CommunicatorType = typename Expression::CommunicatorType;

   ResultType result = 0;
   if( expression.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const ResultType localResult = ExpressionSum( expression.getConstLocalView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, expression.getCommunicationGroup() );
   }
   return result;
}

template< typename Expression, typename Real >
auto DistributedExpressionLpNorm( const Expression& expression, const Real& p ) ->
   typename std::conditional<
      std::is_same< std::decay_t< decltype( expression[0] ) >, bool >::value,
      double, // TODO: Solve this some better way
      std::remove_reference_t< decltype( expression[0] ) >
   >::type
{
   using ResultTypeBase = std::decay_t< decltype( expression[0] ) >;
   using IndexType = typename Expression::IndexType;
   using ResultType = typename std::conditional< std::is_same< ResultTypeBase, bool >::value, double, ResultTypeBase >::type;
   using CommunicatorType = typename Expression::CommunicatorType;

   ResultType result = 0;
   if( expression.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const ResultType localResult = ExpressionLpNorm( expression.getConstLocalView(), p );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_SUM, expression.getCommunicationGroup() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionProduct( const Expression& expression ) -> std::remove_reference_t< decltype( expression[0] * expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] ) >;
   using CommunicatorType = typename Expression::CommunicatorType;

   ResultType result = 1;
   if( expression.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const ResultType localResult = ExpressionProduct( expression.getConstLocalView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_PROD, expression.getCommunicationGroup() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionLogicalAnd( const Expression& expression ) -> std::remove_reference_t< decltype( expression[0] && expression[0] ) >
{
   using ResultType = std::remove_reference_t< decltype( expression[0] && expression[0] ) >;
   using CommunicatorType = typename Expression::CommunicatorType;

   ResultType result = std::numeric_limits< ResultType >::max();
   if( expression.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const ResultType localResult = ExpressionLogicalAnd( expression.getConstLocalView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LAND, expression.getCommunicationGroup() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionLogicalOr( const Expression& expression ) -> std::remove_reference_t< decltype( expression[0] || expression[0] ) >
{
   using ResultType = std::remove_reference_t< decltype( expression[0] || expression[0] ) >;
   using CommunicatorType = typename Expression::CommunicatorType;

   ResultType result = 0;
   if( expression.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const ResultType localResult = ExpressionLogicalOr( expression.getConstLocalView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_LOR, expression.getCommunicationGroup() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionBinaryAnd( const Expression& expression ) -> std::remove_reference_t< decltype( expression[0] | expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] & expression[0] ) >;
   using CommunicatorType = typename Expression::CommunicatorType;

   ResultType result = std::numeric_limits< ResultType >::max();
   if( expression.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const ResultType localResult = ExpressionLogicalBinaryAnd( expression.getConstLocalView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_BAND, expression.getCommunicationGroup() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionBinaryOr( const Expression& expression ) -> std::remove_reference_t< decltype( expression[0] | expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] | expression[0] ) >;
   using CommunicatorType = typename Expression::CommunicatorType;

   ResultType result = 0;
   if( expression.getCommunicationGroup() != CommunicatorType::NullGroup ) {
      const ResultType localResult = ExpressionBinaryOr( expression.getConstLocalView() );
      CommunicatorType::Allreduce( &localResult, &result, 1, MPI_BOR, expression.getCommunicationGroup() );
   }
   return result;
}

} // namespace Expressions
} // namespace Containers
} // namespace TNL
