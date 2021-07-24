/***************************************************************************
                          DistributedVerticalOperations.h  -  description
                             -------------------
    begin                : Aug 6, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/MPI/Wrappers.h>
#include <TNL/Algorithms/reduce.h>

namespace TNL {
namespace Containers {
namespace Expressions {

template< typename Expression >
auto DistributedExpressionMin( const Expression& expression ) -> std::decay_t< decltype( expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] ) >;

   static_assert( std::numeric_limits< ResultType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's result type" );
   ResultType result = std::numeric_limits< ResultType >::max();
   if( expression.getCommunicationGroup() != MPI::NullGroup() ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), TNL::Min{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_MIN, expression.getCommunicationGroup() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionArgMin( const Expression& expression )
-> std::pair< std::decay_t< decltype( expression[0] ) >, typename Expression::IndexType >
{
   using RealType = std::decay_t< decltype( expression[0] ) >;
   using IndexType = typename Expression::IndexType;
   using ResultType = std::pair< RealType, IndexType >;

   static_assert( std::numeric_limits< RealType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's real type" );
   ResultType result( -1, std::numeric_limits< RealType >::max() );
   const auto group = expression.getCommunicationGroup();
   if( group != MPI::NullGroup() ) {
      // compute local argMin
      ResultType localResult = Algorithms::reduceWithArgument( expression.getConstLocalView(), TNL::MinWithArg{} );
      // transform local index to global index
      localResult.second += expression.getLocalRange().getBegin();

      // scatter local result to all processes and gather their results
      const int nproc = MPI::GetSize( group );
      ResultType dataForScatter[ nproc ];
      for( int i = 0; i < nproc; i++ ) dataForScatter[ i ] = localResult;
      ResultType gatheredResults[ nproc ];
      // NOTE: exchanging general data types does not work with MPI
      //MPI::Alltoall( dataForScatter, 1, gatheredResults, 1, group );
      MPI::Alltoall( (char*) dataForScatter, sizeof(ResultType), (char*) gatheredResults, sizeof(ResultType), group );

      // reduce the gathered data
      const auto* _data = gatheredResults;  // workaround for nvcc which does not allow to capture variable-length arrays (even in pure host code!)
      auto fetch = [_data] ( IndexType i ) { return _data[ i ].first; };
      result = Algorithms::reduceWithArgument< Devices::Host >( (IndexType) 0, (IndexType) nproc, fetch, TNL::MinWithArg{} );
      result.second = gatheredResults[ result.second ].second;
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionMax( const Expression& expression ) -> std::decay_t< decltype( expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] ) >;

   static_assert( std::numeric_limits< ResultType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's result type" );
   ResultType result = std::numeric_limits< ResultType >::lowest();
   if( expression.getCommunicationGroup() != MPI::NullGroup() ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), TNL::Max{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_MAX, expression.getCommunicationGroup() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionArgMax( const Expression& expression )
-> std::pair< std::decay_t< decltype( expression[0] ) >, typename Expression::IndexType >
{
   using RealType = std::decay_t< decltype( expression[0] ) >;
   using IndexType = typename Expression::IndexType;
   using ResultType = std::pair< RealType, IndexType >;

   static_assert( std::numeric_limits< RealType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's real type" );
   ResultType result( -1, std::numeric_limits< RealType >::lowest() );
   const auto group = expression.getCommunicationGroup();
   if( group != MPI::NullGroup() ) {
      // compute local argMax
      ResultType localResult = Algorithms::reduceWithArgument( expression.getConstLocalView(), TNL::MaxWithArg{} );
      // transform local index to global index
      localResult.second += expression.getLocalRange().getBegin();

      // scatter local result to all processes and gather their results
      const int nproc = MPI::GetSize( group );
      ResultType dataForScatter[ nproc ];
      for( int i = 0; i < nproc; i++ ) dataForScatter[ i ] = localResult;
      ResultType gatheredResults[ nproc ];
      // NOTE: exchanging general data types does not work with MPI
      //MPI::Alltoall( dataForScatter, 1, gatheredResults, 1, group );
      MPI::Alltoall( (char*) dataForScatter, sizeof(ResultType), (char*) gatheredResults, sizeof(ResultType), group );

      // reduce the gathered data
      const auto* _data = gatheredResults;  // workaround for nvcc which does not allow to capture variable-length arrays (even in pure host code!)
      auto fetch = [_data] ( IndexType i ) { return _data[ i ].first; };
      result = Algorithms::reduceWithArgument< Devices::Host >( ( IndexType ) 0, (IndexType) nproc, fetch, TNL::MaxWithArg{} );
      result.second = gatheredResults[ result.second ].second;
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionSum( const Expression& expression ) -> std::decay_t< decltype( expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] ) >;

   ResultType result = 0;
   if( expression.getCommunicationGroup() != MPI::NullGroup() ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), TNL::Plus{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_SUM, expression.getCommunicationGroup() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionProduct( const Expression& expression ) -> std::decay_t< decltype( expression[0] * expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] ) >;

   ResultType result = 1;
   if( expression.getCommunicationGroup() != MPI::NullGroup() ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), TNL::Multiplies{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_PROD, expression.getCommunicationGroup() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionLogicalAnd( const Expression& expression ) -> std::decay_t< decltype( expression[0] && expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] && expression[0] ) >;

   static_assert( std::numeric_limits< ResultType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's result type" );
   ResultType result = std::numeric_limits< ResultType >::max();
   if( expression.getCommunicationGroup() != MPI::NullGroup() ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), TNL::LogicalAnd{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_LAND, expression.getCommunicationGroup() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionLogicalOr( const Expression& expression ) -> std::decay_t< decltype( expression[0] || expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] || expression[0] ) >;

   ResultType result = 0;
   if( expression.getCommunicationGroup() != MPI::NullGroup() ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), TNL::LogicalOr{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_LOR, expression.getCommunicationGroup() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionBinaryAnd( const Expression& expression ) -> std::decay_t< decltype( expression[0] | expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] & expression[0] ) >;

   static_assert( std::numeric_limits< ResultType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's result type" );
   ResultType result = std::numeric_limits< ResultType >::max();
   if( expression.getCommunicationGroup() != MPI::NullGroup() ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), TNL::BitAnd{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_BAND, expression.getCommunicationGroup() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionBinaryOr( const Expression& expression ) -> std::decay_t< decltype( expression[0] | expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] | expression[0] ) >;

   ResultType result = 0;
   if( expression.getCommunicationGroup() != MPI::NullGroup() ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), TNL::BitOr{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_BOR, expression.getCommunicationGroup() );
   }
   return result;
}

} // namespace Expressions
} // namespace Containers
} // namespace TNL
