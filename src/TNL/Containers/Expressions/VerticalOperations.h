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

template< typename Expression >
__cuda_callable__
auto StaticExpressionMin( const Expression& expression ) -> typename std::remove_reference< decltype( expression[ 0 ] ) >::type
{
   auto aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux = TNL::min( aux, expression[ i ] );
   return aux;
}

template< typename Expression >
__cuda_callable__
auto StaticExpressionMax( const Expression& expression ) -> typename std::remove_reference< decltype( expression[ 0 ] ) >::type
{
   auto aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux = TNL::max( aux, expression[ i ] );
   return aux;
}

template< typename Expression >
__cuda_callable__
auto StaticExpressionSum( const Expression& expression ) -> typename std::remove_reference< decltype( expression[ 0 ] ) >::type
{
   auto aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux += expression[ i ];
   return aux;
}

template< typename Expression, typename Real >
__cuda_callable__
auto StaticExpressionLpNorm( const Expression& expression, const Real& p ) -> typename std::remove_reference< decltype( expression[ 0 ] ) >::type
{
   if( p == ( Real ) 1.0 )
   {
      auto aux = TNL::abs( expression[ 0 ] );
      for( int i = 1; i < expression.getSize(); i++ )
         aux += TNL::abs( expression[ i ] );
      return aux;
   }
   if( p == ( Real ) 2.0 )
   {
      auto aux = expression[ 0 ] * expression[ 0 ];
      for( int i = 1; i < expression.getSize(); i++ )
         aux += expression[ i ] * expression[ i ];
      return TNL::sqrt( aux );
   }
   auto aux = TNL::pow( expression[ 0 ], p );
   for( int i = 1; i < expression.getSize(); i++ )
      aux += TNL::pow( expression[ i ], p );
   return TNL::pow( aux, 1.0 / p );
}


template< typename Expression >
__cuda_callable__
auto StaticExpressionProduct( const Expression& expression ) -> typename std::remove_reference< decltype( expression[ 0 ] ) >::type
{
   auto aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux *= expression[ i ];
   return aux;
}

template< typename Expression >
__cuda_callable__
bool StaticExpressionLogicalAnd( const Expression& expression )
{
   auto aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux = aux && expression[ i ];
   return aux;
}

template< typename Expression >
__cuda_callable__
bool StaticExpressionLogicalOr( const Expression& expression )
{
   auto aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux = aux || expression[ i ];
   return aux;
}

template< typename Expression >
__cuda_callable__
auto StaticExpressionBinaryAnd( const Expression& expression ) -> typename std::remove_reference< decltype( expression[ 0 ] ) >::type
{
   auto aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux = aux & expression[ i ];
   return aux;
}

template< typename Expression >
__cuda_callable__
auto StaticExpressionBinaryOr( const Expression& expression ) -> typename std::remove_reference< decltype( expression[ 0 ] ) >::type
{
   auto aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux = aux | expression[ i ];
   return aux;
}

////
// Non-static operations
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
auto ExpressionMax( const Expression& expression ) -> typename std::remove_reference< decltype( expression[ 0 ] ) >::type
{
   using ResultType = typename std::remove_cv< typename std::remove_reference< decltype( expression[ 0 ] ) >::type >::type;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  expression[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a = a > b ? a : b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a = a > b ? a : b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, std::numeric_limits< ResultType >::min() );
}

template< typename Expression >
auto ExpressionSum( const Expression& expression ) -> typename std::remove_reference< decltype( expression[ 0 ] ) >::type
{
   using ResultType = typename std::remove_cv< typename std::remove_reference< decltype( expression[ 0 ] ) >::type >::type;
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  expression[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a += b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a += b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, ( ResultType ) 0.0 );
}

template< typename Expression, typename Real >
auto ExpressionLpNorm( const Expression& expression, const Real& p ) -> typename std::remove_reference< decltype( expression[ 0 ] ) >::type
{
   using ResultType = typename std::remove_cv< typename std::remove_reference< decltype( expression[ 0 ] ) >::type >::type;
   using IndexType = typename Expression::IndexType;

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
      return TNL::sqrt( Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, ( ResultType ) 0.0 ) );
   }
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  TNL::pow( expression[ i ], p ); };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a += b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a += b; };
   return TNL::pow( Algorithms::Reduction< typename Expression::DeviceType >::reduce( expression.getSize(), reduction, volatileReduction, fetch, ( ResultType ) 0.0 ), ( Real ) 1.0 / p );
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
