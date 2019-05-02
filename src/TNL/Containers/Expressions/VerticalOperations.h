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

template< typename T >
__cuda_callable__
auto StaticExpressionMin( const T& a ) -> decltype( a[ 0 ] )
{
   auto aux = a[ 0 ];
   for( int i = 1; i < a.getSize(); i++ )
      aux = TNL::min( aux, a[ i ] );
   return aux;
}

template< typename T >
__cuda_callable__
auto StaticExpressionMax( const T& a ) -> decltype( a[ 0 ] )
{
   auto aux = a[ 0 ];
   for( int i = 1; i < a.getSize(); i++ )
      aux = TNL::max( aux, a[ i ] );
   return aux;
}

template< typename T >
__cuda_callable__
auto StaticExpressionSum( const T& a ) -> decltype( a[ 0 ] )
{
   auto aux = a[ 0 ];
   for( int i = 1; i < a.getSize(); i++ )
      aux += a[ i ];
   return aux;
}

template< typename T >
__cuda_callable__
auto StaticExpressionProduct( const T& a ) -> decltype( a[ 0 ] )
{
   auto aux = a[ 0 ];
   for( int i = 1; i < a.getSize(); i++ )
      aux *= a[ i ];
   return aux;
}

template< typename T >
__cuda_callable__
bool StaticExpressionLogicalAnd( const T& a )
{
   auto aux = a[ 0 ];
   for( int i = 1; i < a.getSize(); i++ )
      aux = aux && a[ i ];
   return aux;
}

template< typename T >
__cuda_callable__
bool StaticExpressionLogicalOr( const T& a )
{
   auto aux = a[ 0 ];
   for( int i = 1; i < a.getSize(); i++ )
      aux = aux || a[ i ];
   return aux;
}

template< typename T >
__cuda_callable__
auto StaticExpressionBinaryAnd( const T& a ) -> decltype( a[ 0 ] )
{
   auto aux = a[ 0 ];
   for( int i = 1; i < a.getSize(); i++ )
      aux = aux & a[ i ];
   return aux;
}

template< typename T >
__cuda_callable__
auto StaticExpressionBinaryOr( const T& a ) -> decltype( a[ 0 ] )
{
   auto aux = a[ 0 ];
   for( int i = 1; i < a.getSize(); i++ )
      aux = aux | a[ i ];
   return aux;
}

////
// Non-static operations
template< typename Expression >
__cuda_callable__
auto ExpressionMin( const Expression& a ) -> decltype( a[ 0 ] )
{
   using ResultType = decltype( a[ 0 ] );
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  a[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a = a < b ? a : b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a = a < b ? a : b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, std::numeric_limits< ResultType >::max() );
}

template< typename Expression >
__cuda_callable__
auto ExpressionMax( const Expression& a ) -> decltype( a[ 0 ] )
{
   using ResultType = decltype( a[ 0 ] );
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  a[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a = a > b ? a : b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a = a > b ? a : b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, std::numeric_limits< ResultType >::max() );
}

template< typename Expression >
__cuda_callable__
auto ExpressionSum( const Expression& a ) -> decltype( a[ 0 ] )
{
   using ResultType = decltype( a[ 0 ] );
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  a[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a += b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a += b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, ( ResultType ) 0.0 );
}

template< typename Expression >
__cuda_callable__
auto ExpressionProduct( const Expression& a ) -> decltype( a[ 0 ] )
{
   using ResultType = decltype( a[ 0 ] );
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  a[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a *= b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a *= b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, ( ResultType ) 1.0 );
}

template< typename Expression >
__cuda_callable__
bool ExpressionLogicalAnd( const Expression& a )
{
   using ResultType = decltype( a[ 0 ] );
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  a[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a = a && b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a = a && b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, true );
}

template< typename Expression >
__cuda_callable__
bool ExpressionLogicalOr( const Expression& a )
{
   using ResultType = decltype( a[ 0 ] );
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  a[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a = a || b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a = a || b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch,  );
}

template< typename Expression >
__cuda_callable__
auto ExpressionBinaryAnd( const Expression& a ) -> decltype( a[ 0 ] )
{
   using ResultType = decltype( a[ 0 ] );
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  a[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a = a & b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a = a & b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, std::numeric_limits< ResultType >::max() );
}

template< typename Expression >
__cuda_callable__
auto ExpressionBinaryOr( const Expression& a ) -> decltype( a[ 0 ] )
{
   using ResultType = decltype( a[ 0 ] );
   using IndexType = typename Expression::IndexType;

   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return  a[ i ]; };
   auto reduction = [=] __cuda_callable__ ( ResultType& a, const ResultType& b ) { a = a | b; };
   auto volatileReduction = [=] __cuda_callable__ ( volatile ResultType& a, volatile ResultType& b ) { a = a | b; };
   return Algorithms::Reduction< typename Expression::DeviceType >::reduce( a.getSize(), reduction, volatileReduction, fetch, ( ResultType ) 0 );

}

      } //namespace Expressions
   } // namespace Containers
} // namespace TNL
