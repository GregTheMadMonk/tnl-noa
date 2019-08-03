/***************************************************************************
                          StaticVerticalOperations.h  -  description
                             -------------------
    begin                : Jul 3, 2019
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
auto StaticExpressionMin( const Expression& expression )
{
   auto aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux = TNL::min( aux, expression[ i ] );
   return aux;
}

template< typename Expression, typename Real >
__cuda_callable__
auto StaticExpressionArgMin( const Expression& expression, int& arg )
{
   auto value = expression[ 0 ];
   arg = 0;
   for( int i = 1; i < expression.getSize(); i++ )
   {
      if( expression[ i ] < value )
      {
         value = expression[ i ];
         arg = i;
      }
   }
   return value;
}

template< typename Expression >
__cuda_callable__
auto StaticExpressionMax( const Expression& expression )
{
   auto aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux = TNL::max( aux, expression[ i ] );
   return aux;
}

template< typename Expression, typename Real >
__cuda_callable__
auto StaticExpressionArgMax( const Expression& expression, int& arg )
{
   auto value = expression[ 0 ];
   arg = 0;
   for( int i = 1; i < expression.getSize(); i++ )
   {
      if( expression[ i ] > value )
      {
         value = expression[ i ];
         arg = i;
      }
   }
   return value;
}

template< typename Expression >
__cuda_callable__
auto StaticExpressionSum( const Expression& expression )
{
   auto aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux += expression[ i ];
   return aux;
}

template< typename Expression, typename Real >
__cuda_callable__
auto StaticExpressionLpNorm( const Expression& expression, const Real& p )
-> typename std::remove_reference< decltype( expression[ 0 ] ) >::type
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
      return aux;
   }
   auto aux = TNL::pow( expression[ 0 ], p );
   for( int i = 1; i < expression.getSize(); i++ )
      aux += TNL::pow( TNL::abs( expression[ i ] ), p );
   return aux;
}


template< typename Expression >
__cuda_callable__
auto StaticExpressionProduct( const Expression& expression )
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
auto StaticExpressionBinaryAnd( const Expression& expression )
{
   auto aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux = aux & expression[ i ];
   return aux;
}

template< typename Expression >
__cuda_callable__
auto StaticExpressionBinaryOr( const Expression& expression )
{
   auto aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux = aux | expression[ i ];
   return aux;
}

} // namespace Expressions
} // namespace Containers
} // namespace TNL
