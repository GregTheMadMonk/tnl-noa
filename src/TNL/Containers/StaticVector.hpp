/***************************************************************************
                          StaticVector_impl.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/StaticVectorExpressions.h>
#include <TNL/Containers/Algorithms/VectorAssignment.h>

namespace TNL {
namespace Containers {

template< int Size, typename Real >
   template< typename T1,
             typename T2,
             template< typename, typename > class Operation >
StaticVector< Size, Real >::StaticVector( const Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& op )
{
   Algorithms::VectorAssignment< StaticVector< Size, Real >, Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation > >::assignStatic( *this, op );
}

template< int Size,
          typename Real >
   template< typename T,
             template< typename > class Operation >
__cuda_callable__
StaticVector< Size, Real >::StaticVector( const Expressions::StaticUnaryExpressionTemplate< T, Operation >& op )
{
   Algorithms::VectorAssignment< StaticVector< Size, Real >, Expressions::StaticUnaryExpressionTemplate< T, Operation > >::assignStatic( *this, op );
}

template< int Size, typename Real >
bool
StaticVector< Size, Real >::setup( const Config::ParameterContainer& parameters,
                                   const String& prefix )
{
   for( int i = 0; i < Size; i++ )
   {
      double aux;
      if( ! parameters.template getParameter< double >( prefix + convertToString( i ), aux ) )
         return false;
      this->data[ i ] = aux;
   }
   return true;
}

template< int Size, typename Real >
String StaticVector< Size, Real >::getType()
{
   return String( "Containers::StaticVector< " ) +
          convertToString( Size ) +
          String( ", " ) +
          TNL::getType< Real >() +
          String( " >" );
}

template< int Size, typename Real >
   template< typename VectorExpression >
StaticVector< Size, Real >&
StaticVector< Size, Real >::operator=( const VectorExpression& expression )
{
   Algorithms::VectorAssignment< StaticVector< Size, Real >, VectorExpression >::assignStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
   template< typename VectorExpression >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator+=( const VectorExpression& expression )
{
   Algorithms::VectorAssignmentWithOperation< StaticVector, VectorExpression >::additionStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
   template< typename VectorExpression >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator-=( const VectorExpression& expression )
{
   Algorithms::VectorAssignmentWithOperation< StaticVector, VectorExpression >::subtractionStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
   template< typename VectorExpression >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator*=( const VectorExpression& expression )
{
   Algorithms::VectorAssignmentWithOperation< StaticVector, VectorExpression >::multiplicationStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
   template< typename VectorExpression >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator/=( const VectorExpression& expression )
{
   Algorithms::VectorAssignmentWithOperation< StaticVector, VectorExpression >::divisionStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
   template< typename OtherReal >
__cuda_callable__
StaticVector< Size, Real >::
operator StaticVector< Size, OtherReal >() const
{
   StaticVector< Size, OtherReal > aux;
   StaticFor< 0, Size >::exec( detail::assignArrayFunctor< OtherReal, Real >{}, aux.getData(), this->getData() );
   return aux;
}

} // namespace Containers
} // namespace TNL
