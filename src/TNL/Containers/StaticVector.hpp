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
#include <TNL/Containers/detail/VectorAssignment.h>

namespace TNL {
namespace Containers {

template< int Size, typename Real >
   template< typename T1,
             typename T2,
             typename Operation >
__cuda_callable__
StaticVector< Size, Real >::StaticVector( const Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& expr )
{
   detail::VectorAssignment< StaticVector< Size, Real >, Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation > >::assignStatic( *this, expr );
}

template< int Size,
          typename Real >
   template< typename T,
             typename Operation >
__cuda_callable__
StaticVector< Size, Real >::StaticVector( const Expressions::StaticUnaryExpressionTemplate< T, Operation >& expr )
{
   detail::VectorAssignment< StaticVector< Size, Real >, Expressions::StaticUnaryExpressionTemplate< T, Operation > >::assignStatic( *this, expr );
}

template< int Size, typename Real >
bool
StaticVector< Size, Real >::setup( const Config::ParameterContainer& parameters,
                                   const String& prefix )
{
   for( int i = 0; i < Size; i++ )
      this->data[ i ] = parameters.template getParameter< double >( prefix + convertToString( i ) );
   return true;
}

template< int Size, typename Real >
   template< typename VectorExpression >
__cuda_callable__
StaticVector< Size, Real >&
StaticVector< Size, Real >::operator=( const VectorExpression& expression )
{
   detail::VectorAssignment< StaticVector< Size, Real >, VectorExpression >::assignStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
   template< typename VectorExpression >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator+=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< StaticVector, VectorExpression >::additionStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
   template< typename VectorExpression >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator-=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< StaticVector, VectorExpression >::subtractionStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
   template< typename VectorExpression >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator*=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< StaticVector, VectorExpression >::multiplicationStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
   template< typename VectorExpression >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator/=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< StaticVector, VectorExpression >::divisionStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
   template< typename OtherReal >
__cuda_callable__
StaticVector< Size, Real >::
operator StaticVector< Size, OtherReal >() const
{
   StaticVector< Size, OtherReal > aux;
   Algorithms::UnrolledFor< 0, Size >::exec( detail::AssignArrayFunctor{}, aux.getData(), this->getData() );
   return aux;
}

} // namespace Containers
} // namespace TNL
