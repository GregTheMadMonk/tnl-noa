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

namespace detail {

////
// Functors used together with StaticFor for static loop unrolling in the
// implementation of the StaticVector
template< typename LeftReal, typename RightReal = LeftReal >
struct addVectorFunctor
{
   void __cuda_callable__ operator()( int i, LeftReal* data, const RightReal* v ) const
   {
      data[ i ] += v[ i ];
   }
};

template< typename LeftReal, typename RightReal = LeftReal >
struct subtractVectorFunctor
{
   void __cuda_callable__ operator()( int i, LeftReal* data, const RightReal* v ) const
   {
      data[ i ] -= v[ i ];
   }
};

template< typename LeftReal, typename RightReal = LeftReal >
struct scalarMultiplicationFunctor
{
   void __cuda_callable__ operator()( int i, LeftReal* data, const RightReal v ) const
   {
      data[ i ] *= v;
   }
};

} // namespace detail

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
   template< typename RHS >
StaticVector< Size, Real >&
StaticVector< Size, Real >::operator=( const RHS& rhs )
{
   Algorithms::VectorAssignment< StaticVector< Size, Real >, RHS >::assignStatic( *this, rhs );
   return *this;
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator+=( const StaticVector& v )
{
   StaticFor< 0, Size >::exec( detail::addVectorFunctor< Real >{}, this->getData(), v.getData() );
   return *this;
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator-=( const StaticVector& v )
{
   StaticFor< 0, Size >::exec( detail::subtractVectorFunctor< Real >{}, this->getData(), v.getData() );
   return *this;
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator*=( const Real& c )
{
   StaticFor< 0, Size >::exec( detail::scalarMultiplicationFunctor< Real >{}, this->getData(), c );
   return *this;
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator/=( const Real& c )
{
   StaticFor< 0, Size >::exec( detail::scalarMultiplicationFunctor< Real >{}, this->getData(), 1.0 / c );
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
