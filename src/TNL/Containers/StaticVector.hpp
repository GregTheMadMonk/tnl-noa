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

namespace Detail {

////
// Lambdas used together with StaticFor for static loop unrolling in the
// implementation of the StaticVector
template< typename LeftReal, typename RightReal = LeftReal >
auto addVectorLambda = [] __cuda_callable__ ( int i, LeftReal* data, const RightReal* v ) { data[ i ] += v[ i ]; };

template< typename LeftReal, typename RightReal = LeftReal >
auto subtractVectorLambda = [] __cuda_callable__ ( int i, LeftReal* data, const RightReal* v ) { data[ i ] -= v[ i ]; };

template< typename LeftReal, typename RightReal = LeftReal >
auto scalarMultiplicationLambda = [] __cuda_callable__ ( int i, LeftReal* data, const RightReal v ) { data[ i ] *= v; };

} //namespace Detail

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >::StaticVector()
{
}

template< int Size, typename Real >
   template< typename _unused >
__cuda_callable__
StaticVector< Size, Real >::StaticVector( const Real v[ Size ] )
: StaticArray< Size, Real >( v )
{
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >::StaticVector( const Real& v )
: StaticArray< Size, Real >( v )
{
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >::StaticVector( const StaticVector< Size, Real >& v )
: StaticArray< Size, Real >( v )
{
}

template< int Size, typename Real >
StaticVector< Size, Real >::StaticVector( const std::initializer_list< Real > &elems )
: StaticArray< Size, Real >( elems )
{
}

template< int Size, typename Real >
 __cuda_callable__
StaticVector< Size, Real >::StaticVector( const Real& v1, const Real& v2 )
: StaticArray< Size, Real >( v1, v2 )
{
}

template< int Size, typename Real >
 __cuda_callable__
StaticVector< Size, Real >::StaticVector( const Real& v1, const Real& v2, const Real& v3 )
: StaticArray< Size, Real >( v1, v2, v3 )
{
}

template< int Size, typename Real >
   template< typename T1,
             typename T2,
             template< typename, typename > class Operation >
StaticVector< Size, Real >::StaticVector( const Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& op )
{
   Algorithms::VectorAssignment< StaticVector< Size, Real >, Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation > >::assignStatic( *this, op );
};

template< int Size,
          typename Real >
   template< typename T,
             template< typename > class Operation >
__cuda_callable__
StaticVector< Size, Real >::StaticVector( const Expressions::StaticUnaryExpressionTemplate< T, Operation >& op )
{
   Algorithms::VectorAssignment< StaticVector< Size, Real >, Expressions::StaticUnaryExpressionTemplate< T, Operation > >::assignStatic( *this, op );
};

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
StaticVector< Size, Real >::operator =( const RHS& rhs )
{
   Algorithms::VectorAssignment< StaticVector< Size, Real >, RHS >::assignStatic( *this, rhs );
   return *this;
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator += ( const StaticVector& v )
{
   StaticFor< 0, Size >::exec( Detail::addVectorLambda< Real >, this->getData(), v.getData() );
   return *this;
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator -= ( const StaticVector& v )
{
   StaticFor< 0, Size >::exec( Detail::subtractVectorLambda< Real >, this->getData(), v.getData() );
   return *this;
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator *= ( const Real& c )
{
   StaticFor< 0, Size >::exec( Detail::scalarMultiplicationLambda< Real >, this->getData(), c );
   return *this;
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator /= ( const Real& c )
{
   StaticFor< 0, Size >::exec( Detail::scalarMultiplicationLambda< Real >, this->getData(), 1.0 / c );
   return *this;
}

template< int Size, typename Real >
   template< typename OtherReal >
__cuda_callable__
StaticVector< Size, Real >::
operator StaticVector< Size, OtherReal >() const
{
   StaticVector< Size, OtherReal > aux;
   StaticFor< 0, Size >::exec( Detail::assignArrayLambda< OtherReal, Real >, aux.getData(), this->getData() );
   return aux;
}

} // namespace Containers
} // namespace TNL
