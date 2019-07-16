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
   //for( int i = 0; i < Size; i++ )
   //   this->data[ i ] += v[ i ];
   StaticFor< 0, Size >::exec( Detail::addVectorLambda< Real >, this->getData(), v.getData() );
   return *this;
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator -= ( const StaticVector& v )
{
   //for( int i = 0; i < Size; i++ )
   //   this->data[ i ] -= v[ i ];
   StaticFor< 0, Size >::exec( Detail::subtractVectorLambda< Real >, this->getData(), v.getData() );
   return *this;
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator *= ( const Real& c )
{
   //for( int i = 0; i < Size; i++ )
   //   this->data[ i ] *= c;
   StaticFor< 0, Size >::exec( Detail::scalarMultiplicationLambda< Real >, this->getData(), c );
   return *this;
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator /= ( const Real& c )
{
   //const RealType d = 1.0 / c;
   //for( int i = 0; i < Size; i++ )
   //   this->data[ i ] *= d;
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
   //for( int i = 0; i < Size; i++ )
   //   aux[ i ] = this->data[ i ];
   StaticFor< 0, Size >::exec( Detail::assignArrayLambda< OtherReal, Real >, aux.getData(), this->getData() );
   return aux;
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >
StaticVector< Size, Real >::abs() const
{
   StaticVector< Size, Real > v;
   for( int i = 0; i < Size; i++ )
      v.data[ i ] = TNL::abs( this->data[ i ] );
   return v;
}

template< int Size, typename Real >
__cuda_callable__
Real
StaticVector< Size, Real >::lpNorm( const Real& p ) const
{
   if( p == 1.0 )
   {
      Real aux = TNL::abs( this->data[ 0 ] );
      for( int i = 1; i < Size; i++ )
         aux += TNL::abs( this->data[ i ] );
      return aux;
   }
   if( p == 2.0 )
   {
      Real aux = this->data[ 0 ] * this->data[ 0 ];
      for( int i = 1; i < Size; i++ )
         aux += this->data[ i ] * this->data[ i ];
      return TNL::sqrt( aux );
   }
   Real aux = TNL::pow( TNL::abs( this->data[ 0 ] ), p );
   for( int i = 1; i < Size; i++ )
      aux += TNL::pow( TNL::abs( this->data[ i ] ), p );
   return TNL::pow( aux, 1.0 / p );
}

} // namespace Containers
} // namespace TNL
