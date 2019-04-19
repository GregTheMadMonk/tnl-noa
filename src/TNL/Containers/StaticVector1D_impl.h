/***************************************************************************
                          StaticVector1D_impl.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/Algorithms/VectorAssignment.h>
#include <TNL/Containers/StaticVectorExpressions.h>

namespace TNL {
namespace Containers {

template< typename Real >
__cuda_callable__
StaticVector< 1, Real >::StaticVector()
{
}

template< typename Real >
   template< typename _unused >
__cuda_callable__
StaticVector< 1, Real >::StaticVector( const Real v[ 1 ] )
: StaticArray< 1, Real >( v )
{
}

template< typename Real >
__cuda_callable__
StaticVector< 1, Real >::StaticVector( const Real& v )
: StaticArray< 1, Real >( v )
{
}

template< typename Real >
__cuda_callable__
StaticVector< 1, Real >::StaticVector( const StaticVector< 1, Real >& v )
: StaticArray< 1, Real >( v )
{
}

template< typename Real >
StaticVector< 1, Real >::StaticVector( const std::initializer_list< Real > &elems )
: StaticArray< 1, Real >( elems )
{
}

template< typename Real >
   template< typename T1,
             typename T2,
             template< typename, typename > class Operation >
StaticVector< 1, Real >::StaticVector( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& op )
{
   Algorithms::VectorAssignment< StaticVector< 1, Real >, Expressions::BinaryExpressionTemplate< T1, T2, Operation > >::assign( *this, op );
};

template< typename Real >
bool
StaticVector< 1, Real >::setup( const Config::ParameterContainer& parameters,
                                const String& prefix )
{
   this->data[ 0 ] = parameters.getParameter< double >( prefix + "0" );
   return true;
}

template< typename Real >
String StaticVector< 1, Real >::getType()
{
   return String( "Containers::StaticVector< " ) +
          convertToString( 1 ) +
          String( ", " ) +
          TNL::getType< Real >() +
          String( " >" );
}

template< typename Real >
   template< typename RHS >
StaticVector< 1, Real >&
StaticVector< 1, Real >::operator =( const RHS& rhs )
{
   Algorithms::VectorAssignment< StaticVector< 1, Real >, RHS >::assign( *this, rhs );
   return *this;
}

template< typename Real >
__cuda_callable__
StaticVector< 1, Real >& StaticVector< 1, Real >::operator += ( const StaticVector& v )
{
   this->data[ 0 ] += v[ 0 ];
   return *this;
}

template< typename Real >
__cuda_callable__
StaticVector< 1, Real >& StaticVector< 1, Real >::operator -= ( const StaticVector& v )
{
   this->data[ 0 ] -= v[ 0 ];
   return *this;
}

template< typename Real >
__cuda_callable__
StaticVector< 1, Real >& StaticVector< 1, Real >::operator *= ( const Real& c )
{
   this->data[ 0 ] *= c;
   return *this;
}

template< typename Real >
__cuda_callable__
StaticVector< 1, Real >& StaticVector< 1, Real >::operator /= ( const Real& c )
{
   this->data[ 0 ] /= c;
   return *this;
}

#ifdef UNDEF
template< typename Real >
__cuda_callable__
StaticVector< 1, Real > StaticVector< 1, Real >::operator + ( const StaticVector& u ) const
{
   StaticVector< 1, Real > res;
   res[ 0 ] = this->data[ 0 ] + u[ 0 ];
   return res;
}

template< typename Real >
__cuda_callable__
StaticVector< 1, Real > StaticVector< 1, Real >::operator - ( const StaticVector& u ) const
{
   StaticVector< 1, Real > res;
   res[ 0 ] = this->data[ 0 ] - u[ 0 ];
   return res;
}

template< typename Real >
__cuda_callable__
StaticVector< 1, Real > StaticVector< 1, Real >::operator * ( const Real& c ) const
{
   StaticVector< 1, Real > res;
   res[ 0 ] = c * this->data[ 0 ];
   return res;
}

template< typename Real >
__cuda_callable__
Real StaticVector< 1, Real >::operator * ( const StaticVector& u ) const
{
   return this->data[ 0 ] * u[ 0 ];
}
#endif

template< typename Real >
__cuda_callable__
bool StaticVector< 1, Real >::operator < ( const StaticVector& v ) const
{
   return ( this->data[ 0 ] < v[ 0 ] );
}

template< typename Real >
__cuda_callable__
bool StaticVector< 1, Real >::operator <= ( const StaticVector& v ) const
{
   return ( this->data[ 0 ] <= v[ 0 ] );
}

template< typename Real >
__cuda_callable__
bool StaticVector< 1, Real >::operator > ( const StaticVector& v ) const
{
   return ( this->data[ 0 ] > v[ 0 ] );
}

template< typename Real >
__cuda_callable__
bool StaticVector< 1, Real >::operator >= ( const StaticVector& v ) const
{
   return ( this->data[ 0 ] >= v[ 0 ] );
}

template< typename Real >
   template< typename OtherReal >
__cuda_callable__
StaticVector< 1, Real >::
operator StaticVector< 1, OtherReal >() const
{
   StaticVector< 1, OtherReal > aux;
   aux[ 0 ] = this->data[ 0 ];
   return aux;
}

template< typename Real >
__cuda_callable__
StaticVector< 1, Real >
StaticVector< 1, Real >::abs() const
{
   return StaticVector< 1, Real >( TNL::abs( this->data[ 0 ] ) );
}

template< typename Real >
__cuda_callable__
Real
StaticVector< 1, Real >::lpNorm( const Real& p ) const
{
   return TNL::abs( this->data[ 0 ] );
}

} // namespace Containers
} // namespace TNL
