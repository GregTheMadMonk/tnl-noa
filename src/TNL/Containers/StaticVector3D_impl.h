/***************************************************************************
                          StaticVector3D_impl.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/StaticVectorExpressions.h>

namespace TNL {
namespace Containers {

template< typename Real >
__cuda_callable__
StaticVector< 3, Real >::StaticVector()
{
}

template< typename Real >
   template< typename _unused >
__cuda_callable__
StaticVector< 3, Real >::StaticVector( const Real v[ 3 ] )
: StaticArray< 3, Real >( v )
{
}

template< typename Real >
__cuda_callable__
StaticVector< 3, Real >::StaticVector( const Real& v )
: StaticArray< 3, Real >( v )
{
}

template< typename Real >
__cuda_callable__
StaticVector< 3, Real >::StaticVector( const Real& v1, const Real& v2, const Real& v3 )
: StaticArray< 3, Real >( v1, v2, v3 )
{
}

template< typename Real >
__cuda_callable__
StaticVector< 3, Real >::StaticVector( const StaticVector< 3, Real >& v )
: StaticArray< 3, Real >( v )
{
}

template< typename Real >
StaticVector< 3, Real >::StaticVector( const std::initializer_list< Real > &elems )
: StaticArray< 3, Real >( elems )
{
}

template< typename Real >
   template< typename T1,
             typename T2,
             template< typename, typename > class Operation >
StaticVector< 3, Real >::StaticVector( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& op )
{
   Algorithms::VectorAssignment< StaticVector< 3, Real >, Expressions::BinaryExpressionTemplate< T1, T2, Operation > >::assign( *this, op );
};


template< typename Real >
bool
StaticVector< 3, Real >::setup( const Config::ParameterContainer& parameters,
                                const String& prefix )
{
   this->data[ 0 ] = parameters.getParameter< double >( prefix + "0" );
   this->data[ 1 ] = parameters.getParameter< double >( prefix + "1" );
   this->data[ 2 ] = parameters.getParameter< double >( prefix + "2" );
   return true;
}

template< typename Real >
String StaticVector< 3, Real >::getType()
{
   return String( "Containers::StaticVector< " ) +
          convertToString( 3 ) +
          String( ", " ) +
          TNL::getType< Real >() +
          String( " >" );
}

template< typename Real >
   template< typename RHS >
StaticVector< 3, Real >&
StaticVector< 3, Real >::operator =( const RHS& rhs )
{
   Algorithms::VectorAssignment< StaticVector< 3, Real >, RHS >::assign( *this, rhs );
   return *this;
}

template< typename Real >
__cuda_callable__
StaticVector< 3, Real >& StaticVector< 3, Real >::operator += ( const StaticVector& v )
{
   this->data[ 0 ] += v[ 0 ];
   this->data[ 1 ] += v[ 1 ];
   this->data[ 2 ] += v[ 2 ];
   return *this;
}

template< typename Real >
__cuda_callable__
StaticVector< 3, Real >& StaticVector< 3, Real >::operator -= ( const StaticVector& v )
{
   this->data[ 0 ] -= v[ 0 ];
   this->data[ 1 ] -= v[ 1 ];
   this->data[ 2 ] -= v[ 2 ];
   return *this;
}

template< typename Real >
__cuda_callable__
StaticVector< 3, Real >& StaticVector< 3, Real >::operator *= ( const Real& c )
{
   this->data[ 0 ] *= c;
   this->data[ 1 ] *= c;
   this->data[ 2 ] *= c;
   return *this;
}

template< typename Real >
__cuda_callable__
StaticVector< 3, Real >& StaticVector< 3, Real >::operator /= ( const Real& c )
{
   const RealType d = 1.0 / c;
   this->data[ 0 ] *= d;
   this->data[ 1 ] *= d;
   this->data[ 2 ] *= d;
   return *this;
}

#ifdef UNDEF
template< typename Real >
__cuda_callable__
StaticVector< 3, Real > StaticVector< 3, Real >::operator + ( const StaticVector& u ) const
{
   StaticVector< 3, Real > res;
   res[ 0 ] = this->data[ 0 ] + u[ 0 ];
   res[ 1 ] = this->data[ 1 ] + u[ 1 ];
   res[ 2 ] = this->data[ 2 ] + u[ 2 ];
   return res;
}

template< typename Real >
__cuda_callable__
StaticVector< 3, Real > StaticVector< 3, Real >::operator - ( const StaticVector& u ) const
{
   StaticVector< 3, Real > res;
   res[ 0 ] = this->data[ 0 ] - u[ 0 ];
   res[ 1 ] = this->data[ 1 ] - u[ 1 ];
   res[ 2 ] = this->data[ 2 ] - u[ 2 ];
   return res;
}

template< typename Real >
__cuda_callable__
StaticVector< 3, Real > StaticVector< 3, Real >::operator * ( const Real& c ) const
{
   StaticVector< 3, Real > res;
   res[ 0 ] = c * this->data[ 0 ];
   res[ 1 ] = c * this->data[ 1 ];
   res[ 2 ] = c * this->data[ 2 ];
   return res;
}

template< typename Real >
__cuda_callable__
Real StaticVector< 3, Real >::operator * ( const StaticVector& u ) const
{
   return this->data[ 0 ] * u[ 0 ] +
          this->data[ 1 ] * u[ 1 ] +
          this->data[ 2 ] * u[ 2 ];
}
#endif


template< typename Real >
__cuda_callable__
bool StaticVector< 3, Real >::operator < ( const StaticVector& v ) const
{
   return ( this->data[ 0 ] < v[ 0 ] &&
            this->data[ 1 ] < v[ 1 ] &&
            this->data[ 2 ] < v[ 2 ] );
}

template< typename Real >
__cuda_callable__
bool StaticVector< 3, Real >::operator <= ( const StaticVector& v ) const
{
   return ( this->data[ 0 ] <= v[ 0 ] &&
            this->data[ 1 ] <= v[ 1 ] &&
            this->data[ 2 ] <= v[ 2 ] );
}

template< typename Real >
__cuda_callable__
bool StaticVector< 3, Real >::operator > ( const StaticVector& v ) const
{
   return ( this->data[ 0 ] > v[ 0 ] &&
            this->data[ 1 ] > v[ 1 ] &&
            this->data[ 2 ] > v[ 2 ] );
}

template< typename Real >
__cuda_callable__
bool StaticVector< 3, Real >::operator >= ( const StaticVector& v ) const
{
   return ( this->data[ 0 ] >= v[ 0 ] &&
            this->data[ 1 ] >= v[ 1 ] &&
            this->data[ 2 ] >= v[ 2 ] );
}

template< typename Real >
   template< typename OtherReal >
__cuda_callable__
StaticVector< 3, Real >::
operator StaticVector< 3, OtherReal >() const
{
   StaticVector< 3, OtherReal > aux;
   aux[ 0 ] = this->data[ 0 ];
   aux[ 1 ] = this->data[ 1 ];
   aux[ 2 ] = this->data[ 2 ];
   return aux;
}

template< typename Real >
__cuda_callable__
StaticVector< 3, Real >
StaticVector< 3, Real >::abs() const
{
   return StaticVector< 3, Real >( TNL::abs( this->data[ 0 ] ),
                                   TNL::abs( this->data[ 1 ] ),
                                   TNL::abs( this->data[ 2 ] ) );
}

template< typename Real >
__cuda_callable__
Real
StaticVector< 3, Real >::lpNorm( const Real& p ) const
{
   if( p == 1.0 )
      return TNL::abs( this->data[ 0 ] ) +
             TNL::abs( this->data[ 1 ] ) +
             TNL::abs( this->data[ 2 ] );
   if( p == 2.0 )
      return TNL::sqrt( this->data[ 0 ] * this->data[ 0 ] +
                        this->data[ 1 ] * this->data[ 1 ] +
                        this->data[ 2 ] * this->data[ 2 ] );
   return TNL::pow( TNL::pow( TNL::abs( this->data[ 0 ] ), p ) +
                    TNL::pow( TNL::abs( this->data[ 1 ] ), p ) +
                    TNL::pow( TNL::abs( this->data[ 2 ] ), p ), 1.0 / p );
}

} // namespace Containers
} // namespace TNL
