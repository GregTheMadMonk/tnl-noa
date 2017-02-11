/***************************************************************************
                          StaticVector1D_impl.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

namespace TNL {
namespace Containers {   

template< typename Real >
__cuda_callable__
StaticVector< 1, Real >::StaticVector()
{
}

template< typename Real >
__cuda_callable__
StaticVector< 1, Real >::StaticVector( const Real& v )
: Containers::StaticArray< 1, Real >( v )
{
}

template< typename Real >
__cuda_callable__
StaticVector< 1, Real >::StaticVector( const StaticVector< 1, Real >& v )
: Containers::StaticArray< 1, Real >( v )
{
}

template< typename Real >
bool
StaticVector< 1, Real >::setup( const Config::ParameterContainer& parameters,
                                const String& prefix )
{
   return parameters.getParameter< double >( prefix + "0", this->data[ 0 ] );
}

template< typename Real >
String StaticVector< 1, Real >::getType()
{
   return String( "Containers::StaticVector< " ) +
          String( 1 ) +
          String( ", " ) +
          TNL::getType< Real >() +
          String( " >" );
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

#ifdef UNDEF //TEMPLATE_EXPLICIT_INSTANTIATION

#ifndef HAVE_CUDA
// TODO: does not work with CUDA
#ifdef INSTANTIATE_FLOAT
extern template class StaticVector< 1, float >;
#endif
extern template class StaticVector< 1, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class StaticVector< 1, long double >;
#endif
#endif

#endif

} // namespace Containers
} // namespace TNL
