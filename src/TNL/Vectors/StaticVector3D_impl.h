/***************************************************************************
                          StaticVector3D_impl.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Vectors {   

template< typename Real >
__cuda_callable__
StaticVector< 3, Real >::StaticVector()
{
}

template< typename Real >
__cuda_callable__
StaticVector< 3, Real >::StaticVector( const Real v[ 3 ] )
: Arrays::tnlStaticArray< 3, Real >( v )
{
}

template< typename Real >
__cuda_callable__
StaticVector< 3, Real >::StaticVector( const Real& v )
: Arrays::tnlStaticArray< 3, Real >( v )
{
}

template< typename Real >
__cuda_callable__
StaticVector< 3, Real >::StaticVector( const Real& v1, const Real& v2, const Real& v3 )
: Arrays::tnlStaticArray< 3, Real >( v1, v2, v3 )
{
}

template< typename Real >
__cuda_callable__
StaticVector< 3, Real >::StaticVector( const StaticVector< 3, Real >& v )
: Arrays::tnlStaticArray< 3, Real >( v )
{
}

template< typename Real >
String StaticVector< 3, Real >::getType()
{
   return String( "StaticVector< " ) +
          String( 3 ) +
          String( ", " ) +
         TNL::getType< Real >() +
          String( " >" );
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
   return StaticVector< 3, Real >( ::abs( this->data[ 0 ] ),
                                      ::abs( this->data[ 1 ] ),
                                      ::abs( this->data[ 2 ] ) );
}


#ifdef UNDEF //TEMPLATE_EXPLICIT_INSTANTIATION

#ifndef HAVE_CUDA
// TODO: does not work with CUDA
#ifdef INSTANTIATE_FLOAT
extern template class StaticVector< 3, float >;
#endif
extern template class StaticVector< 3, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class StaticVector< 3, long double >;
#endif
#endif

#endif

} // namespace Vectors
} // namespace TNL
