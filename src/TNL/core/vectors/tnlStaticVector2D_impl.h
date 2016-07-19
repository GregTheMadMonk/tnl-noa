/***************************************************************************
                          tnlStaticVector2D_impl.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/core/mfuncs.h>

namespace TNL {

template< typename Real >
__cuda_callable__
tnlStaticVector< 2, Real >::tnlStaticVector()
{
}

template< typename Real >
__cuda_callable__
tnlStaticVector< 2, Real >::tnlStaticVector( const Real v[ 2 ] )
: tnlStaticArray< 2, Real >( v )
{
}

template< typename Real >
__cuda_callable__
tnlStaticVector< 2, Real >::tnlStaticVector( const Real& v )
: tnlStaticArray< 2, Real >( v )
{
}

template< typename Real >
__cuda_callable__
tnlStaticVector< 2, Real >::tnlStaticVector( const Real& v1, const Real& v2 )
: tnlStaticArray< 2, Real >( v1, v2 )
{
}

template< typename Real >
__cuda_callable__
tnlStaticVector< 2, Real >::tnlStaticVector( const tnlStaticVector< 2, Real >& v )
: tnlStaticArray< 2, Real >( v )
{
}

template< typename Real >
tnlString tnlStaticVector< 2, Real >::getType()
{
   return tnlString( "tnlStaticVector< " ) +
          tnlString( 2 ) +
          tnlString( ", " ) +
         TNL::getType< Real >() +
          tnlString( " >" );
}

template< typename Real >
__cuda_callable__
tnlStaticVector< 2, Real >& tnlStaticVector< 2, Real >::operator += ( const tnlStaticVector& v )
{
   this->data[ 0 ] += v[ 0 ];
   this->data[ 1 ] += v[ 1 ];
   return *this;
}

template< typename Real >
__cuda_callable__
tnlStaticVector< 2, Real >& tnlStaticVector< 2, Real >::operator -= ( const tnlStaticVector& v )
{
   this->data[ 0 ] -= v[ 0 ];
   this->data[ 1 ] -= v[ 1 ];
   return *this;
}

template< typename Real >
__cuda_callable__
tnlStaticVector< 2, Real >& tnlStaticVector< 2, Real >::operator *= ( const Real& c )
{
   this->data[ 0 ] *= c;
   this->data[ 1 ] *= c;
   return *this;
}

template< typename Real >
__cuda_callable__
tnlStaticVector< 2, Real > tnlStaticVector< 2, Real >::operator + ( const tnlStaticVector& u ) const
{
   tnlStaticVector< 2, Real > res;
   res[ 0 ] = this->data[ 0 ] + u[ 0 ];
   res[ 1 ] = this->data[ 1 ] + u[ 1 ];
   return res;
}

template< typename Real >
__cuda_callable__
tnlStaticVector< 2, Real > tnlStaticVector< 2, Real >::operator - ( const tnlStaticVector& u ) const
{
   tnlStaticVector< 2, Real > res;
   res[ 0 ] = this->data[ 0 ] - u[ 0 ];
   res[ 1 ] = this->data[ 1 ] - u[ 1 ];
   return res;
}

template< typename Real >
__cuda_callable__
tnlStaticVector< 2, Real > tnlStaticVector< 2, Real >::operator * ( const Real& c ) const
{
   tnlStaticVector< 2, Real > res;
   res[ 0 ] = c * this->data[ 0 ];
   res[ 1 ] = c * this->data[ 1 ];
   return res;
}

template< typename Real >
__cuda_callable__
Real tnlStaticVector< 2, Real >::operator * ( const tnlStaticVector& u ) const
{
   return this->data[ 0 ] * u[ 0 ] +
          this->data[ 1 ] * u[ 1 ];
}

template< typename Real >
__cuda_callable__
bool tnlStaticVector< 2, Real >::operator < ( const tnlStaticVector& v ) const
{
   return ( this->data[ 0 ] < v[ 0 ] &&
            this->data[ 1 ] < v[ 1 ] );
}

template< typename Real >
__cuda_callable__
bool tnlStaticVector< 2, Real >::operator <= ( const tnlStaticVector& v ) const
{
   return ( this->data[ 0 ] <= v[ 0 ] &&
            this->data[ 1 ] <= v[ 1 ] );
}

template< typename Real >
__cuda_callable__
bool tnlStaticVector< 2, Real >::operator > ( const tnlStaticVector& v ) const
{
   return ( this->data[ 0 ] > v[ 0 ] &&
            this->data[ 1 ] > v[ 1 ] );
}

template< typename Real >
__cuda_callable__
bool tnlStaticVector< 2, Real >::operator >= ( const tnlStaticVector& v ) const
{
   return ( this->data[ 0 ] >= v[ 0 ] &&
            this->data[ 1 ] >= v[ 1 ] );
}

template< typename Real >
   template< typename OtherReal >
__cuda_callable__
tnlStaticVector< 2, Real >::
operator tnlStaticVector< 2, OtherReal >() const
{
   tnlStaticVector< 2, OtherReal > aux;
   aux[ 0 ] = this->data[ 0 ];
   aux[ 1 ] = this->data[ 1 ];
   return aux;
}

template< typename Real >
__cuda_callable__
tnlStaticVector< 2, Real >
tnlStaticVector< 2, Real >::abs() const
{
   return tnlStaticVector< 2, Real >( ::abs( this->data[ 0 ] ),
                                      ::abs( this->data[ 1 ] ) );
}


#ifdef UNDEF //TEMPLATE_EXPLICIT_INSTANTIATION

#ifndef HAVE_CUDA
// TODO: does not work with CUDA
#ifdef INSTANTIATE_FLOAT
extern template class tnlStaticVector< 2, float >;
#endif
extern template class tnlStaticVector< 2, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlStaticVector< 2, long double >;
#endif
#endif

#endif

} // namespace TNL
