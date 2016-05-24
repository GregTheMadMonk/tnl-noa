/***************************************************************************
                          tnlStaticVector1D_impl.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLSTATICVECTOR1D_IMPL_H_
#define TNLSTATICVECTOR1D_IMPL_H_

template< typename Real >
__device_callable__
tnlStaticVector< 1, Real >::tnlStaticVector()
{
}

template< typename Real >
__device_callable__
tnlStaticVector< 1, Real >::tnlStaticVector( const Real& v )
: tnlStaticArray< 1, Real >( v )
{
}

template< typename Real >
__device_callable__
tnlStaticVector< 1, Real >::tnlStaticVector( const tnlStaticVector< 1, Real >& v )
: tnlStaticArray< 1, Real >( v )
{
}

template< typename Real >
tnlString tnlStaticVector< 1, Real >::getType()
{
   return tnlString( "tnlStaticVector< " ) +
          tnlString( 1 ) +
          tnlString( ", " ) +
          ::getType< Real >() +
          tnlString( " >" );
}

template< typename Real >
__device_callable__
tnlStaticVector< 1, Real >& tnlStaticVector< 1, Real >::operator += ( const tnlStaticVector& v )
{
   this->data[ 0 ] += v[ 0 ];
   return *this;
}

template< typename Real >
__device_callable__
tnlStaticVector< 1, Real >& tnlStaticVector< 1, Real >::operator -= ( const tnlStaticVector& v )
{
   this->data[ 0 ] -= v[ 0 ];
   return *this;
}

template< typename Real >
__device_callable__
tnlStaticVector< 1, Real >& tnlStaticVector< 1, Real >::operator *= ( const Real& c )
{
   this->data[ 0 ] *= c;
   return *this;
}

template< typename Real >
__device_callable__
tnlStaticVector< 1, Real > tnlStaticVector< 1, Real >::operator + ( const tnlStaticVector& u ) const
{
   tnlStaticVector< 1, Real > res;
   res[ 0 ] = this->data[ 0 ] + u[ 0 ];
   return res;
}

template< typename Real >
__device_callable__
tnlStaticVector< 1, Real > tnlStaticVector< 1, Real >::operator - ( const tnlStaticVector& u ) const
{
   tnlStaticVector< 1, Real > res;
   res[ 0 ] = this->data[ 0 ] - u[ 0 ];
   return res;
}

template< typename Real >
__device_callable__
tnlStaticVector< 1, Real > tnlStaticVector< 1, Real >::operator * ( const Real& c ) const
{
   tnlStaticVector< 1, Real > res;
   res[ 0 ] = c * this->data[ 0 ];
   return res;
}

template< typename Real >
__device_callable__
Real tnlStaticVector< 1, Real >::operator * ( const tnlStaticVector& u ) const
{
   return this->data[ 0 ] * u[ 0 ];
}

template< typename Real >
__device_callable__
bool tnlStaticVector< 1, Real >::operator < ( const tnlStaticVector& v ) const
{
   return ( this->data[ 0 ] < v[ 0 ] );
}

template< typename Real >
__device_callable__
bool tnlStaticVector< 1, Real >::operator <= ( const tnlStaticVector& v ) const
{
   return ( this->data[ 0 ] <= v[ 0 ] );
}

template< typename Real >
__device_callable__
bool tnlStaticVector< 1, Real >::operator > ( const tnlStaticVector& v ) const
{
   return ( this->data[ 0 ] > v[ 0 ] );
}

template< typename Real >
__device_callable__
bool tnlStaticVector< 1, Real >::operator >= ( const tnlStaticVector& v ) const
{
   return ( this->data[ 0 ] >= v[ 0 ] );
}

template<typename Real >
__device_callable__
inline tnlStaticVector< 1, Real >& tnlStaticVector< 1, Real >::operator = ( const tnlStaticVector< 1, Real >& vct )
{
    tnlStaticArray<1,Real>::operator =(vct);
}

template<typename Real >
template< typename Vct >
__device_callable__
inline tnlStaticVector< 1, Real >& tnlStaticVector< 1, Real >::operator = ( const Vct& vct )
{
    tnlStaticArray<1,Real>::operator =(vct);
}

template< typename Real >
   template< typename OtherReal >
__device_callable__
tnlStaticVector< 1, Real >::
operator tnlStaticVector< 1, OtherReal >() const
{
   tnlStaticVector< 1, OtherReal > aux;
   aux[ 0 ] = this->data[ 0 ];
   return aux;
}

template< typename Real >
__device_callable__
tnlStaticVector< 1, Real >
tnlStaticVector< 1, Real >::abs() const
{
   return tnlStaticVector< 1, Real >( tnlAbs( this->data[ 0 ] ) );
} 

#ifdef UNDEF //TEMPLATE_EXPLICIT_INSTANTIATION

#ifndef HAVE_CUDA
// TODO: does not work with CUDA
#ifdef INSTANTIATE_FLOAT
extern template class tnlStaticVector< 1, float >;
#endif
extern template class tnlStaticVector< 1, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlStaticVector< 1, long double >;
#endif
#endif

#endif

#endif /* TNLSTATICVECTOR1D_IMPL_H_ */
