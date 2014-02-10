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
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 1, Real >::tnlStaticVector()
{
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 1, Real >::tnlStaticVector( const Real& v )
: tnlStaticArray< 1, Real >( v )
{
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 1, Real >::tnlStaticVector( const tnlStaticVector< 1, Real >& v )
: tnlStaticArray< 1, Real >( v )
{
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 1, Real >& tnlStaticVector< 1, Real >::operator += ( const tnlStaticVector& v )
{
   this->data[ 0 ] += v[ 0 ];
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 1, Real >& tnlStaticVector< 1, Real >::operator -= ( const tnlStaticVector& v )
{
   this->data[ 0 ] -= v[ 0 ];
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 1, Real >& tnlStaticVector< 1, Real >::operator *= ( const Real& c )
{
   this->data[ 0 ] *= c;
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 1, Real > tnlStaticVector< 1, Real >::operator + ( const tnlStaticVector& u ) const
{
   tnlStaticVector< 1, Real > res;
   res[ 0 ] = this->data[ 0 ] + u[ 0 ];
   return res;
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 1, Real > tnlStaticVector< 1, Real >::operator - ( const tnlStaticVector& u ) const
{
   tnlStaticVector< 1, Real > res;
   res[ 0 ] = this->data[ 0 ] - u[ 0 ];
   return res;
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 1, Real > tnlStaticVector< 1, Real >::operator * ( const Real& c ) const
{
   tnlStaticVector< 1, Real > res;
   res[ 0 ] = c * this->data[ 0 ];
   return res;
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
Real tnlStaticVector< 1, Real >::operator * ( const tnlStaticVector& u ) const
{
   return this->data[ 0 ] * u[ 0 ];
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
bool tnlStaticVector< 1, Real >::operator < ( const tnlStaticVector& v ) const
{
   return ( this->data[ 0 ] < v[ 0 ] );
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
bool tnlStaticVector< 1, Real >::operator <= ( const tnlStaticVector& v ) const
{
   return ( this->data[ 0 ] <= v[ 0 ] );
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
bool tnlStaticVector< 1, Real >::operator > ( const tnlStaticVector& v ) const
{
   return ( this->data[ 0 ] > v[ 0 ] );
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
bool tnlStaticVector< 1, Real >::operator >= ( const tnlStaticVector& v ) const
{
   return ( this->data[ 0 ] >= v[ 0 ] );
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

extern template class tnlStaticVector< 1, float >;
extern template class tnlStaticVector< 1, double >;
extern template class tnlStaticVector< 1, long double >;

#endif

#endif /* TNLSTATICVECTOR1D_IMPL_H_ */
