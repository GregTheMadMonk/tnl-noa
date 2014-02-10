/***************************************************************************
                          tnlStaticVector2D_impl.h  -  description
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

#ifndef TNLSTATICVECTOR2D_IMPL_H_
#define TNLSTATICVECTOR2D_IMPL_H_

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 2, Real >::tnlStaticVector()
{
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 2, Real >::tnlStaticVector( const Real v[ 2 ] )
: tnlStaticArray< 2, Real >( v )
{
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 2, Real >::tnlStaticVector( const Real& v )
: tnlStaticArray< 2, Real >( v )
{
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 2, Real >::tnlStaticVector( const Real& v1, const Real& v2 )
: tnlStaticArray< 2, Real >( v1, v2 )
{
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 2, Real >::tnlStaticVector( const tnlStaticVector< 2, Real >& v )
: tnlStaticArray< 2, Real >( v )
{
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 2, Real >& tnlStaticVector< 2, Real >::operator += ( const tnlStaticVector& v )
{
   this->data[ 0 ] += v[ 0 ];
   this->data[ 1 ] += v[ 1 ];
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 2, Real >& tnlStaticVector< 2, Real >::operator -= ( const tnlStaticVector& v )
{
   this->data[ 0 ] -= v[ 0 ];
   this->data[ 1 ] -= v[ 1 ];
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 2, Real >& tnlStaticVector< 2, Real >::operator *= ( const Real& c )
{
   this->data[ 0 ] *= c;
   this->data[ 1 ] *= c;
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 2, Real > tnlStaticVector< 2, Real >::operator + ( const tnlStaticVector& u ) const
{
   tnlStaticVector< 2, Real > res;
   res[ 0 ] = this->data[ 0 ] + u[ 0 ];
   res[ 1 ] = this->data[ 1 ] + u[ 1 ];
   return res;
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 2, Real > tnlStaticVector< 2, Real >::operator - ( const tnlStaticVector& u ) const
{
   tnlStaticVector< 2, Real > res;
   res[ 0 ] = this->data[ 0 ] - u[ 0 ];
   res[ 1 ] = this->data[ 1 ] - u[ 1 ];
   return res;
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 2, Real > tnlStaticVector< 2, Real >::operator * ( const Real& c ) const
{
   tnlStaticVector< 2, Real > res;
   res[ 0 ] = c * this->data[ 0 ];
   res[ 1 ] = c * this->data[ 1 ];
   return res;
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
Real tnlStaticVector< 2, Real >::operator * ( const tnlStaticVector& u ) const
{
   return this->data[ 0 ] * u[ 0 ] +
          this->data[ 1 ] * u[ 1 ];
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
bool tnlStaticVector< 2, Real >::operator < ( const tnlStaticVector& v ) const
{
   return ( this->data[ 0 ] < v[ 0 ] &&
            this->data[ 1 ] < v[ 1 ] );
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
bool tnlStaticVector< 2, Real >::operator <= ( const tnlStaticVector& v ) const
{
   return ( this->data[ 0 ] <= v[ 0 ] &&
            this->data[ 1 ] <= v[ 1 ] );
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
bool tnlStaticVector< 2, Real >::operator > ( const tnlStaticVector& v ) const
{
   return ( this->data[ 0 ] > v[ 0 ] &&
            this->data[ 1 ] > v[ 1 ] );
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
bool tnlStaticVector< 2, Real >::operator >= ( const tnlStaticVector& v ) const
{
   return ( this->data[ 0 ] >= v[ 0 ] &&
            this->data[ 1 ] >= v[ 1 ] );
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

extern template class tnlStaticVector< 2, float >;
extern template class tnlStaticVector< 2, double >;
extern template class tnlStaticVector< 2, long double >;

#endif

#endif /* TNLSTATICVECTOR2D_IMPL_H_ */
