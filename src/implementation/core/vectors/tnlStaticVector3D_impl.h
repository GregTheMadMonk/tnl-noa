/***************************************************************************
                          tnlStaticVector3D_impl.h  -  description
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

#ifndef TNLSTATICVECTOR3D_IMPL_H_
#define TNLSTATICVECTOR3D_IMPL_H_

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 3, Real >::tnlStaticVector()
{
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 3, Real >::tnlStaticVector( const Real v[ 3 ] )
: tnlStaticArray< 3, Real >( v )
{
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 3, Real >::tnlStaticVector( const Real& v )
: tnlStaticArray< 3, Real >( v )
{
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 3, Real >::tnlStaticVector( const Real& v1, const Real& v2, const Real& v3 )
: tnlStaticArray< 3, Real >( v1, v2, v3 )
{
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 3, Real >::tnlStaticVector( const tnlStaticVector< 3, Real >& v )
: tnlStaticArray< 3, Real >( v )
{
}

template< typename Real >
tnlString tnlStaticVector< 3, Real >::getType()
{
   return tnlString( "tnlStaticVector< " ) +
          tnlString( 3 ) +
          tnlString( ", " ) +
          ::getType< Real >() +
          tnlString( " >" );
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 3, Real >& tnlStaticVector< 3, Real >::operator += ( const tnlStaticVector& v )
{
   this->data[ 0 ] += v[ 0 ];
   this->data[ 1 ] += v[ 1 ];
   this->data[ 2 ] += v[ 2 ];
   return *this;
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 3, Real >& tnlStaticVector< 3, Real >::operator -= ( const tnlStaticVector& v )
{
   this->data[ 0 ] -= v[ 0 ];
   this->data[ 1 ] -= v[ 1 ];
   this->data[ 2 ] -= v[ 2 ];
   return *this;
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 3, Real >& tnlStaticVector< 3, Real >::operator *= ( const Real& c )
{
   this->data[ 0 ] *= c;
   this->data[ 1 ] *= c;
   this->data[ 2 ] *= c;
   return *this;
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 3, Real > tnlStaticVector< 3, Real >::operator + ( const tnlStaticVector& u ) const
{
   tnlStaticVector< 3, Real > res;
   res[ 0 ] = this->data[ 0 ] + u[ 0 ];
   res[ 1 ] = this->data[ 1 ] + u[ 1 ];
   res[ 2 ] = this->data[ 2 ] + u[ 2 ];
   return res;
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 3, Real > tnlStaticVector< 3, Real >::operator - ( const tnlStaticVector& u ) const
{
   tnlStaticVector< 3, Real > res;
   res[ 0 ] = this->data[ 0 ] - u[ 0 ];
   res[ 1 ] = this->data[ 1 ] - u[ 1 ];
   res[ 2 ] = this->data[ 2 ] - u[ 2 ];
   return res;
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< 3, Real > tnlStaticVector< 3, Real >::operator * ( const Real& c ) const
{
   tnlStaticVector< 3, Real > res;
   res[ 0 ] = c * this->data[ 0 ];
   res[ 1 ] = c * this->data[ 1 ];
   res[ 2 ] = c * this->data[ 2 ];
   return res;
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
Real tnlStaticVector< 3, Real >::operator * ( const tnlStaticVector& u ) const
{
   return this->data[ 0 ] * u[ 0 ] +
          this->data[ 1 ] * u[ 1 ] +
          this->data[ 2 ] * u[ 2 ];
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
bool tnlStaticVector< 3, Real >::operator < ( const tnlStaticVector& v ) const
{
   return ( this->data[ 0 ] < v[ 0 ] &&
            this->data[ 1 ] < v[ 1 ] &&
            this->data[ 2 ] < v[ 2 ] );
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
bool tnlStaticVector< 3, Real >::operator <= ( const tnlStaticVector& v ) const
{
   return ( this->data[ 0 ] <= v[ 0 ] &&
            this->data[ 1 ] <= v[ 1 ] &&
            this->data[ 2 ] <= v[ 2 ] );
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
bool tnlStaticVector< 3, Real >::operator > ( const tnlStaticVector& v ) const
{
   return ( this->data[ 0 ] > v[ 0 ] &&
            this->data[ 1 ] > v[ 1 ] &&
            this->data[ 2 ] > v[ 2 ] );
}

template< typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
bool tnlStaticVector< 3, Real >::operator >= ( const tnlStaticVector& v ) const
{
   return ( this->data[ 0 ] >= v[ 0 ] &&
            this->data[ 1 ] >= v[ 1 ] &&
            this->data[ 2 ] >= v[ 2 ] );
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifndef HAVE_CUDA
// TODO: does not work with CUDA
extern template class tnlStaticVector< 3, float >;
extern template class tnlStaticVector< 3, double >;
//extern template class tnlStaticVector< 3, long double >;
#endif

#endif

#endif /* TNLSTATICVECTOR3D_IMPL_H_ */
