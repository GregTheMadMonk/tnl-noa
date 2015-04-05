/***************************************************************************
                          tnlStaticVector_impl.h  -  description
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

#ifndef TNLSTATICVECTOR_IMPL_H_
#define TNLSTATICVECTOR_IMPL_H_

template< int Size, typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< Size, Real >::tnlStaticVector()
{
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< Size, Real >::tnlStaticVector( const Real v[ Size ] )
: tnlStaticArray< Size, Real >( v )
{
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< Size, Real >::tnlStaticVector( const Real& v )
: tnlStaticArray< Size, Real >( v )
{
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< Size, Real >::tnlStaticVector( const tnlStaticVector< Size, Real >& v )
: tnlStaticArray< Size, Real >( v )
{
}

template< int Size, typename Real >
tnlString tnlStaticVector< Size, Real >::getType()
{
   return tnlString( "tnlStaticVector< " ) +
          tnlString( Size ) +
          tnlString( ", " ) +
          ::getType< Real >() +
          tnlString( " >" );
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< Size, Real >& tnlStaticVector< Size, Real >::operator += ( const tnlStaticVector& v )
{
   for( int i = 0; i < Size; i++ )
      this->data[ i ] += v[ i ];
   return *this;
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< Size, Real >& tnlStaticVector< Size, Real >::operator -= ( const tnlStaticVector& v )
{
   for( int i = 0; i < Size; i++ )
      this->data[ i ] -= v[ i ];
   return *this;
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< Size, Real >& tnlStaticVector< Size, Real >::operator *= ( const Real& c )
{
   for( int i = 0; i < Size; i++ )
      this->data[ i ] *= c;
   return *this;
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< Size, Real > tnlStaticVector< Size, Real >::operator + ( const tnlStaticVector& u ) const
{
   tnlStaticVector< Size, Real > res;
   for( int i = 0; i < Size; i++ )
      res[ i ] = this->data[ i ] + u[ i ];
   return res;
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< Size, Real > tnlStaticVector< Size, Real >::operator - ( const tnlStaticVector& u ) const
{
   tnlStaticVector< Size, Real > res;
   for( int i = 0; i < Size; i++ )
      res[ i ] = this->data[ i ] - u[ i ];
   return res;
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
tnlStaticVector< Size, Real > tnlStaticVector< Size, Real >::operator * ( const Real& c ) const
{
   tnlStaticVector< Size, Real > res;
   for( int i = 0; i < Size; i++ )
      res[ i ] = c * this->data[ i ];
   return res;
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
Real tnlStaticVector< Size, Real >::operator * ( const tnlStaticVector& u ) const
{
   Real res( 0.0 );
   for( int i = 0; i < Size; i++ )
      res += this->data[ i ] * u[ i ];
   return res;
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
bool tnlStaticVector< Size, Real >::operator < ( const tnlStaticVector& v ) const
{
   for( int i = 0; i < Size; i++ )
      if( this->data[ i ] >= v[ i ] )
         return false;
   return true;
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
bool tnlStaticVector< Size, Real >::operator <= ( const tnlStaticVector& v ) const
{
   for( int i = 0; i < Size; i++ )
      if( this->data[ i ] > v[ i ] )
         return false;
   return true;
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
bool tnlStaticVector< Size, Real >::operator > ( const tnlStaticVector& v ) const
{
   for( int i = 0; i < Size; i++ )
      if( this->data[ i ] <= v[ i ] )
         return false;
   return true;
}

template< int Size, typename Real >
#ifdef HAVE_CUDA
__host__ __device__
#endif
bool tnlStaticVector< Size, Real >::operator >= ( const tnlStaticVector& v ) const
{
   for( int i = 0; i < Size; i++ )
      if( this->data[ i ] < v[ i ] )
         return false;
   return true;
}

template< int Size, typename Real >
tnlStaticVector< Size, Real > operator * ( const Real& c, const tnlStaticVector< Size, Real >& u )
{
   return u * c;
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifndef HAVE_CUDA
// TODO: does not work with CUDA
extern template class tnlStaticVector< 4, float >;
extern template class tnlStaticVector< 4, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlStaticVector< 4, long double >;
#endif
#endif

#endif

#endif /* TNLSTATICVECTOR_IMPL_H_ */
