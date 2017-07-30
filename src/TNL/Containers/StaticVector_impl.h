/***************************************************************************
                          StaticVector_impl.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/StaticVector.h>

namespace TNL {
namespace Containers {   

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >::StaticVector()
{
}

template< int Size, typename Real >
   template< typename _unused >
__cuda_callable__
StaticVector< Size, Real >::StaticVector( const Real v[ Size ] )
: Containers::StaticArray< Size, Real >( v )
{
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >::StaticVector( const Real& v )
: Containers::StaticArray< Size, Real >( v )
{
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >::StaticVector( const StaticVector< Size, Real >& v )
: Containers::StaticArray< Size, Real >( v )
{
}

template< int Size, typename Real >
bool
StaticVector< Size, Real >::setup( const Config::ParameterContainer& parameters,
                                   const String& prefix )
{
   for( int i = 0; i < Size; i++ )
      if( ! parameters.template getParameter< double >( prefix + String( i ), this->data[ i ] ) )
         return false;
   return true;
}

template< int Size, typename Real >
String StaticVector< Size, Real >::getType()
{
   return String( "Containers::StaticVector< " ) +
          String( Size ) +
          String( ", " ) +
          TNL::getType< Real >() +
          String( " >" );
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator += ( const StaticVector& v )
{
   for( int i = 0; i < Size; i++ )
      this->data[ i ] += v[ i ];
   return *this;
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator -= ( const StaticVector& v )
{
   for( int i = 0; i < Size; i++ )
      this->data[ i ] -= v[ i ];
   return *this;
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator *= ( const Real& c )
{
   for( int i = 0; i < Size; i++ )
      this->data[ i ] *= c;
   return *this;
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real > StaticVector< Size, Real >::operator + ( const StaticVector& u ) const
{
   StaticVector< Size, Real > res;
   for( int i = 0; i < Size; i++ )
      res[ i ] = this->data[ i ] + u[ i ];
   return res;
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real > StaticVector< Size, Real >::operator - ( const StaticVector& u ) const
{
   StaticVector< Size, Real > res;
   for( int i = 0; i < Size; i++ )
      res[ i ] = this->data[ i ] - u[ i ];
   return res;
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real > StaticVector< Size, Real >::operator * ( const Real& c ) const
{
   StaticVector< Size, Real > res;
   for( int i = 0; i < Size; i++ )
      res[ i ] = c * this->data[ i ];
   return res;
}

template< int Size, typename Real >
__cuda_callable__
Real StaticVector< Size, Real >::operator * ( const StaticVector& u ) const
{
   Real res( 0.0 );
   for( int i = 0; i < Size; i++ )
      res += this->data[ i ] * u[ i ];
   return res;
}

template< int Size, typename Real >
__cuda_callable__
bool StaticVector< Size, Real >::operator < ( const StaticVector& v ) const
{
   for( int i = 0; i < Size; i++ )
      if( this->data[ i ] >= v[ i ] )
         return false;
   return true;
}

template< int Size, typename Real >
__cuda_callable__
bool StaticVector< Size, Real >::operator <= ( const StaticVector& v ) const
{
   for( int i = 0; i < Size; i++ )
      if( this->data[ i ] > v[ i ] )
         return false;
   return true;
}

template< int Size, typename Real >
__cuda_callable__
bool StaticVector< Size, Real >::operator > ( const StaticVector& v ) const
{
   for( int i = 0; i < Size; i++ )
      if( this->data[ i ] <= v[ i ] )
         return false;
   return true;
}

template< int Size, typename Real >
__cuda_callable__
bool StaticVector< Size, Real >::operator >= ( const StaticVector& v ) const
{
   for( int i = 0; i < Size; i++ )
      if( this->data[ i ] < v[ i ] )
         return false;
   return true;
}

template< int Size, typename Real >
   template< typename OtherReal >
__cuda_callable__
StaticVector< Size, Real >::
operator StaticVector< Size, OtherReal >() const
{
   StaticVector< Size, OtherReal > aux;
   for( int i = 0; i < Size; i++ )
      aux[ i ] = this->data[ i ];
   return aux;
}

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real >
StaticVector< Size, Real >::abs() const
{
   StaticVector< Size, Real > v;
   for( int i = 0; i < Size; i++ )
      v.data[ i ] = TNL::abs( this->data[ i ] );
   return v;
}

template< int Size, typename Real >
__cuda_callable__
Real
StaticVector< Size, Real >::lpNorm( const Real& p ) const
{
   if( p == 1.0 )
   {
      Real aux = TNL::abs( this->data[ 0 ] );
      for( int i = 1; i < Size; i++ )
         aux += TNL::abs( this->data[ i ] );
      return aux;
   }
   if( p == 2.0 )
   {
      Real aux = this->data[ 0 ] * this->data[ 0 ];
      for( int i = 1; i < Size; i++ )
         aux += this->data[ i ] * this->data[ i ];
      return std::sqrt( aux );
   }
   Real aux = std::pow( TNL::abs( this->data[ 0 ] ), p );
   for( int i = 1; i < Size; i++ )
      aux += std::pow( TNL::abs( this->data[ i ] ), p );
   return std::pow( aux, 1.0 / p );
}

template< int Size, typename Real, typename Scalar >
__cuda_callable__
StaticVector< Size, Real > operator * ( const Scalar& c, const StaticVector< Size, Real >& u )
{
   return u * c;
}

#ifdef UNDEF //TEMPLATE_EXPLICIT_INSTANTIATION

#ifndef HAVE_CUDA
// TODO: does not work with CUDA
#ifdef INSTANTIATE_FLOAT
extern template class StaticVector< 4, float >;
#endif
extern template class StaticVector< 4, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class StaticVector< 4, long double >;
#endif
#endif

#endif

} // namespace Containers
} // namespace TNL
