/***************************************************************************
                          StaticVector.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/StaticArray.h>

namespace TNL {
namespace Containers {   

template< int Size, typename Real = double >
class StaticVector : public Containers::StaticArray< Size, Real >
{
   public:
   typedef Real RealType;
   typedef StaticVector< Size, Real > ThisType;
   enum { size = Size };

   using Containers::StaticArray< Size, Real >::operator=;
   
   __cuda_callable__
   StaticVector();

   __cuda_callable__
   StaticVector( const Real v[ Size ] );

   //! This sets all vector components to v
   __cuda_callable__
   StaticVector( const Real& v );

   //! Copy constructor
   __cuda_callable__
   StaticVector( const StaticVector< Size, Real >& v );
   
   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" );      

   static String getType();

   //! Adding operator
   __cuda_callable__
   StaticVector& operator += ( const StaticVector& v );

   //! Subtracting operator
   __cuda_callable__
   StaticVector& operator -= ( const StaticVector& v );

   //! Multiplication with number
   __cuda_callable__
   StaticVector& operator *= ( const Real& c );

   //! Addition operator
   __cuda_callable__
   StaticVector operator + ( const StaticVector& u ) const;

   //! Subtraction operator
   __cuda_callable__
   StaticVector operator - ( const StaticVector& u ) const;

   //! Multiplication with number
   __cuda_callable__
   StaticVector operator * ( const Real& c ) const;

   //! Scalar product
   __cuda_callable__
   Real operator * ( const StaticVector& u ) const;

   __cuda_callable__
   bool operator < ( const StaticVector& v ) const;

   __cuda_callable__
   bool operator <= ( const StaticVector& v ) const;

   __cuda_callable__
   bool operator > ( const StaticVector& v ) const;

   __cuda_callable__
   bool operator >= ( const StaticVector& v ) const;

   template< typename OtherReal >
   __cuda_callable__
   operator StaticVector< Size, OtherReal >() const;
 
   __cuda_callable__
   ThisType abs() const;
   
   __cuda_callable__
   Real lpNorm( const Real& p ) const;
};

template< typename Real >
class StaticVector< 1, Real > : public Containers::StaticArray< 1, Real >
{
   public:
   typedef Real RealType;
   typedef StaticVector< 1, Real > ThisType;
   enum { size = 1 };
   
   __cuda_callable__
   StaticVector();

   //! This sets all vector components to v
   __cuda_callable__
   StaticVector( const Real& v );

   //! Copy constructor
   __cuda_callable__
   StaticVector( const StaticVector< 1, Real >& v );
   
   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" );      

   static String getType();

   //! Addition operator
   __cuda_callable__
   StaticVector& operator += ( const StaticVector& v );

   //! Subtraction operator
   __cuda_callable__
   StaticVector& operator -= ( const StaticVector& v );

   //! Multiplication with number
   __cuda_callable__
   StaticVector& operator *= ( const Real& c );

   //! Addition operator
   __cuda_callable__
   StaticVector operator + ( const StaticVector& u ) const;

   //! Subtraction operator
   __cuda_callable__
   StaticVector operator - ( const StaticVector& u ) const;

   //! Multiplication with number
   __cuda_callable__
   StaticVector operator * ( const Real& c ) const;

   //! Scalar product
   __cuda_callable__
   Real operator * ( const StaticVector& u ) const;

   __cuda_callable__
   bool operator < ( const StaticVector& v ) const;

   __cuda_callable__
   bool operator <= ( const StaticVector& v ) const;

   __cuda_callable__
   bool operator > ( const StaticVector& v ) const;

   __cuda_callable__
   bool operator >= ( const StaticVector& v ) const;

   template< typename OtherReal >
   __cuda_callable__
   operator StaticVector< 1, OtherReal >() const;
 
   __cuda_callable__
   ThisType abs() const;
   
   __cuda_callable__
   Real lpNorm( const Real& p ) const;   
};

template< typename Real >
class StaticVector< 2, Real > : public Containers::StaticArray< 2, Real >
{
   public:
   typedef Real RealType;
   typedef StaticVector< 2, Real > ThisType;
   enum { size = 2 };
   
   __cuda_callable__
   StaticVector();

   __cuda_callable__
   StaticVector( const Real v[ 2 ] );

   //! This sets all vector components to v
   __cuda_callable__
   StaticVector( const Real& v );

   __cuda_callable__
   StaticVector( const Real& v1, const Real& v2 );

   //! Copy constructor
   __cuda_callable__
   StaticVector( const StaticVector< 2, Real >& v );
   
   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" );      

   static String getType();

   //! Adding operator
   __cuda_callable__
   StaticVector& operator += ( const StaticVector& v );

   //! Subtracting operator
   __cuda_callable__
   StaticVector& operator -= ( const StaticVector& v );

   //! Multiplication with number
   __cuda_callable__
   StaticVector& operator *= ( const Real& c );

   //! Adding operator
   __cuda_callable__
   StaticVector operator + ( const StaticVector& u ) const;

   //! Subtracting operator
   __cuda_callable__
   StaticVector operator - ( const StaticVector& u ) const;

   //! Multiplication with number
   __cuda_callable__
   StaticVector operator * ( const Real& c ) const;

   //! Scalar product
   __cuda_callable__
   Real operator * ( const StaticVector& u ) const;

   __cuda_callable__
   bool operator < ( const StaticVector& v ) const;

   __cuda_callable__
   bool operator <= ( const StaticVector& v ) const;

   __cuda_callable__
   bool operator > ( const StaticVector& v ) const;

   __cuda_callable__
   bool operator >= ( const StaticVector& v ) const;
 
   template< typename OtherReal >
   __cuda_callable__
   operator StaticVector< 2, OtherReal >() const;
 
   __cuda_callable__
   ThisType abs() const;
   
   __cuda_callable__
   Real lpNorm( const Real& p ) const;   
};

template< typename Real >
class StaticVector< 3, Real > : public Containers::StaticArray< 3, Real >
{
   public:
   typedef Real RealType;
   typedef StaticVector< 3, Real > ThisType;
   enum { size = 3 };
   
   __cuda_callable__
   StaticVector();

   __cuda_callable__
   StaticVector( const Real v[ 3 ] );

   //! This sets all vector components to v
   __cuda_callable__
   StaticVector( const Real& v );

   __cuda_callable__
   StaticVector( const Real& v1, const Real& v2, const Real& v3 );

   //! Copy constructor
   __cuda_callable__
   StaticVector( const StaticVector< 3, Real >& v );
   
   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" );      

   static String getType();

   //! Addition operator
   __cuda_callable__
   StaticVector& operator += ( const StaticVector& v );

   //! Subtraction operator
   __cuda_callable__
   StaticVector& operator -= ( const StaticVector& v );

   //! Multiplication with number
   __cuda_callable__
   StaticVector& operator *= ( const Real& c );

   //! Addition operator
   __cuda_callable__
   StaticVector operator + ( const StaticVector& u ) const;

   //! Subtraction operator
   __cuda_callable__
   StaticVector operator - ( const StaticVector& u ) const;

   //! Multiplication with number
   __cuda_callable__
   StaticVector operator * ( const Real& c ) const;

   //! Scalar product
   __cuda_callable__
   Real operator * ( const StaticVector& u ) const;

   __cuda_callable__
   bool operator < ( const StaticVector& v ) const;

   __cuda_callable__
   bool operator <= ( const StaticVector& v ) const;

   __cuda_callable__
   bool operator > ( const StaticVector& v ) const;

   __cuda_callable__
   bool operator >= ( const StaticVector& v ) const;

   template< typename OtherReal >
   __cuda_callable__
   operator StaticVector< 3, OtherReal >() const;
 
   __cuda_callable__
   ThisType abs() const;
   
   __cuda_callable__
   Real lpNorm( const Real& p ) const;   
};

template< int Size, typename Real, typename Scalar >
__cuda_callable__
StaticVector< Size, Real > operator * ( const Scalar& c, const StaticVector< Size, Real >& u );

template< int Size, typename Real >
__cuda_callable__
StaticVector< Size, Real > abs( const StaticVector< Size, Real >& u ) { return u.abs(); };

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/StaticVector_impl.h>
#include <TNL/Containers/StaticVector1D_impl.h>
#include <TNL/Containers/StaticVector2D_impl.h>
#include <TNL/Containers/StaticVector3D_impl.h>


namespace TNL {
namespace Containers {   
// TODO: move to some other source file

template< typename Real >
StaticVector< 3, Real > VectorProduct( const StaticVector< 3, Real >& u,
                                             const StaticVector< 3, Real >& v )
{
   StaticVector< 3, Real > p;
   p[ 0 ] = u[ 1 ] * v[ 2 ] - u[ 2 ] * v[ 1 ];
   p[ 1 ] = u[ 2 ] * v[ 0 ] - u[ 0 ] * v[ 2 ];
   p[ 2 ] = u[ 0 ] * v[ 1 ] - u[ 1 ] * v[ 0 ];
   return p;
};

template< typename Real >
Real tnlScalarProduct( const StaticVector< 2, Real >& u,
                       const StaticVector< 2, Real >& v )
{
   return u[ 0 ] * v[ 0 ] + u[ 1 ] * v[ 1 ];
};

template< typename Real >
Real tnlScalarProduct( const StaticVector< 3, Real >& u,
                       const StaticVector< 3, Real >& v )
{
   return u[ 0 ] * v[ 0 ] + u[ 1 ] * v[ 1 ] + u[ 2 ] * v[ 2 ];
};

template< typename Real >
Real tnlTriangleArea( const StaticVector< 2, Real >& a,
                      const StaticVector< 2, Real >& b,
                      const StaticVector< 2, Real >& c )
{
   StaticVector< 3, Real > u1, u2;
   u1. x() = b. x() - a. x();
   u1. y() = b. y() - a. y();
   u1. z() = 0.0;
   u2. x() = c. x() - a. x();
   u2. y() = c. y() - a. y();
   u2. z() = 0;

   const StaticVector< 3, Real > v = VectorProduct( u1, u2 );
   return 0.5 * ::sqrt( tnlScalarProduct( v, v ) );
};

template< typename Real >
Real tnlTriangleArea( const StaticVector< 3, Real >& a,
                      const StaticVector< 3, Real >& b,
                      const StaticVector< 3, Real >& c )
{
   StaticVector< 3, Real > u1, u2;
   u1. x() = b. x() - a. x();
   u1. y() = b. y() - a. y();
   u1. z() = b. z() - a. z();
   u2. x() = c. x() - a. x();
   u2. y() = c. y() - a. y();
   u2. z() = c. z() - a. z();

   const StaticVector< 3, Real > v = VectorProduct( u1, u2 );
   return 0.5 * ::sqrt( tnlScalarProduct( v, v ) );
};

} // namespace Containers
} // namespace TNL
