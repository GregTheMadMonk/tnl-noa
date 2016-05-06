/***************************************************************************
                          tnlStaticVector.h  -  description
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

#ifndef TNLSTATICVECTOR_H_
#define TNLSTATICVECTOR_H_

#include <core/arrays/tnlStaticArray.h>
#include <core/tnlDevice_Callable.h>

template< int Size, typename Real = double >
class tnlStaticVector : public tnlStaticArray< Size, Real >
{
   public:
   typedef Real RealType;
   typedef tnlStaticVector< Size, Real > ThisType;
   enum { size = Size };

   __device_callable__
   tnlStaticVector();

   __device_callable__
   tnlStaticVector( const Real v[ Size ] );

   //! This sets all vector components to v
   __device_callable__
   tnlStaticVector( const Real& v );

   //! Copy constructor
   __device_callable__
   tnlStaticVector( const tnlStaticVector< Size, Real >& v );

   static tnlString getType();

   //! Adding operator
   __device_callable__
   tnlStaticVector& operator += ( const tnlStaticVector& v );

   //! Subtracting operator
   __device_callable__
   tnlStaticVector& operator -= ( const tnlStaticVector& v );

   //! Multiplication with number
   __device_callable__
   tnlStaticVector& operator *= ( const Real& c );

   //! Addition operator
   __device_callable__
   tnlStaticVector operator + ( const tnlStaticVector& u ) const;

   //! Subtraction operator
   __device_callable__
   tnlStaticVector operator - ( const tnlStaticVector& u ) const;

   //! Multiplication with number
   __device_callable__
   tnlStaticVector operator * ( const Real& c ) const;

   //! Scalar product
   __device_callable__
   Real operator * ( const tnlStaticVector& u ) const;

   __device_callable__
   bool operator < ( const tnlStaticVector& v ) const;

   __device_callable__
   bool operator <= ( const tnlStaticVector& v ) const;

   __device_callable__
   bool operator > ( const tnlStaticVector& v ) const;

   __device_callable__
   bool operator >= ( const tnlStaticVector& v ) const;

   template< typename OtherReal >
   __device_callable__
   operator tnlStaticVector< Size, OtherReal >() const;   
   
   __device_callable__
   ThisType abs() const;
};

template< typename Real >
class tnlStaticVector< 1, Real > : public tnlStaticArray< 1, Real >
{
   public:
   typedef Real RealType;
   typedef tnlStaticVector< 1, Real > ThisType;
   enum { size = 1 };

   __device_callable__
   tnlStaticVector();

   //! This sets all vector components to v
   __device_callable__
   tnlStaticVector( const Real& v );

   //! Copy constructor
   __device_callable__
   tnlStaticVector( const tnlStaticVector< 1, Real >& v );

   static tnlString getType();

   //! Addition operator
   __device_callable__
   tnlStaticVector& operator += ( const tnlStaticVector& v );

   //! Subtraction operator
   __device_callable__
   tnlStaticVector& operator -= ( const tnlStaticVector& v );

   //! Multiplication with number
   __device_callable__
   tnlStaticVector& operator *= ( const Real& c );

   //! Addition operator
   __device_callable__
   tnlStaticVector operator + ( const tnlStaticVector& u ) const;

   //! Subtraction operator
   __device_callable__
   tnlStaticVector operator - ( const tnlStaticVector& u ) const;

   //! Multiplication with number
   __device_callable__
   tnlStaticVector operator * ( const Real& c ) const;

   //! Scalar product
   __device_callable__
   Real operator * ( const tnlStaticVector& u ) const;

   __device_callable__
   bool operator < ( const tnlStaticVector& v ) const;

   __device_callable__
   bool operator <= ( const tnlStaticVector& v ) const;

   __device_callable__
   bool operator > ( const tnlStaticVector& v ) const;

   __device_callable__
   bool operator >= ( const tnlStaticVector& v ) const;

   template< typename OtherReal >
   __device_callable__
   operator tnlStaticVector< 1, OtherReal >() const;   
   
   __device_callable__
   ThisType abs() const;
};

template< typename Real >
class tnlStaticVector< 2, Real > : public tnlStaticArray< 2, Real >
{
   public:
   typedef Real RealType;
   typedef tnlStaticVector< 2, Real > ThisType;
   enum { size = 2 };

   __device_callable__
   tnlStaticVector();

   __device_callable__
   tnlStaticVector( const Real v[ 2 ] );

   //! This sets all vector components to v
   __device_callable__
   tnlStaticVector( const Real& v );

   __device_callable__
   tnlStaticVector( const Real& v1, const Real& v2 );

   //! Copy constructor
   __device_callable__
   tnlStaticVector( const tnlStaticVector< 2, Real >& v );

   static tnlString getType();

   //! Adding operator
   __device_callable__
   tnlStaticVector& operator += ( const tnlStaticVector& v );

   //! Subtracting operator
   __device_callable__
   tnlStaticVector& operator -= ( const tnlStaticVector& v );

   //! Multiplication with number
   __device_callable__
   tnlStaticVector& operator *= ( const Real& c );

   //! Adding operator
   __device_callable__
   tnlStaticVector operator + ( const tnlStaticVector& u ) const;

   //! Subtracting operator
   __device_callable__
   tnlStaticVector operator - ( const tnlStaticVector& u ) const;

   //! Multiplication with number
   __device_callable__
   tnlStaticVector operator * ( const Real& c ) const;

   //! Scalar product
   __device_callable__
   Real operator * ( const tnlStaticVector& u ) const;

   __device_callable__
   bool operator < ( const tnlStaticVector& v ) const;

   __device_callable__
   bool operator <= ( const tnlStaticVector& v ) const;

   __device_callable__
   bool operator > ( const tnlStaticVector& v ) const;

   __device_callable__
   bool operator >= ( const tnlStaticVector& v ) const;
   
   template< typename OtherReal >
   __device_callable__
   operator tnlStaticVector< 2, OtherReal >() const;   
   
   __device_callable__
   ThisType abs() const;
};

template< typename Real >
class tnlStaticVector< 3, Real > : public tnlStaticArray< 3, Real >
{
   public:
   typedef Real RealType;
   typedef tnlStaticVector< 3, Real > ThisType;
   enum { size = 3 };

   __device_callable__
   tnlStaticVector();

   __device_callable__
   tnlStaticVector( const Real v[ 3 ] );

   //! This sets all vector components to v
   __device_callable__
   tnlStaticVector( const Real& v );

   __device_callable__
   tnlStaticVector( const Real& v1, const Real& v2, const Real& v3 );

   //! Copy constructor
   __device_callable__
   tnlStaticVector( const tnlStaticVector< 3, Real >& v );

   static tnlString getType();

   //! Addition operator
   __device_callable__
   tnlStaticVector& operator += ( const tnlStaticVector& v );

   //! Subtraction operator
   __device_callable__
   tnlStaticVector& operator -= ( const tnlStaticVector& v );

   //! Multiplication with number
   __device_callable__
   tnlStaticVector& operator *= ( const Real& c );

   //! Addition operator
   __device_callable__
   tnlStaticVector operator + ( const tnlStaticVector& u ) const;

   //! Subtraction operator
   __device_callable__
   tnlStaticVector operator - ( const tnlStaticVector& u ) const;

   //! Multiplication with number
   __device_callable__
   tnlStaticVector operator * ( const Real& c ) const;

   //! Scalar product
   __device_callable__
   Real operator * ( const tnlStaticVector& u ) const;

   __device_callable__
   bool operator < ( const tnlStaticVector& v ) const;

   __device_callable__
   bool operator <= ( const tnlStaticVector& v ) const;

   __device_callable__
   bool operator > ( const tnlStaticVector& v ) const;

   __device_callable__
   bool operator >= ( const tnlStaticVector& v ) const;

   template< typename OtherReal >
   __device_callable__
   operator tnlStaticVector< 3, OtherReal >() const;   
   
   __device_callable__
   ThisType abs() const;
};

template< int Size, typename Real >
tnlStaticVector< Size, Real > operator * ( const Real& c, const tnlStaticVector< Size, Real >& u );

template< int Size, typename Real >
tnlStaticVector< Size, Real > tnlAbs( const tnlStaticVector< Size, Real >& u ) { return u.abs(); };

#include <core/vectors/tnlStaticVector_impl.h>
#include <core/vectors/tnlStaticVector1D_impl.h>
#include <core/vectors/tnlStaticVector2D_impl.h>
#include <core/vectors/tnlStaticVector3D_impl.h>

// TODO: move to some other source file

template< typename Real >
tnlStaticVector< 3, Real > tnlVectorProduct( const tnlStaticVector< 3, Real >& u,
                                             const tnlStaticVector< 3, Real >& v )
{
   tnlStaticVector< 3, Real > p;
   p[ 0 ] = u[ 1 ] * v[ 2 ] - u[ 2 ] * v[ 1 ];
   p[ 1 ] = u[ 2 ] * v[ 0 ] - u[ 0 ] * v[ 2 ];
   p[ 2 ] = u[ 0 ] * v[ 1 ] - u[ 1 ] * v[ 0 ];
   return p;
};

template< typename Real >
Real tnlScalarProduct( const tnlStaticVector< 2, Real >& u,
                       const tnlStaticVector< 2, Real >& v )
{
   return u[ 0 ] * v[ 0 ] + u[ 1 ] * v[ 1 ];
};

template< typename Real >
Real tnlScalarProduct( const tnlStaticVector< 3, Real >& u,
                       const tnlStaticVector< 3, Real >& v )
{
   return u[ 0 ] * v[ 0 ] + u[ 1 ] * v[ 1 ] + u[ 2 ] * v[ 2 ];
};

template< typename Real >
Real tnlTriangleArea( const tnlStaticVector< 2, Real >& a,
                      const tnlStaticVector< 2, Real >& b,
                      const tnlStaticVector< 2, Real >& c )
{
   tnlStaticVector< 3, Real > u1, u2;
   u1. x() = b. x() - a. x();
   u1. y() = b. y() - a. y();
   u1. z() = 0.0;
   u2. x() = c. x() - a. x();
   u2. y() = c. y() - a. y();
   u2. z() = 0;

   const tnlStaticVector< 3, Real > v = tnlVectorProduct( u1, u2 );
   return 0.5 * sqrt( tnlScalarProduct( v, v ) );
};

template< typename Real >
Real tnlTriangleArea( const tnlStaticVector< 3, Real >& a,
                      const tnlStaticVector< 3, Real >& b,
                      const tnlStaticVector< 3, Real >& c )
{
   tnlStaticVector< 3, Real > u1, u2;
   u1. x() = b. x() - a. x();
   u1. y() = b. y() - a. y();
   u1. z() = 0.0;
   u2. x() = c. x() - a. x();
   u2. y() = c. y() - a. y();
   u2. z() = 0;

   const tnlStaticVector< 3, Real > v = tnlVectorProduct( u1, u2 );
   return 0.5 * sqrt( tnlScalarProduct( v, v ) );
};
#endif /* TNLSTATICVECTOR_H_ */
