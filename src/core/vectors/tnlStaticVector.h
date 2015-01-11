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

template< int Size, typename Real = double >
class tnlStaticVector : public tnlStaticArray< Size, Real >
{
   public:
   typedef Real RealType;
   enum { size = Size };

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector();

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector( const Real v[ Size ] );

   //! This sets all vector components to v
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector( const Real& v );

   //! Copy constructor
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector( const tnlStaticVector< Size, Real >& v );

   static tnlString getType();

   //! Adding operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector& operator += ( const tnlStaticVector& v );

   //! Subtracting operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector& operator -= ( const tnlStaticVector& v );

   //! Multiplication with number
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector& operator *= ( const Real& c );

   //! Adding operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector operator + ( const tnlStaticVector& u ) const;

   //! Subtracting operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector operator - ( const tnlStaticVector& u ) const;

   //! Multiplication with number
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector operator * ( const Real& c ) const;

   //! Scalar product
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   Real operator * ( const tnlStaticVector& u ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator < ( const tnlStaticVector& v ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator <= ( const tnlStaticVector& v ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator > ( const tnlStaticVector& v ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator >= ( const tnlStaticVector& v ) const;
};

template< typename Real >
class tnlStaticVector< 1, Real > : public tnlStaticArray< 1, Real >
{
   public:
   typedef Real RealType;
   enum { size = 1 };

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector();

   //! This sets all vector components to v
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector( const Real& v );

   //! Copy constructor
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector( const tnlStaticVector< 1, Real >& v );

   static tnlString getType();

   //! Adding operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector& operator += ( const tnlStaticVector& v );

   //! Subtracting operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector& operator -= ( const tnlStaticVector& v );

   //! Multiplication with number
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector& operator *= ( const Real& c );

   //! Adding operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector operator + ( const tnlStaticVector& u ) const;

   //! Subtracting operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector operator - ( const tnlStaticVector& u ) const;

   //! Multiplication with number
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector operator * ( const Real& c ) const;

   //! Scalar product
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   Real operator * ( const tnlStaticVector& u ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator < ( const tnlStaticVector& v ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator <= ( const tnlStaticVector& v ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator > ( const tnlStaticVector& v ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator >= ( const tnlStaticVector& v ) const;
};

template< typename Real >
class tnlStaticVector< 2, Real > : public tnlStaticArray< 2, Real >
{
   public:
   typedef Real RealType;
   enum { size = 2 };

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector();

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector( const Real v[ 2 ] );

   //! This sets all vector components to v
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector( const Real& v );

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector( const Real& v1, const Real& v2 );

   //! Copy constructor
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector( const tnlStaticVector< 2, Real >& v );

   static tnlString getType();

   //! Adding operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector& operator += ( const tnlStaticVector& v );

   //! Subtracting operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector& operator -= ( const tnlStaticVector& v );

   //! Multiplication with number
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector& operator *= ( const Real& c );

   //! Adding operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector operator + ( const tnlStaticVector& u ) const;

   //! Subtracting operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector operator - ( const tnlStaticVector& u ) const;

   //! Multiplication with number
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector operator * ( const Real& c ) const;

   //! Scalar product
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   Real operator * ( const tnlStaticVector& u ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator < ( const tnlStaticVector& v ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator <= ( const tnlStaticVector& v ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator > ( const tnlStaticVector& v ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator >= ( const tnlStaticVector& v ) const;
};

template< typename Real >
class tnlStaticVector< 3, Real > : public tnlStaticArray< 3, Real >
{
   public:
   typedef Real RealType;
   enum { size = 3 };

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector();

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector( const Real v[ 3 ] );

   //! This sets all vector components to v
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector( const Real& v );

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector( const Real& v1, const Real& v2, const Real& v3 );

   //! Copy constructor
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector( const tnlStaticVector< 3, Real >& v );

   static tnlString getType();

   //! Adding operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector& operator += ( const tnlStaticVector& v );

   //! Subtracting operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector& operator -= ( const tnlStaticVector& v );

   //! Multiplication with number
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector& operator *= ( const Real& c );

   //! Adding operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector operator + ( const tnlStaticVector& u ) const;

   //! Subtracting operator
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector operator - ( const tnlStaticVector& u ) const;

   //! Multiplication with number
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   tnlStaticVector operator * ( const Real& c ) const;

   //! Scalar product
#ifdef HAVE_CUDA
   __host__ __device__
#endif
   Real operator * ( const tnlStaticVector& u ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator < ( const tnlStaticVector& v ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator <= ( const tnlStaticVector& v ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator > ( const tnlStaticVector& v ) const;

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   bool operator >= ( const tnlStaticVector& v ) const;
};

template< int Size, typename Real >
tnlStaticVector< Size, Real > operator * ( const Real& c, const tnlStaticVector< Size, Real >& u );

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
