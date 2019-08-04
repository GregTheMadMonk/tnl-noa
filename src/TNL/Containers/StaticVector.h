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
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/Expressions/StaticExpressionTemplates.h>

namespace TNL {
namespace Containers {

 /**
 * \brief Vector with constant size.
 *
 * \param Size Size of static vector. Number of its elements.
 * \param Real Type of the values in the static vector.
 */
template< int Size, typename Real = double >
class StaticVector : public StaticArray< Size, Real >
{
public:
   using RealType = Real;
   using IndexType = int;

   //! \brief Default constructor.
   __cuda_callable__
   StaticVector() = default;

   //! \brief Default copy constructor.
   __cuda_callable__
   StaticVector( const StaticVector& ) = default;

   //! \brief Default copy-assignment operator.
   StaticVector& operator=( const StaticVector& ) = default;

   //! \brief Default move-assignment operator.
   StaticVector& operator=( StaticVector&& ) = default;

   //! Constructors and assignment operators are inherited from the class \ref StaticArray.
   using StaticArray< Size, Real >::StaticArray;
   using StaticArray< Size, Real >::operator=;

   template< typename T1,
             typename T2,
             template< typename, typename > class Operation >
   __cuda_callable__
   StaticVector( const Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& op );

   template< typename T,
             template< typename > class Operation >
   __cuda_callable__
   StaticVector( const Expressions::StaticUnaryExpressionTemplate< T, Operation >& op );

   /**
    * \brief Sets up a new (vector) parameter which means it can have more elements.
    *
    * @param parameters Reference to a parameter container where the new parameter is saved.
    * @param prefix Name of now parameter/prefix.
    */
   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" );

   /**
    * \brief Gets type of this vector.
    */
   static String getType();

   template< typename VectorExpression >
   StaticVector& operator=( const VectorExpression& expression );

   template< typename VectorExpression >
   __cuda_callable__
   StaticVector& operator+=( const VectorExpression& expression );

   template< typename VectorExpression >
   __cuda_callable__
   StaticVector& operator-=( const VectorExpression& expression );

   template< typename VectorExpression >
   __cuda_callable__
   StaticVector& operator*=( const VectorExpression& expression );

   template< typename VectorExpression >
   __cuda_callable__
   StaticVector& operator/=( const VectorExpression& expression );

   /**
    * \brief Changes the type of static vector to \e OtherReal while the size remains the same.
    *
    * \tparam OtherReal Other type of the static vector values.
    */
   template< typename OtherReal >
   __cuda_callable__
   operator StaticVector< Size, OtherReal >() const;
};

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/StaticVector.hpp>
#include <TNL/Containers/StaticVectorExpressions.h>
#include <TNL/Containers/Expressions/StaticExpressionTemplates.h>

// TODO: move to some other source file
namespace TNL {
namespace Containers {

template< typename Real >
StaticVector< 3, Real > VectorProduct( const StaticVector< 3, Real >& u,
                                       const StaticVector< 3, Real >& v )
{
   StaticVector< 3, Real > p;
   p[ 0 ] = u[ 1 ] * v[ 2 ] - u[ 2 ] * v[ 1 ];
   p[ 1 ] = u[ 2 ] * v[ 0 ] - u[ 0 ] * v[ 2 ];
   p[ 2 ] = u[ 0 ] * v[ 1 ] - u[ 1 ] * v[ 0 ];
   return p;
}

template< typename Real >
Real TriangleArea( const StaticVector< 2, Real >& a,
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
   return 0.5 * TNL::sqrt( dot( v, v ) );
}

template< typename Real >
Real TriangleArea( const StaticVector< 3, Real >& a,
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
   return 0.5 * TNL::sqrt( dot( v, v ) );
}

} // namespace Containers
} // namespace TNL
