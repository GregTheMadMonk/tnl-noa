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

   /**
    * \brief Type of numbers stored in this vector.
    */
   using RealType = Real;

   /**
    * \brief Indexing type
    */
   using IndexType = int;

   /**
    * \brief Default constructor.
    */
   __cuda_callable__
   StaticVector() = default;

   /**
    * \brief Default copy constructor.
    */
   __cuda_callable__
   StaticVector( const StaticVector& ) = default;

   /**
    * \brief Default copy-assignment operator.
    */
   __cuda_callable__
   StaticVector& operator=( const StaticVector& ) = default;

   /**
    * \brief Default move-assignment operator.
    */
   __cuda_callable__
   StaticVector& operator=( StaticVector&& ) = default;

   //! Constructors and assignment operators are inherited from the class \ref StaticArray.
   using StaticArray< Size, Real >::StaticArray;
   using StaticArray< Size, Real >::operator=;

   /**
    * \brief Constructor from binary vector expression.
    * 
    * \param expr is binary expression.
    */
   template< typename T1,
             typename T2,
             template< typename, typename > class Operation >
   __cuda_callable__
   StaticVector( const Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& expr );

   /**
    * \brief Constructor from unary expression.
    * 
    * \param expr is unary expression
    */
   template< typename T,
             template< typename > class Operation >
   __cuda_callable__
   StaticVector( const Expressions::StaticUnaryExpressionTemplate< T, Operation >& expr );

   /**
    * \brief Sets up a new (vector) parameter which means it can have more elements.
    *
    * \param parameters Reference to a parameter container where the new parameter is saved.
    * \param prefix Name of now parameter/prefix.
    */
   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" );

   /**
    * \brief Assignment operator with a vector expression.
    * 
    * The vector expression can be even just static vector.
    * 
    * \param expression is the vector expression
    * \return reference to this vector
    */
   template< typename VectorExpression >
   __cuda_callable__
   StaticVector& operator=( const VectorExpression& expression );

   /**
    * \brief Addition operator with a vector expression
    * 
    * The vector expression can be even just static vector.
    * 
    * \param expression is the vector expression
    * \return reference to this vector
    */
   template< typename VectorExpression >
   __cuda_callable__
   StaticVector& operator+=( const VectorExpression& expression );

   /**
    * \brief Subtraction operator with a vector expression.
    * 
    * The vector expression can be even just static vector.
    * 
    * \param expression is the vector expression
    * \return reference to this vector
    */
   template< typename VectorExpression >
   __cuda_callable__
   StaticVector& operator-=( const VectorExpression& expression );

   /**
    * \brief Elementwise multiplication by a vector expression.
    *
    * The vector expression can be even just static vector.
    * 
    * \param expression is the vector expression.
    * \return reference to this vector
    */
   template< typename VectorExpression >
   __cuda_callable__
   StaticVector& operator*=( const VectorExpression& expression );

   /**
    * \brief Elementwise division by a vector expression.
    * 
    * The vector expression can be even just static vector.
    * 
    * \param expression is the vector expression
    * \return reference to this vector
    */
   template< typename VectorExpression >
   __cuda_callable__
   StaticVector& operator/=( const VectorExpression& expression );

   /**
    * \brief Cast operator for changing of the \e Value type.
    * 
    * Returns static array having \e ValueType set to \e OtherValue, i.e.
    * StaticArray< Size, OtherValue >.
    * 
    * \tparam OtherValue is the \e Value type of the static array the casting 
    * will be performed to.
    * 
    * \return instance of StaticVector< Size, OtherValue >
    */
   template< typename OtherReal >
   __cuda_callable__
   operator StaticVector< Size, OtherReal >() const;
};

// Enable expression templates for StaticVector
namespace Expressions {
   template< int Size, typename Real >
   struct HasEnabledStaticExpressionTemplates< StaticVector< Size, Real > >
   : std::true_type
   {};
} // namespace Expressions

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/StaticVector.hpp>

// TODO: move to some other source file
namespace TNL {
namespace Containers {

template< typename Real >
__cuda_callable__
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
__cuda_callable__
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
__cuda_callable__
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
