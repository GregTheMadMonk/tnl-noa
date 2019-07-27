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
   using ThisType = StaticVector< Size, Real >;

   constexpr static int size = Size;

   using StaticArray< Size, Real >::getSize;
   //using StaticArray< Size, Real >::operator ==;
   //using StaticArray< Size, Real >::operator !=;

   /**
    * \brief Basic constructor.
    *
    * Constructs an empty static vector.
    */
   __cuda_callable__
   StaticVector();

   /**
    * \brief Constructor that sets all vector components (with the number of \e Size) to value \e v.
    *
    * Once this static array is constructed, its size can not be changed.
    * \tparam _unused
    * \param v[Size]
    */
   // Note: the template avoids ambiguity of overloaded functions with literal 0 and pointer
   // reference: https://stackoverflow.com/q/4610503
   template< typename _unused = void >
   __cuda_callable__
   StaticVector( const Real v[ Size ] );

   /**
    * \brief Constructor that sets all vector components to value \e v.
    *
    * \param v Reference to a value.
    */
   __cuda_callable__
   StaticVector( const Real& v );

   /**
    * \brief Copy constructor.
    *
    * Constructs a copy of another static vector \e v.
    */
   __cuda_callable__
   StaticVector( const StaticVector< Size, Real >& v );

   StaticVector( const std::initializer_list< Real > &elems );

   /**
    * \brief Constructor that sets components of arrays with Size = 2.
    *
    * \param v1 Real of the first array component.
    * \param v2 Real of the second array component.
    */
   __cuda_callable__
   inline StaticVector( const Real& v1, const Real& v2 );

   /**
    * \brief Constructor that sets components of arrays with Size = 3.
    *
    * \param v1 Real of the first array component.
    * \param v2 Real of the second array component.
    * \param v3 Real of the third array component.
    */
   __cuda_callable__
   inline StaticVector( const Real& v1, const Real& v2, const Real& v3 );


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

   template< typename StaticVectorOperationType >
   StaticVector& operator = ( const StaticVectorOperationType& vo );

   /**
    * \brief Adding operator.
    *
    * This function adds \e vector from this static vector and returns the resulting static vector.
    * The addition is applied to all the vector elements separately.
    * \param vector Reference to another vector.
    */
   __cuda_callable__
   StaticVector& operator += ( const StaticVector& v );

   /**
    * \brief Subtracting operator.
    *
    * This function subtracts \e vector from this static vector and returns the resulting static vector.
    * The subtraction is applied to all the vector elements separately.
    * \param vector Reference to another vector.
    */
   __cuda_callable__
   StaticVector& operator -= ( const StaticVector& v );

   /**
    * \brief Multiplication by number.
    *
    * This function multiplies this static vector by \e c and returns the resulting static vector.
    * The multiplication is applied to all the vector elements separately.
    * \param c Multiplicator.
    */
   __cuda_callable__
   StaticVector& operator *= ( const Real& c );

   /**
    * \brief Division by number
    *
    * This function divides this static veSize of static array. Number of its elements.ctor by \e c and returns the resulting static vector.
    * The division is applied to all the vector elements separately.
    * \param c Divisor.
    */
   __cuda_callable__
   StaticVector& operator /= ( const Real& c );

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


namespace TNL {
namespace Containers {
// TODO: move to some other source file

template< int Size, typename Real1, typename Real2 >
struct StaticScalarProductGetter
{
   __cuda_callable__
   static auto compute( const Real1* u, const Real2* v ) -> decltype( u[ 0 ] * v[ 0 ] )
   {
      return u[ 0 ] * v[ 0 ] + StaticScalarProductGetter< Size - 1, Real1, Real2 >::compute( &u[ 1 ], &v[ 1 ] );
   }
};

template< typename Real1, typename Real2 >
struct StaticScalarProductGetter< 1, Real1, Real2 >
{
   __cuda_callable__
   static auto compute( const Real1* u, const Real2* v ) -> decltype( u[ 0 ] * v[ 0 ] )
   {
      return u[ 0 ] * v[ 0 ];
   }
};

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
auto ScalarProduct( const StaticVector< Size, Real1 >& u,
                    const StaticVector< Size, Real2 >& v ) -> decltype( u[ 0 ] * v[ 0 ] )
{
   return StaticScalarProductGetter< Size, Real1, Real2 >::compute( u.getData(), v.getData() );
}

template< int Size, typename Real1, typename Real2 >
__cuda_callable__
auto operator,( const StaticVector< Size, Real1 >& u,
                    const StaticVector< Size, Real2 >& v ) -> decltype( u[ 0 ] * v[ 0 ] )
{
   return StaticScalarProductGetter< Size, Real1, Real2 >::compute( u.getData(), v.getData() );
}


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

template< typename T1,
          typename T2>
StaticVector<1, T1> Scale( const StaticVector< 1, T1 >& u,
                           const StaticVector< 1, T2 >& v )
{
   StaticVector<1, T1> ret;
   ret[0]=u[0]*v[0];
   return ret;
}

template< typename T1,
          typename T2>
StaticVector<2, T1> Scale( const StaticVector< 2, T1 >& u,
                           const StaticVector< 2, T2 >& v )
{
   StaticVector<2, T1> ret;
   ret[0]=u[0]*v[0];
   ret[1]=u[1]*v[1];
   return ret;
}

template< typename T1,
          typename T2>
StaticVector<3, T1> Scale( const StaticVector< 3, T1 >& u,
                           const StaticVector< 3, T2 >& v )
{
   StaticVector<3, T1> ret;
   ret[0]=u[0]*v[0];
   ret[1]=u[1]*v[1];
   ret[2]=u[2]*v[2];
   return ret;
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
   return 0.5 * TNL::sqrt( tnlScalarProduct( v, v ) );
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
   return 0.5 * TNL::sqrt( ScalarProduct( v, v ) );
}

} // namespace Containers

template< int Size, typename Real >
struct IsStatic< Containers::StaticVector< Size, Real > >
{
   static constexpr bool Value = true;
};

} // namespace TNL

#include <TNL/Containers/StaticVectorExpressions.h>
#include <TNL/Containers/Expressions/StaticExpressionTemplates.h>
