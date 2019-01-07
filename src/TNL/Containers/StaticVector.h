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

namespace TNL {
namespace Containers {   

 /**
 * \brief Vector with constant size.
 *
 * \param Size Size of static array. Number of its elements.
 * \param Real Type of the values in the static vector.
 */
template< int Size, typename Real = double >
class StaticVector : public StaticArray< Size, Real >
{
   public:
   typedef Real RealType;
   typedef StaticVector< Size, Real > ThisType;
   enum { size = Size };

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

   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" );      

   /**
    * \brief Gets type of this vector.
    */
   static String getType();

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
    * This function divides this static vector by \e c and returns the resulting static vector.
    * The division is applied to all the vector elements separately.
    * \param c Divisor.
    */
   __cuda_callable__
   StaticVector& operator /= ( const Real& c );
   
   /**
    * \brief Addition operator.
    *
    * This function adds static vector \e u to this static vector and returns the resulting static vector.
    * The addition is applied to all the vector elements separately.
    * \param u Reference to another static vector.
    */
   __cuda_callable__
   StaticVector operator + ( const StaticVector& u ) const;

   /**
    * \brief Subtraction operator.
    *
    * This function subtracts static vector \e u from this static vector and returns the resulting static vector.
    * The subtraction is applied to all the vector elements separately.
    * \param u Reference to another static vector.
    */
   __cuda_callable__
   StaticVector operator - ( const StaticVector& u ) const;

   /**
    * \brief Multiplication by number.
    *
    * This function multipies this static vector by \e c and returns the resulting static vector.
    * The addition is applied to all the vector elements separately.
    * \param c Multiplicator.
    */
   __cuda_callable__
   StaticVector operator * ( const Real& c ) const;

   /**
    * \brief Computes scalar (dot) product.
    *
    * An algebraic operation that takes two equal-length vectors and returns a single number.
    *
    * \param u Reference to another static vector of the same size as this static vector.
    */
   __cuda_callable__
   Real operator * ( const StaticVector& u ) const;

   /**
    * \brief Compares this static vector with static vector \e v.
    *
    * Returns \e true if this static vector is smaller then static vector \e v.
    * \param v Another static vector.
    */
   __cuda_callable__
   bool operator < ( const StaticVector& v ) const;

   /**
    * \brief Compares this static vector with static vector \e v.
    *
    * Returns \e true if this static vector is smaller then or equal to static vector \e v.
    * \param v Another static vector.
    */
   __cuda_callable__
   bool operator <= ( const StaticVector& v ) const;

   /**
    * \brief Compares this static vector with static vector \e v.
    *
    * Returns \e true if this static vector is greater then static vector \e v.
    * \param v Another static vector.
    */
   __cuda_callable__
   bool operator > ( const StaticVector& v ) const;

   /**
    * \brief Compares this static vector with static vector \e v.
    *
    * Returns \e true if this static vector is greater then or equal to static vector \e v.
    * \param v Another static vector.
    */
   __cuda_callable__
   bool operator >= ( const StaticVector& v ) const;

   template< typename OtherReal >
   __cuda_callable__
   operator StaticVector< Size, OtherReal >() const;

   /**
    * \brief Returns static vector with all elements in absolute value.
    */
   __cuda_callable__
   ThisType abs() const;

   /**
    * \brief Returns the length of this vector in p-dimensional vector space.
    *
    * \param p Number specifying the dimension of vector space.
    */
   __cuda_callable__
   Real lpNorm( const Real& p ) const;

#ifdef HAVE_MIC
   __cuda_callable__
   inline StaticVector< Size, Real >& operator=( const StaticVector< Size, Real >& vector )
   {
      StaticArray< Size, Real >::operator=( vector );
      return *this;
   }

   template< typename Vector >
   __cuda_callable__
   inline StaticVector< Size, Real >& operator=( const Vector& vector )
   {
      StaticArray< Size, Real >::operator=( vector );
      return *this;
   }
#endif
};

/**
 * \brief Specific static vector with the size of 1. Works like the class StaticVector.
 */
template< typename Real >
class StaticVector< 1, Real > : public StaticArray< 1, Real >
{
   public:
   typedef Real RealType;
   typedef StaticVector< 1, Real > ThisType;
   enum { size = 1 };

   /** \brief See StaticVector::StaticVector().*/
   __cuda_callable__
   StaticVector();

   /** \brief See StaticVector::StaticVector(const Real v[Size]).*/
   // Note: the template avoids ambiguity of overloaded functions with literal 0 and pointer
   // reference: https://stackoverflow.com/q/4610503
   template< typename _unused = void >
   __cuda_callable__
   StaticVector( const Real v[ 1 ] );

   /** \brief See StaticVector::StaticVector( const Real& v ).*/
   __cuda_callable__
   StaticVector( const Real& v );

   /** \brief See StaticVector::StaticVector( const StaticVector< Size, Real >& v ).*/
   __cuda_callable__
   StaticVector( const StaticVector< 1, Real >& v );
   
   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" );      

   /** \brief See StaticVector::getType().*/
   static String getType();

   /** \brief See StaticVector::operator += ( const StaticVector& v ).*/
   __cuda_callable__
   StaticVector& operator += ( const StaticVector& v );

   /** \brief See StaticVector::operator -= ( const StaticVector& v ).*/
   __cuda_callable__
   StaticVector& operator -= ( const StaticVector& v );

   /** \brief See StaticVector::operator *= ( const Real& c ).*/
   __cuda_callable__
   StaticVector& operator *= ( const Real& c );
   
   /** \brief See StaticVector::operator *= ( const Real& c ).*/
   __cuda_callable__
   StaticVector& operator /= ( const Real& c );   

   /** \brief See StaticVector::operator + ( const StaticVector& u ) const.*/
   __cuda_callable__
   StaticVector operator + ( const StaticVector& u ) const;

   /** \brief See StaticVector::operator - ( const StaticVector& u ) const.*/
   __cuda_callable__
   StaticVector operator - ( const StaticVector& u ) const;

   /** \brief See StaticVector::operator * ( const Real& c ) const.*/
   __cuda_callable__
   StaticVector operator * ( const Real& c ) const;

   /** \brief See StaticVector::operator * ( const StaticVector& u ) const.*/
   __cuda_callable__
   Real operator * ( const StaticVector& u ) const;

   /** \brief See StaticVector::operator <.*/
   __cuda_callable__
   bool operator < ( const StaticVector& v ) const;

   /** \brief See StaticVector::operator <=.*/
   __cuda_callable__
   bool operator <= ( const StaticVector& v ) const;

   /** \brief See StaticVector::operator <.*/
   __cuda_callable__
   bool operator > ( const StaticVector& v ) const;

   /** \brief See StaticVector::operator <=.*/
   __cuda_callable__
   bool operator >= ( const StaticVector& v ) const;

   template< typename OtherReal >
   __cuda_callable__
   operator StaticVector< 1, OtherReal >() const;

   /** \brief See StaticVector::abs() const.*/
   __cuda_callable__
   ThisType abs() const;

   /** \brief See StaticVector::lpNorm( const Real& p ) const.*/
   __cuda_callable__
   Real lpNorm( const Real& p ) const;   

#ifdef HAVE_MIC
   __cuda_callable__
   inline StaticVector< 1, Real >& operator=( const StaticVector< 1, Real >& vector )
   {
      StaticArray< 1, Real >::operator=( vector );
      return *this;
   }

   template< typename Vector >
   __cuda_callable__
   inline StaticVector< 1, Real >& operator=( const Vector& vector )
   {
      StaticArray< 1, Real >::operator=( vector );
      return *this;
   }
#endif
};

template< typename Real >
class StaticVector< 2, Real > : public StaticArray< 2, Real >
{
   public:
   typedef Real RealType;
   typedef StaticVector< 2, Real > ThisType;
   enum { size = 2 };

   __cuda_callable__
   StaticVector();

   // Note: the template avoids ambiguity of overloaded functions with literal 0 and pointer
   // reference: https://stackoverflow.com/q/4610503
   template< typename _unused = void >
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

   //! Division by number
   __cuda_callable__
   StaticVector& operator /= ( const Real& c );   

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

#ifdef HAVE_MIC
   __cuda_callable__
   inline StaticVector< 2, Real >& operator=( const StaticVector< 2, Real >& vector )
   {
      StaticArray< 2, Real >::operator=( vector );
      return *this;
   }

   template< typename Vector >
   __cuda_callable__
   inline StaticVector< 2, Real >& operator=( const Vector& vector )
   {
      StaticArray< 2, Real >::operator=( vector );
      return *this;
   }
#endif
};

template< typename Real >
class StaticVector< 3, Real > : public StaticArray< 3, Real >
{
   public:
   typedef Real RealType;
   typedef StaticVector< 3, Real > ThisType;
   enum { size = 3 };

   __cuda_callable__
   StaticVector();

   // Note: the template avoids ambiguity of overloaded functions with literal 0 and pointer
   // reference: https://stackoverflow.com/q/4610503
   template< typename _unused = void >
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
   
   //! Division by number
   __cuda_callable__
   StaticVector& operator /= ( const Real& c );
   

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

#ifdef HAVE_MIC
   __cuda_callable__
   inline StaticVector< 3, Real >& operator=( const StaticVector< 3, Real >& vector )
   {
      StaticArray< 3, Real >::operator=( vector );
      return *this;
   }

   template< typename Vector >
   __cuda_callable__
   inline StaticVector< 3, Real >& operator=( const Vector& vector )
   {
      StaticArray< 3, Real >::operator=( vector );
      return *this;
   }
#endif
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
}

template< typename Real >
Real ScalarProduct( const StaticVector< 2, Real >& u,
                    const StaticVector< 2, Real >& v )
{
   return u[ 0 ] * v[ 0 ] + u[ 1 ] * v[ 1 ];
}

template< typename Real >
Real ScalarProduct( const StaticVector< 3, Real >& u,
                    const StaticVector< 3, Real >& v )
{
   return u[ 0 ] * v[ 0 ] + u[ 1 ] * v[ 1 ] + u[ 2 ] * v[ 2 ];
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
} // namespace TNL
