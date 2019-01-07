/***************************************************************************
                          Vector.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Array.h>

namespace TNL {
namespace Containers {

/**
 * \brief Class for storing vector elements and handling vector operations.
 */
template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int >
class Vector
: public Array< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef Vector< Real, TNL::Devices::Host, Index > HostType;
   typedef Vector< Real, TNL::Devices::Cuda, Index > CudaType;

   /** Constructors and assignment operators are inherited from the class Array. */
   using Array< Real, Device, Index >::Array;
   using Array< Real, Device, Index >::operator=;

   /** \brief Returns type of vector Real value, Device type and the type of Index. */
   static String getType();

   /** \brief Returns type of vector Real value, Device type and the type of Index. */
   virtual String getTypeVirtual() const;

   /** \brief Returns (host) type of vector Real value, Device type and the type of Index. */
   static String getSerializationType();

   /** \brief Returns (host) type of vector Real value, Device type and the type of Index. */
   virtual String getSerializationTypeVirtual() const;

   /**
    * \brief Adds another element to this vector.
    *
    * New element has index type \e i and reference to its real type \e value.
    */
   void addElement( const IndexType i,
                    const RealType& value );

   /**
    * \brief Adds another element with multiplicator to this vector.
    *
    * New element has index type \e i and reference to its real type \e value
    * multiplied by \e thisElementMultiplicator.
    */
   template< typename Scalar >
   void addElement( const IndexType i,
                    const RealType& value,
                    const Scalar thisElementMultiplicator );

   /**
    * \brief This function subtracts \e vector from this vector and returns the resulting vector.
    *
    * The subtraction is applied to all the vector elements separately.
    * \param vector Reference to another vector.
    */
   template< typename VectorT >
   Vector& operator -= ( const VectorT& vector );

   /**
    * \brief This function adds \e vector to this vector and returns the resulting vector.
    *
    * The addition is applied to all the vector elements separately.
    * \param vector Reference to another vector.
    */
   template< typename VectorT >
   Vector& operator += ( const VectorT& vector );

   /**
    * \brief This function multiplies this vector by \e c and returns the resulting vector.
    *
    * The multiplication is applied to all the vector elements separately.
    * \param c Multiplicator.
    */
   template< typename Scalar >
   Vector& operator *= ( const Scalar c );

   /**
    * \brief This function divides this vector by \e c and returns the resulting vector.
    *
    * The division is applied to all the vector elements separately.
    * \param c Divisor.
    */
   template< typename Scalar >
   Vector& operator /= ( const Scalar c );

   /**
    * \brief Returns the maximum value out of all vector elements.
    */
   Real max() const;

   /**
    * \brief Returns the minimum value out of all vector elements.
    */
   Real min() const;

   /**
    * \brief Returns the maximum absolute value out of all vector elements.
    */
   Real absMax() const;

   /**
    * \brief Returns the minimum absolute value out of all vector elements.
    */
   Real absMin() const;

   /**
    * \brief Returns the length of this vector in p-dimensional vector space.
    *
    * \tparam
    * \param p Number specifying the dimension of vector space.
    */
   template< typename ResultType = RealType, typename Scalar >
   ResultType lpNorm( const Scalar p ) const;

   /**
    * \brief Returns sum of all vector elements.
    */
   template< typename ResultType = RealType >
   ResultType sum() const;

   /**
    * \brief Returns maximal difference between elements of this vector and vector \e v.
    *
    * \tparam Vector Type of vector.
    * \param v Reference to another vector of the same size as this vector.
    */
   template< typename Vector >
   Real differenceMax( const Vector& v ) const;

   /**
    * \brief Returns minimal difference between elements of this vector and vector \e v.
    *
    * \tparam Vector Type of vector.
    * \param v Reference to another vector of the same size as this vector.
    */
   template< typename Vector >
   Real differenceMin( const Vector& v ) const;

   /**
    * \brief Returns maximal absolute difference between elements of this vector and vector \e v.
    *
    * \tparam Vector Type of vector.
    * \param v Reference to another vector of the same size as this vector.
    */
   template< typename Vector >
   Real differenceAbsMax( const Vector& v ) const;

   /**
    * \brief Returns minimal absolute difference between elements of this vector and vector \e v.
    *
    * \tparam Vector Type of vector.
    * \param v Reference to another vector of the same size as this vector.
    */
   template< typename Vector >
   Real differenceAbsMin( const Vector& v ) const;

   /**
    * \brief Returns difference between L^p norms of this vector and vector \e v.
    *
    * See also lpNorm(const Real_ p).
    *
    * \param v Reference to another vector.
    * \param p Number specifying the dimension of vector space.
    */
   template< typename ResultType = RealType, typename Vector, typename Scalar >
   ResultType differenceLpNorm( const Vector& v, const Scalar p ) const;

   /**
    * \brief Returns difference between sums of elements of this vector and vector \e v.
    *
    * \param v Reference to another vector.
    */
   template< typename ResultType = RealType, typename Vector >
   ResultType differenceSum( const Vector& v ) const;

   /**
    * \brief Returns this vector multiplied by scalar \e alpha.
    *
    * This function multiplies every element of this vector by scalar \e alpha.
    * \param alpha Reference to a real number.
    */
   template< typename Scalar >
   void scalarMultiplication( const Scalar alpha );

   /**
    * \brief Computes scalar (dot) product.
    *
    * An algebraic operation that takes two equal-length vectors and returns a single number.
    *
    * \tparam vector Type of vector.
    * \param v Reference to another vector of the same size as this vector.
    */
   template< typename Vector >
   Real scalarProduct( const Vector& v ) const;

   /**
    * \brief Returns the result of following: thisMultiplicator * this + multiplicator * v.
    */
   template< typename Vector, typename Scalar1 = Real, typename Scalar2 = Real >
   void addVector( const Vector& v,
                   const Scalar1 multiplicator = 1.0,
                   const Scalar2 thisMultiplicator = 1.0 );

   /**
    * \brief Returns the result of following: thisMultiplicator * this + multiplicator1 * v1 + multiplicator2 * v2.
    */
   template< typename Vector1, typename Vector2, typename Scalar1, typename Scalar2, typename Scalar3 = Real >
   void addVectors( const Vector1& v1,
                    const Scalar1 multiplicator1,
                    const Vector2& v2,
                    const Scalar2 multiplicator2,
                    const Scalar3 thisMultiplicator = 1.0 );

   /**
    * \brief Returns specific sums of elements of this vector.
    *
    * Goes in order from the first element to the last one and for every element
    * in this vector computes sum of all previous elements including the element.
    * Therefore this method returns a new vector with the length of this vector.
    */
   void computePrefixSum();

   /**
    * \brief Returns specific sums of elements of this vector.
    *
    * Does the same as computePrefixSum(), but computes only sums for elements
    * with the index in range from \e begin to \e end. The other elements of this
    * vector remain untouched - with the same value. Therefore this method returns
    * a new vector with the length of this vector.
    *
    * \param begin Index of the element in this vector which to begin with.
    * \param end Index of the element in this vector which to end with.
    */
   void computePrefixSum( const IndexType begin, const IndexType end );

   /**
    * \brief Returns specific sums of elements of this vector.
    *
    * Goes in order from the first element to the last one and for every element
    * in this vector computes sum of all previous elements excluding the element.
    * Therefore returns a new vector with the length of this vector.
    */
   void computeExclusivePrefixSum();

   /**
    * \brief Returns specific sums of elements of this vector.
    *
    * Does the same as computeExclusivePrefixSum(), but computes only sums for elements
    * with the index in range from \e begin to \e end. The other elements of this
    * vector remain untouched - with the same value. Therefore this method returns
    * a new vector with the length of this vector.
    *
    * \param begin Index of the element in this vector which to begin with.
    * \param end Index of the element in this vector which to end with.
    */
   void computeExclusivePrefixSum( const IndexType begin, const IndexType end );
};

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/Vector_impl.h>
