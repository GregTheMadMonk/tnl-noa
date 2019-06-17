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
#include <TNL/Containers/Algorithms/PrefixSumType.h>

namespace TNL {
namespace Containers {

template< typename Real, typename Device, typename Index >
class VectorView;

/**
 * \brief This class extends TNL::Array with algebraic operations.
 *
 * \tparam Real is numeric type usually float or double.
 * \tparam Device is device where the array is going to be allocated - some of \ref Devices::Host and \ref Devices::Cuda.
 * \tparam Index is indexing type.
 *
 * \par Example
 * \include VectorExample.cpp
 */
template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int >
class Vector
: public Array< Real, Device, Index >
{
public:
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using HostType = Vector< Real, TNL::Devices::Host, Index >;
   using CudaType = Vector< Real, TNL::Devices::Cuda, Index >;
   using ViewType = VectorView< Real, Device, Index >;
   using ConstViewType = VectorView< std::add_const_t< Real >, Device, Index >;

   /** Subscript operator is inherited from the class \ref Array. */
   using Array< Real, Device, Index >::operator[];

   /**
    * \brief Basic constructor.
    *
    * Constructs an empty vector with zero size.
    */
   Vector();

   /**
    * \brief Constructor with vector size.
    *
    * \param size is number of vector elements.
    */
   Vector( const IndexType& size );

   /**
    * \brief Constructor with data pointer and size.
    *
    * In this case, the Vector just encapsulates the pointer \e data. No
    * deallocation is done in destructor.
    *
    * This behavior of the Vector is deprecated and \ref VectorView should be used
    * instead.
    *
    * \param data Pointer to data.
    * \param size Number of vector elements.
    */
   Vector( Real* data,
          const IndexType& size );

   /**
    * \brief Copy constructor.
    *
    * \param vector is an vector to be copied.
    */
   explicit Vector( const Vector& vector );

   template< typename Real_, typename Device_, typename Index_ >
   Vector( const Vector< Real_, Device_, Index_ >& vector );

   /**
    * \brief Bind constructor .
    *
    * The constructor does not make a deep copy, but binds to the supplied vector.
    * This is also deprecated, \ref VectorView should be used instead.
    *
    * \param vector is an vector that is to be bound.
    * \param begin is the first index which should be bound.
    * \param size is number of array elements that should be bound.
    */
   Vector( Vector& vector,
           const IndexType& begin = 0,
           const IndexType& size = 0 );

   /**
    * \brief Move constructor.
    *
    * @param vector is an vector to be moved
    */
   Vector( Vector&& vector );

   /**
    * \brief Initialize the vector from initializer list, i.e. { ... }
    *
    * @param list Initializer list.
    */
   Vector( const std::initializer_list< Real >& list );

   /**
    * \brief Initialize the vector from std::list.
    *
    * @param list Input STL list.
    */
   template< typename InReal >
   Vector( const std::list< InReal >& list );

   /**
    * \brief Initialize the vector from std::vector.
    *
    * @param vector Input STL vector.
    */
   template< typename InReal >
   Vector( const std::vector< InReal >& vector );

   /** \brief Returns type of vector Real value, Device type and the type of Index. */
   static String getType();

   /** \brief Returns type of vector Real value, Device type and the type of Index. */
   virtual String getTypeVirtual() const;

   /**
    * \brief Returns a modifiable view of the vector.
    *
    * If \e begin and \e end is set, view for sub-interval [ \e begin, \e end )
    * is returned.
    *
    * \param begin is the beginning of the Vector sub-interval, 0 by default.
    * \param end is the end of the Vector sub-interval. Default value is 0 which is,
    * however, replaced with the Vector size.
    */
   ViewType getView( IndexType begin = 0, IndexType end = 0 );

   /**
    * \brief Returns a non-modifiable view of the vector.
    *
    * If \e begin and \e end is set, view for sub-interval [ \e begin, \e end )
    * is returned.
    *
    * \param begin is the beginning of the sub-interval, 0 by default.
    * \param end is the end of the sub-interval. Default value is 0 which is,
    * however, replaced with the Vector size.
    */
   ConstViewType getConstView( IndexType begin = 0, IndexType end = 0 ) const;

   /**
    * \brief Conversion operator to a modifiable view of the vector.
    */
   operator ViewType();

   /**
    * \brief Conversion operator to a non-modifiable view of the vector.
    */
   operator ConstViewType() const;

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

   __cuda_callable__ Real& operator[]( const Index& i );

   __cuda_callable__ const Real& operator[]( const Index& i ) const;

   Vector& operator = ( const Vector& v );

   template< typename Real_, typename Device_, typename Index_ >
   Vector& operator = ( const Vector< Real_, Device_, Index_ >& v );

   template< typename Real_, typename Device_, typename Index_ >
   Vector& operator = ( const VectorView< Real_, Device_, Index_ >& v );

   template< typename VectorExpression >
   Vector& operator = ( const VectorExpression& expression );

   /**
    * \brief This function subtracts \e vector from this vector and returns the resulting vector.
    *
    * The subtraction is applied to all the vector elements separately.
    * \param vector Reference to another vector.
    */
   template< typename VectorExpression >
   Vector& operator -= ( const VectorExpression& expression );

   /**
    * \brief This function adds \e vector to this vector and returns the resulting vector.
    *
    * The addition is applied to all the vector elements separately.
    * \param vector Reference to another vector.
    */
   template< typename VectorExpression >
   Vector& operator += ( const VectorExpression& expression );

   /**
    * \brief This function multiplies this vector by \e c and returns the resulting vector.
    *
    * The multiplication is applied to all the vector elements separately.
    * \param c Multiplicator.
    */
   template< typename VectorExpression >
   Vector& operator *= ( const VectorExpression& expression );

   /**
    * \brief This function divides this vector by \e c and returns the resulting vector.
    *
    * The division is applied to all the vector elements separately.
    * \param c Divisor.
    */
   template< typename VectorExpression >
   Vector& operator /= ( const VectorExpression& expression );

   /**
    * \brief Scalar product
    * @param v
    * @return
    */
   template< typename Vector_ >
   Real operator, ( const Vector_& v ) const;

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
    * See also \ref lpNorm.
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
    * Does the same as \ref computePrefixSum, but computes only sums for elements
    * with the index in range from \e begin to \e end. The other elements of this
    * vector remain untouched - with the same value. Therefore this method returns
    * a new vector with the length of this vector.
    *
    * \param begin Index of the element in this vector which to begin with.
    * \param end Index of the element in this vector which to end with.
    */
   template< Algorithms::PrefixSumType Type = Algorithms::PrefixSumType::Inclusive >
   void prefixSum( const IndexType begin = 0, const IndexType end = 0 );

   template< Algorithms::PrefixSumType Type = Algorithms::PrefixSumType::Inclusive,
             typename FlagsArray >
   void segmentedPrefixSum( FlagsArray& flags, const IndexType begin = 0, const IndexType end = 0 );

   template< Algorithms::PrefixSumType Type = Algorithms::PrefixSumType::Inclusive,
             typename VectorExpression >
   void prefixSum( const VectorExpression& expression, const IndexType begin = 0, const IndexType end = 0 );

   template< Algorithms::PrefixSumType Type = Algorithms::PrefixSumType::Inclusive,
             typename VectorExpression,
             typename FlagsArray >
   void segmentedPrefixSum( const VectorExpression& expression, FlagsArray& flags, const IndexType begin = 0, const IndexType end = 0 );
};

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/Vector.hpp>
