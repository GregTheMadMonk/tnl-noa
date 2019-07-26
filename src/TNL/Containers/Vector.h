/***************************************************************************
                          Vector.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/TypeTraits.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/VectorView.h>

namespace TNL {
namespace Containers {

/**
 * \brief \e Vector extends \ref Array with algebraic operations.
 *
 * The template parameters have the same meaning as in \ref Array, with \e Real
 * corresponding to \e Array's \e Value parameter.
 *
 * \tparam Real   A numeric type for the vector values, e.g. \ref float or
 *                \ref double.
 * \tparam Device The device to be used for the execution of vector operations.
 * \tparam Index  The indexing type.
 * \tparam Allocator The type of the allocator used for the allocation and
 *                   deallocation of memory used by the array. By default,
 *                   an appropriate allocator for the specified \e Device
 *                   is selected with \ref Allocators::Default.
 *
 * \par Example
 * \include VectorExample.cpp
 */
template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          typename Allocator = typename Allocators::Default< Device >::template Allocator< Real > >
class Vector
: public Array< Real, Device, Index, Allocator >
{
public:
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using AllocatorType = Allocator;
   using HostType = Vector< Real, TNL::Devices::Host, Index >;
   using CudaType = Vector< Real, TNL::Devices::Cuda, Index >;
   using ViewType = VectorView< Real, Device, Index >;
   using ConstViewType = VectorView< std::add_const_t< Real >, Device, Index >;

   //! \brief Default constructor.
   Vector() = default;
   //! \brief Default copy constructor.
   explicit Vector( const Vector& ) = default;
   //! \brief Copy constructor with a specific allocator.
   explicit Vector( const Vector& vector, const AllocatorType& allocator );
   //! \brief Default move constructor.
   Vector( Vector&& ) = default;
   //! \brief Default copy-assignment operator.
   Vector& operator=( const Vector& ) = default;
   //! \brief Default move-assignment operator.
   Vector& operator=( Vector&& ) = default;

   //! Constructors and assignment operators are inherited from the class \ref Array.
   using Array< Real, Device, Index, Allocator >::Array;
   using Array< Real, Device, Index, Allocator >::operator=;

   /** \brief Returns type of vector Real value, Device type and the type of Index. */
   static String getType();

   /** \brief Returns type of vector Real value, Device type and the type of Index. */
   virtual String getTypeVirtual() const;

   /**
    * \brief Returns a modifiable view of the vector.
    *
    * By default, a view for the whole vector is returned. If \e begin or
    * \e end is set to a non-zero value, a view only for the sub-interval
    * `[begin, end)` is returned.
    *
    * \param begin The beginning of the vector sub-interval. It is 0 by
    *              default.
    * \param end The end of the vector sub-interval. The default value is 0
    *            which is, however, replaced with the array size.
    */
   ViewType getView( IndexType begin = 0, IndexType end = 0 );

   /**
    * \brief Returns a non-modifiable view of the vector.
    *
    * By default, a view for the whole vector is returned. If \e begin or
    * \e end is set to a non-zero value, a view only for the sub-interval
    * `[begin, end)` is returned.
    *
    * \param begin The beginning of the vector sub-interval. It is 0 by
    *              default.
    * \param end The end of the vector sub-interval. The default value is 0
    *            which is, however, replaced with the array size.
    */
   ConstViewType getView( IndexType begin = 0, IndexType end = 0 ) const;

   /**
    * \brief Returns a non-modifiable view of the vector.
    *
    * By default, a view for the whole vector is returned. If \e begin or
    * \e end is set to a non-zero value, a view only for the sub-interval
    * `[begin, end)` is returned.
    *
    * \param begin The beginning of the vector sub-interval. It is 0 by
    *              default.
    * \param end The end of the vector sub-interval. The default value is 0
    *            which is, however, replaced with the array size.
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

   template< typename Real_, typename Device_, typename Index_, typename Allocator_ >
   Vector& operator=( const Vector< Real_, Device_, Index_, Allocator_ >& v );

   template< typename Real_, typename Device_, typename Index_ >
   Vector& operator=( const VectorView< Real_, Device_, Index_ >& v );

   template< typename VectorExpression >
   Vector& operator=( const VectorExpression& expression );

   /**
    * \brief This function subtracts \e vector from this vector and returns the resulting vector.
    *
    * The subtraction is applied to all the vector elements separately.
    * \param vector Reference to another vector.
    */
   template< typename VectorExpression >
   Vector& operator-=( const VectorExpression& expression );

   /**
    * \brief This function adds \e vector to this vector and returns the resulting vector.
    *
    * The addition is applied to all the vector elements separately.
    * \param vector Reference to another vector.
    */
   template< typename VectorExpression >
   Vector& operator+=( const VectorExpression& expression );

   /**
    * \brief This function multiplies this vector by \e c and returns the resulting vector.
    *
    * The multiplication is applied to all the vector elements separately.
    * \param c Multiplicator.
    */
   template< typename VectorExpression >
   Vector& operator*=( const VectorExpression& expression );

   /**
    * \brief This function divides this vector by \e c and returns the resulting vector.
    *
    * The division is applied to all the vector elements separately.
    * \param c Divisor.
    */
   template< typename VectorExpression >
   Vector& operator/=( const VectorExpression& expression );

   /**
    * \brief Returns sum of all vector elements.
    */
   template< typename ResultType = RealType >
   ResultType sum() const;

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
   [[deprecated("addVector is deprecated - use expression templates instead.")]]
   void addVector( const Vector& v,
                   const Scalar1 multiplicator = 1.0,
                   const Scalar2 thisMultiplicator = 1.0 );

   /**
    * \brief Returns the result of following: thisMultiplicator * this + multiplicator1 * v1 + multiplicator2 * v2.
    */
   template< typename Vector1, typename Vector2, typename Scalar1, typename Scalar2, typename Scalar3 = Real >
   [[deprecated("addVectors is deprecated - use expression templates instead.")]]
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
   void prefixSum( IndexType begin = 0, IndexType end = 0 );

   template< Algorithms::PrefixSumType Type = Algorithms::PrefixSumType::Inclusive,
             typename FlagsArray >
   void segmentedPrefixSum( FlagsArray& flags, IndexType begin = 0, IndexType end = 0 );

   template< Algorithms::PrefixSumType Type = Algorithms::PrefixSumType::Inclusive,
             typename VectorExpression >
   void prefixSum( const VectorExpression& expression, IndexType begin = 0, IndexType end = 0 );

   template< Algorithms::PrefixSumType Type = Algorithms::PrefixSumType::Inclusive,
             typename VectorExpression,
             typename FlagsArray >
   void segmentedPrefixSum( const VectorExpression& expression, FlagsArray& flags, IndexType begin = 0, IndexType end = 0 );
};

} // namespace Containers

template< typename Real, typename Device, typename Index >
struct ViewTypeGetter< Containers::Vector< Real, Device, Index > >
{
   using Type = Containers::VectorView< Real, Device, Index >;
};

template< typename Real, typename Device, typename Index >
struct IsStatic< Containers::Vector< Real, Device, Index > >
{
   static constexpr bool Value = false;
};


} // namespace TNL

#include <TNL/Containers/Vector.hpp>
#include <TNL/Containers/VectorExpressions.h>
