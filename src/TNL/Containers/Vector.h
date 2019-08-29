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
#include <TNL/Containers/VectorView.h>

namespace TNL {
namespace Containers {

/**
 * \brief \e Vector extends \ref Array with algebraic operations.
 *
 * The template parameters have the same meaning as in \ref Array, with \e Real
 * corresponding to \e Array's \e Value parameter.
 *
 * \tparam Real   An arithmetic type for the vector values, e.g. \ref float or
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

   /**
    * \brief Type of elements stored in this vector.
    */
   using RealType = Real;

   /**
    * \brief Device where the vector is allocated.
    * 
    * See \ref Devices::Host or \ref Devices::Cuda.
    */
   using DeviceType = Device;

   /**
    * \brief Type being used for the vector elements indexing.
    */
   using IndexType = Index;

   /**
    * \brief Allocator type used for allocating this vector.
    * 
    * See \ref Allocators::Cuda, \ref Allocators::CudaHost, \ref Allocators::CudaManaged, \ref Allocators::Host or \ref Allocators:Default.
    */
   using AllocatorType = Allocator;

   /**
    * \brief Defines the same vector type but allocated on host (CPU).
    */
   using HostType = Vector< Real, TNL::Devices::Host, Index >;

   /**
    * \brief Defines the same vector type but allocated on CUDA device (GPU).
    */
   using CudaType = Vector< Real, TNL::Devices::Cuda, Index >;
   using ViewType = VectorView< Real, Device, Index >;
   using ConstViewType = VectorView< std::add_const_t< Real >, Device, Index >;

   // constructors and assignment operators inherited from the class Array
   using Array< Real, Device, Index, Allocator >::Array;
   using Array< Real, Device, Index, Allocator >::operator=;

   /**
    * \brief Constructs an empty array with zero size.
    */
   Vector() = default;

   /**
    * \brief Copy constructor (makes a deep copy).
    */
   explicit Vector( const Vector& ) = default;

   /**
    * \brief Copy constructor with a specific allocator (makes a deep copy).
    */
   explicit Vector( const Vector& vector, const AllocatorType& allocator );

   /**
    * \brief Default move constructor.
    */
   Vector( Vector&& ) = default;

   /**
    * \brief Copy-assignment operator for copying data from another vector.
    */
   Vector& operator=( const Vector& ) = default;

   /**
    * \brief Move-assignment operator for acquiring data from \e rvalues.
    */
   Vector& operator=( Vector&& ) = default;

   /**
    * \brief Returns a \ref String representation of the vector type in C++ style.
    */
   static String getType();

   /**
    * \brief Returns a \ref String representation of the vector type in C++ style.
    */
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
    * \brief Assigns a vector expression to this vector.
    *
    * The assignment is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector. If it evaluates to a vector
    * with a different size than this vector, this vector is reallocated to
    * match the size of the vector expression.
    *
    * \param expression The vector expression to be evaluated and assigned to
    *                   this vector.
    * \return Reference to this vector.
    */
   template< typename VectorExpression,
             typename...,
             typename = std::enable_if_t< Expressions::IsExpressionTemplate< VectorExpression >::value > >
   Vector& operator=( const VectorExpression& expression );

   /**
    * \brief Assigns a value or an array - same as \ref Array::operator=.
    *
    * \return Reference to this vector.
    */
   // operator= from the base class should be hidden according to the C++14 standard,
   // although GCC does not do that - see https://stackoverflow.com/q/57322624
   template< typename T,
             typename...,
             typename = std::enable_if_t< std::is_convertible< T, Real >::value || IsArrayType< T >::value > >
   Array< Real, Device, Index, Allocator >&
   operator=( const T& data )
   {
      return Array< Real, Device, Index, Allocator >::operator=(data);
   }

   /**
    * \brief Adds elements of this vector and a vector expression and
    * stores the result in this vector.
    *
    * The addition is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector of the same size as this vector.
    *
    * \param expression Reference to a vector expression.
    * \return Reference to this vector.
    */
   template< typename VectorExpression >
   Vector& operator+=( const VectorExpression& expression );

   /**
    * \brief Subtracts elements of this vector and a vector expression and
    * stores the result in this vector.
    *
    * The subtraction is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector of the same size as this vector.
    *
    * \param expression Reference to a vector expression.
    * \return Reference to this vector.
    */
   template< typename VectorExpression >
   Vector& operator-=( const VectorExpression& expression );

   /**
    * \brief Multiplies elements of this vector and a vector expression and
    * stores the result in this vector.
    *
    * The multiplication is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector of the same size as this vector.
    *
    * \param expression Reference to a vector expression.
    * \return Reference to this vector.
    */
   template< typename VectorExpression >
   Vector& operator*=( const VectorExpression& expression );

   /**
    * \brief Divides elements of this vector and a vector expression and
    * stores the result in this vector.
    *
    * The division is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector of the same size as this vector.
    *
    * \param expression Reference to a vector expression.
    * \return Reference to this vector.
    */
   template< typename VectorExpression >
   Vector& operator/=( const VectorExpression& expression );

   /**
    * \brief Computes prefix sum of the vector elements.
    *
    * Computes prefix sum for elements within the index range [ \e begin to \e end ).
    * The other elements of this vector remain unchanged.
    *
    * \tparam Type tells the prefix sum type - either \e Inclusive of \e Exclusive.
    * 
    * \param begin beginning of the index range
    * \param end end of the index range.
    */
   template< Algorithms::ScanType Type = Algorithms::ScanType::Inclusive >
   void prefixSum( IndexType begin = 0, IndexType end = 0 );

   /**
    * \brief Computes segmented prefix sum of the vector elements.
    *
    * Computes segmented prefix sum for elements within the index range [ \e begin to \e end ).
    * The other elements of this vector remain unchanged. Whole vector is assumed
    * by default, i.e. when \e begin and \e end are set to zero.
    *
    * \tparam Type tells the prefix sum type - either \e Inclusive of \e Exclusive.
    * \tparam FlagsArray is an array type describing beginnings of the segments.
    * 
    * \param flags is an array having `1` at the beginning of each segment and `0` on any other position
    * \param begin beginning of the index range
    * \param end end of the index range.
    */
   template< Algorithms::ScanType Type = Algorithms::ScanType::Inclusive,
             typename FlagsArray >
   void segmentedPrefixSum( FlagsArray& flags, IndexType begin = 0, IndexType end = 0 );

   /**
    * \brief Computes prefix sum of the vector expression.
    *
    * Computes prefix sum for elements within the index range [ \e begin to \e end ).
    * The other elements of this vector remain unchanged. Whole vector expression is assumed
    * by default, i.e. when \e begin and \e end are set to zero.
    *
    * \tparam Type tells the prefix sum type - either \e Inclusive of \e Exclusive.
    * \tparam VectorExpression is the vector expression.
    * 
    * \param expression is the vector expression.
    * \param begin beginning of the index range
    * \param end end of the index range.
    */
   template< Algorithms::ScanType Type = Algorithms::ScanType::Inclusive,
             typename VectorExpression >
   void prefixSum( const VectorExpression& expression, IndexType begin = 0, IndexType end = 0 );

   /**
    * \brief Computes segmented prefix sum of a vector expression.
    *
    * Computes segmented prefix sum for elements within the index range [ \e begin to \e end ).
    * The other elements of this vector remain unchanged. Whole vector expression is assumed
    * by default, i.e. when \e begin and \e end are set to zero.
    *
    * \tparam Type tells the prefix sum type - either \e Inclusive of \e Exclusive.
    * \tparam VectorExpression is the vector expression.
    * \tparam FlagsArray is an array type describing beginnings of the segments.
    * 
    * \param expression is the vector expression.
    * \param flags is an array having `1` at the beginning of each segment and `0` on any other position
    * \param begin beginning of the index range
    * \param end end of the index range.
    */
   template< Algorithms::ScanType Type = Algorithms::ScanType::Inclusive,
             typename VectorExpression,
             typename FlagsArray >
   void segmentedPrefixSum( const VectorExpression& expression, FlagsArray& flags, IndexType begin = 0, IndexType end = 0 );
};

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/Vector.hpp>
#include <TNL/Containers/VectorExpressions.h>
