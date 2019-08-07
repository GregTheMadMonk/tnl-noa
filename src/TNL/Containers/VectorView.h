/***************************************************************************
                          VectorView.h  -  description
                             -------------------
    begin                : Sep 1, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/Containers/ArrayView.h>
#include <TNL/Containers/Expressions/ExpressionTemplates.h>
#include <TNL/Containers/Algorithms/PrefixSumType.h>

namespace TNL {
namespace Containers {

/**
 * \brief \e VectorView extends \ref ArrayView with algebraic operations.
 *
 * The template parameters have the same meaning as in \ref ArrayView, with
 * \e Real corresponding to \e ArrayView's \e Value parameter.
 *
 * \tparam Real   An arithmetic type for the vector values, e.g. \ref float or
 *                \ref double.
 * \tparam Device The device to be used for the execution of vector operations.
 * \tparam Index  The indexing type.
 */
template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int >
class VectorView
: public ArrayView< Real, Device, Index >
{
   using BaseType = ArrayView< Real, Device, Index >;
   using NonConstReal = typename std::remove_const< Real >::type;
public:
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using HostType = VectorView< Real, Devices::Host, Index >;
   using CudaType = VectorView< Real, Devices::Cuda, Index >;
   using ViewType = VectorView< Real, Device, Index >;
   using ConstViewType = VectorView< std::add_const_t< Real >, Device, Index >;

   // constructors and assignment operators inherited from the class ArrayView
   using ArrayView< Real, Device, Index >::ArrayView;
   using ArrayView< Real, Device, Index >::operator=;

   // In C++14, default constructors cannot be inherited, although Clang
   // and GCC since version 7.0 inherit them.
   // https://stackoverflow.com/a/51854172
   //! \brief Constructs an empty array view with zero size.
   __cuda_callable__
   VectorView() = default;

   //! \brief Constructor for the initialization by a base class object.
   // initialization by base class is not a copy constructor so it has to be explicit
   template< typename Real_ >  // template catches both const and non-const qualified Element
   __cuda_callable__
   VectorView( const ArrayView< Real_, Device, Index >& view )
   : BaseType( view ) {}

   //! \brief Returns a \ref String representation of the vector view type.
   static String getType();

   /**
    * \brief Returns a modifiable view of the vector view.
    *
    * By default, a view for the whole vector is returned. If \e begin or
    * \e end is set to a non-zero value, a view only for the sub-interval
    * `[begin, end)` is returned.
    *
    * \param begin The beginning of the vector view sub-interval. It is 0 by
    *              default.
    * \param end The end of the vector view sub-interval. The default value is 0
    *            which is, however, replaced with the array size.
    */
   __cuda_callable__
   ViewType getView( IndexType begin = 0, IndexType end = 0 );

   /**
    * \brief Returns a non-modifiable view of the vector view.
    *
    * By default, a view for the whole vector is returned. If \e begin or
    * \e end is set to a non-zero value, a view only for the sub-interval
    * `[begin, end)` is returned.
    *
    * \param begin The beginning of the vector view sub-interval. It is 0 by
    *              default.
    * \param end The end of the vector view sub-interval. The default value is 0
    *            which is, however, replaced with the array size.
    */
   __cuda_callable__
   ConstViewType getConstView( IndexType begin = 0, IndexType end = 0 ) const;

   /**
    * \brief Assigns a vector expression to this vector view.
    *
    * The assignment is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector of the same size as this vector
    * view.
    *
    * \param expression The vector expression to be evaluated and assigned to
    *                   this vector view.
    * \return Reference to this vector view.
    */
   template< typename VectorExpression,
             typename...,
             typename = std::enable_if_t< Expressions::IsExpressionTemplate< VectorExpression >::value > >
   VectorView& operator=( const VectorExpression& expression );

   /**
    * \brief Assigns a value or an array - same as \ref ArrayView::operator=.
    *
    * \return Reference to this vector view.
    */
   // operator= from the base class should be hidden according to the C++14 standard,
   // although GCC does not do that - see https://stackoverflow.com/q/57322624
   template< typename T,
             typename...,
             typename = std::enable_if_t< std::is_convertible< T, Real >::value || IsArrayType< T >::value > >
   ArrayView< Real, Device, Index >&
   operator=( const T& data )
   {
      return ArrayView< Real, Device, Index >::operator=(data);
   }

   /**
    * \brief Adds elements of this vector view and a vector expression and
    * stores the result in this vector view.
    *
    * The addition is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector of the same size as this vector
    * view.
    *
    * \param expression Reference to a vector expression.
    * \return Reference to this vector view.
    */
   template< typename VectorExpression >
   VectorView& operator+=( const VectorExpression& expression );

   /**
    * \brief Subtracts elements of this vector view and a vector expression and
    * stores the result in this vector view.
    *
    * The subtraction is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector of the same size as this vector
    * view.
    *
    * \param expression Reference to a vector expression.
    * \return Reference to this vector view.
    */
   template< typename VectorExpression >
   VectorView& operator-=( const VectorExpression& expression );

   /**
    * \brief Multiplies elements of this vector view and a vector expression and
    * stores the result in this vector view.
    *
    * The multiplication is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector of the same size as this vector
    * view.
    *
    * \param expression Reference to a vector expression.
    * \return Reference to this vector view.
    */
   template< typename VectorExpression >
   VectorView& operator*=( const VectorExpression& expression );

   /**
    * \brief Divides elements of this vector view and a vector expression and
    * stores the result in this vector view.
    *
    * The division is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector of the same size as this vector
    * view.
    *
    * \param expression Reference to a vector expression.
    * \return Reference to this vector view.
    */
   template< typename VectorExpression >
   VectorView& operator/=( const VectorExpression& expression );

   /**
    * \brief Returns sum of all vector elements.
    */
   template< typename ResultType = NonConstReal >
   ResultType sum() const;

   /**
    * \brief Returns specific sums of elements of this vector view.
    *
    * FIXME: computePrefixSum does not exist
    *
    * Does the same as \ref computePrefixSum, but computes only sums for elements
    * with the index in range from \e begin to \e end. The other elements of this
    * vector remain untouched - with the same value.
    *
    * \param begin Index of the element in this vector view which to begin with.
    * \param end Index of the element in this vector view which to end with.
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
} // namespace TNL

#include <TNL/Containers/VectorView.hpp>
#include <TNL/Containers/VectorViewExpressions.h>
