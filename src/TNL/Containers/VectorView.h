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

   //! Constructors and assignment operators are inherited from the class \ref Array.
   using ArrayView< Real, Device, Index >::ArrayView;
   using ArrayView< Real, Device, Index >::operator=;

   // In C++14, default constructors cannot be inherited, although Clang
   // and GCC since version 7.0 inherit them.
   // https://stackoverflow.com/a/51854172
   __cuda_callable__
   VectorView() = default;

   // initialization by base class is not a copy constructor so it has to be explicit
   template< typename Real_ >  // template catches both const and non-const qualified Element
   __cuda_callable__
   VectorView( const ArrayView< Real_, Device, Index >& view )
   : BaseType( view ) {}

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

   static String getType();

   // All other Vector methods follow...
   void addElement( IndexType i, RealType value );

   template< typename Scalar >
   void addElement( IndexType i,
                    RealType value,
                    Scalar thisElementMultiplicator );

   template< typename VectorExpression,
             typename...,
             typename = std::enable_if_t< Expressions::IsExpressionTemplate< VectorExpression >::value > >
   VectorView& operator=( const VectorExpression& expression );

   /**
    * \brief Assigns a value or an array - same as \ref ArrayView::operator=.
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

   template< typename VectorExpression >
   VectorView& operator-=( const VectorExpression& expression );

   template< typename VectorExpression >
   VectorView& operator+=( const VectorExpression& expression );

   template< typename VectorExpression >
   VectorView& operator*=( const VectorExpression& expression );

   template< typename VectorExpression >
   VectorView& operator/=( const VectorExpression& expression );

   template< typename ResultType = NonConstReal >
   ResultType sum() const;

   //! Computes scalar dot product
   template< typename Vector >
   NonConstReal scalarProduct( const Vector& v ) const;

   //! Computes this = thisMultiplicator * this + alpha * x.
   template< typename Vector, typename Scalar1 = Real, typename Scalar2 = Real >
   [[deprecated("addVector is deprecated - use expression templates instead.")]]
   void addVector( const Vector& x,
                   Scalar1 alpha = 1.0,
                   Scalar2 thisMultiplicator = 1.0 );

   //! Computes this = thisMultiplicator * this + multiplicator1 * v1 + multiplicator2 * v2.
   template< typename Vector1, typename Vector2, typename Scalar1, typename Scalar2, typename Scalar3 = Real >
   [[deprecated("addVectors is deprecated - use expression templates instead.")]]
   void addVectors( const Vector1& v1,
                    Scalar1 multiplicator1,
                    const Vector2& v2,
                    Scalar2 multiplicator2,
                    Scalar3 thisMultiplicator = 1.0 );

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

#include <TNL/Containers/VectorViewExpressions.h>
#include <TNL/Containers/VectorView.hpp>
