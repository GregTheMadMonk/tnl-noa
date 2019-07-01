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

template< typename Real, typename Device, typename Index >
class Vector;

template< int Size, typename Real >
class StaticVector;

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
   : BaseType::ArrayView( view ) {}

   template< typename T1,
             typename T2,
             template< typename, typename > class Operation >
   __cuda_callable__
   VectorView( const Expressions::BinaryExpressionTemplate< T1, T2, Operation >& expression );

   template< typename T,
             template< typename > class Operation >
   __cuda_callable__
   VectorView( const Expressions::UnaryExpressionTemplate< T, Operation >& expression );

   /**
    * \brief Returns a modifiable view of the vector view.
    *
    * If \e begin and \e end is set, view for sub-interval [ \e begin, \e end )
    * is returned.
    *
    * \param begin is the beginning of the VectorView sub-interval, 0 by default.
    * \param end is the end of the VectorView sub-interval. Default value is 0 which is,
    * however, replaced with the VectorView size.
    */
   __cuda_callable__
   ViewType getView( IndexType begin = 0, IndexType end = 0 );

   /**
    * \brief Returns a non-modifiable view of the vector view.
    *
    * If \e begin and \e end is set, view for sub-interval [ \e begin, \e end )
    * is returned.
    *
    * \param begin is the beginning of the sub-interval, 0 by default.
    * \param end is the end of the sub-interval. Default value is 0 which is,
    * however, replaced with the VectorView size.
    */
   __cuda_callable__
   ConstViewType getConstView( IndexType begin = 0, IndexType end = 0 ) const;

   /**
    * \brief Returns a non-modifiable view of the vector view.
    *
    * If \e begin and \e end is set, view for sub-interval [ \e begin, \e end )
    * is returned.
    *
    * \param begin is the beginning of the sub-interval, 0 by default.
    * \param end is the end of the sub-interval. Default value is 0 which is,
    * however, replaced with the VectorView size.
    */
   __cuda_callable__
   ConstViewType getConstView( const IndexType begin = 0, IndexType end = 0 ) const;


   static String getType();

   //template< typename VectorOperationType >
   //void evaluate( const VectorOperationType& vo );

   template< typename VectorOperationType >
   void evaluateFor( const VectorOperationType& vo );

   // All other Vector methods follow...
   void addElement( IndexType i, RealType value );

   template< typename Scalar >
   void addElement( IndexType i,
                    RealType value,
                    Scalar thisElementMultiplicator );

   template< typename Real_, typename Device_, typename Index_ >
   VectorView& operator=( const VectorView< Real_, Device_, Index_ >& v );

   template< typename Real_, typename Device_, typename Index_ >
   VectorView& operator=( const Vector< Real_, Device_, Index_ >& v );

   template< typename VectorExpression >
   VectorView& operator=( const VectorExpression& expression );

   template< typename VectorExpression >
   VectorView& operator-=( const VectorExpression& expression );

   template< typename VectorExpression >
   VectorView& operator+=( const VectorExpression& expression );

   template< typename VectorExpression >
   VectorView& operator*=( const VectorExpression& expression );

   template< typename VectorExpression >
   VectorView& operator/=( const VectorExpression& expression );

   /**
    * \brief Scalar product
    * @param v
    * @return
    */
   template< typename Vector_ >
   NonConstReal operator,( const Vector_& v ) const;

   NonConstReal max() const;

   NonConstReal min() const;

   NonConstReal absMax() const;

   NonConstReal absMin() const;

   template< typename ResultType = NonConstReal, typename Scalar >
   ResultType lpNorm( Scalar p ) const;

   template< typename ResultType = NonConstReal >
   ResultType sum() const;

   template< typename Scalar >
   void scalarMultiplication( Scalar alpha );

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

template< typename Real, typename Device, typename Index >
struct IsStatic< Containers::VectorView< Real, Device, Index > >
{
   static constexpr bool Value = false;
};

} // namespace TNL

#include <TNL/Containers/VectorView.hpp>
#include <TNL/Containers/VectorViewExpressions.h>
