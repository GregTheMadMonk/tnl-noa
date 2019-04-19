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

   // inherit all ArrayView's constructors
#ifndef __NVCC__
   using BaseType::ArrayView;
#else
   // workaround for nvcc 8.0, otherwise the templated constructor below fails
   // (works fine in nvcc 9.0)
   using ArrayView< Real, Device, Index >::ArrayView;
#endif

   // inherit all ArrayView's assignment operators
   using BaseType::operator=;

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

   /**
    * \brief Returns a modifiable view of the array view.
    */
   __cuda_callable__
   ViewType getView();

   /**
    * \brief Returns a non-modifiable view of the array view.
    */
   __cuda_callable__
   ConstViewType getConstView() const;


   static String getType();


   // All other Vector methods follow...
   void addElement( IndexType i, RealType value );

   template< typename Scalar >
   void addElement( IndexType i,
                    RealType value,
                    Scalar thisElementMultiplicator );

   template< typename Vector >
   VectorView& operator-=( const Vector& vector );

   template< typename Vector >
   VectorView& operator+=( const Vector& vector );

   template< typename Scalar >
   VectorView& operator*=( Scalar c );

   template< typename Scalar >
   VectorView& operator/=( Scalar c );

   NonConstReal max() const;

   NonConstReal min() const;

   NonConstReal absMax() const;

   NonConstReal absMin() const;

   template< typename ResultType = NonConstReal, typename Scalar >
   ResultType lpNorm( Scalar p ) const;

   template< typename ResultType = NonConstReal >
   ResultType sum() const;

   template< typename Vector >
   NonConstReal differenceMax( const Vector& v ) const;

   template< typename Vector >
   NonConstReal differenceMin( const Vector& v ) const;

   template< typename Vector >
   NonConstReal differenceAbsMax( const Vector& v ) const;

   template< typename Vector >
   NonConstReal differenceAbsMin( const Vector& v ) const;

   template< typename ResultType = NonConstReal, typename Vector, typename Scalar >
   ResultType differenceLpNorm( const Vector& v, Scalar p ) const;

   template< typename ResultType = NonConstReal, typename Vector >
   ResultType differenceSum( const Vector& v ) const;

   template< typename Scalar >
   void scalarMultiplication( Scalar alpha );

   //! Computes scalar dot product
   template< typename Vector >
   NonConstReal scalarProduct( const Vector& v ) const;

   //! Computes this = thisMultiplicator * this + alpha * x.
   template< typename Vector, typename Scalar1 = Real, typename Scalar2 = Real >
   void addVector( const Vector& x,
                   Scalar1 alpha = 1.0,
                   Scalar2 thisMultiplicator = 1.0 );

   //! Computes this = thisMultiplicator * this + multiplicator1 * v1 + multiplicator2 * v2.
   template< typename Vector1, typename Vector2, typename Scalar1, typename Scalar2, typename Scalar3 = Real >
   void addVectors( const Vector1& v1,
                    Scalar1 multiplicator1,
                    const Vector2& v2,
                    Scalar2 multiplicator2,
                    Scalar3 thisMultiplicator = 1.0 );

   void computePrefixSum();

   void computePrefixSum( IndexType begin, IndexType end );

   void computeExclusivePrefixSum();

   void computeExclusivePrefixSum( IndexType begin, IndexType end );
};

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/VectorView_impl.h>
