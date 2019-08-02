/***************************************************************************
                          DistributedVectorView.h  -  description
                             -------------------
    begin                : Sep 20, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/Containers/DistributedArrayView.h>
#include <TNL/Containers/VectorView.h>

namespace TNL {
namespace Containers {

template< typename Real,
          typename Device = Devices::Host,
          typename Index = int,
          typename Communicator = Communicators::MpiCommunicator >
class DistributedVectorView
: public DistributedArrayView< Real, Device, Index, Communicator >
{
   using CommunicationGroup = typename Communicator::CommunicationGroup;
   using BaseType = DistributedArrayView< Real, Device, Index, Communicator >;
   using NonConstReal = typename std::remove_const< Real >::type;
public:
   using RealType = Real;
   using DeviceType = Device;
   using CommunicatorType = Communicator;
   using IndexType = Index;
   using LocalVectorViewType = Containers::VectorView< Real, Device, Index >;
   using ConstLocalVectorViewType = Containers::VectorView< std::add_const_t< Real >, Device, Index >;
   using HostType = DistributedVectorView< Real, Devices::Host, Index, Communicator >;
   using CudaType = DistributedVectorView< Real, Devices::Cuda, Index, Communicator >;
   using ViewType = DistributedVectorView< Real, Device, Index, Communicator >;
   using ConstViewType = DistributedVectorView< std::add_const_t< Real >, Device, Index, Communicator >;

   // inherit all constructors and assignment operators from ArrayView
   using BaseType::DistributedArrayView;
   using BaseType::operator=;

   // In C++14, default constructors cannot be inherited, although Clang
   // and GCC since version 7.0 inherit them.
   // https://stackoverflow.com/a/51854172
   __cuda_callable__
   DistributedVectorView() = default;

   // initialization by base class is not a copy constructor so it has to be explicit
   template< typename Real_ >  // template catches both const and non-const qualified Element
   __cuda_callable__
   DistributedVectorView( const DistributedArrayView< Real_, Device, Index, Communicator >& view )
   : BaseType::DistributedArrayView( view ) {}

   LocalVectorViewType getLocalVectorView();

   ConstLocalVectorViewType getLocalVectorView() const;
   
   ConstLocalVectorViewType getConstLocalVectorView() const;

   /**
    * \brief Returns a modifiable view of the array view.
    */
   __cuda_callable__
   ViewType getView();

   /**
    * \brief Returns a modifiable view of the array view.
    */
   __cuda_callable__
   ConstViewType getView() const;


   /**
    * \brief Returns a non-modifiable view of the array view.
    */
   __cuda_callable__
   ConstViewType getConstView() const;

   static String getType();

   /*
    * Usual Vector methods follow below.
    */
   void addElement( IndexType i,
                    RealType value );

   template< typename Scalar >
   void addElement( IndexType i,
                    RealType value,
                    Scalar thisElementMultiplicator );

   template< typename Vector >
   DistributedVectorView& operator-=( const Vector& vector );

   template< typename Vector >
   DistributedVectorView& operator+=( const Vector& vector );

   template< typename Scalar >
   DistributedVectorView& operator*=( Scalar c );

   template< typename Scalar >
   DistributedVectorView& operator/=( Scalar c );

   template< typename ResultType = NonConstReal >
   ResultType sum() const;

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

#include <TNL/Containers/DistributedVectorView_impl.h>
#include <TNL/Containers/DistributedVectorViewExpressions.h>
