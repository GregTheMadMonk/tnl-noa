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
#include <TNL/Containers/Expressions/DistributedExpressionTemplates.h>
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
   using LocalViewType = Containers::VectorView< Real, Device, Index >;
   using ConstLocalViewType = Containers::VectorView< std::add_const_t< Real >, Device, Index >;
   using ViewType = DistributedVectorView< Real, Device, Index, Communicator >;
   using ConstViewType = DistributedVectorView< std::add_const_t< Real >, Device, Index, Communicator >;

   /**
    * \brief A template which allows to quickly obtain a \ref VectorView type with changed template parameters.
    */
   template< typename _Real,
             typename _Device = Device,
             typename _Index = Index,
             typename _Communicator = Communicator >
   using Self = DistributedVectorView< _Real, _Device, _Index, _Communicator >;


   // inherit all constructors and assignment operators from ArrayView
   using BaseType::DistributedArrayView;
#if !defined(__CUDACC_VER_MAJOR__) || __CUDACC_VER_MAJOR__ < 11
   using BaseType::operator=;
#endif

   // In C++14, default constructors cannot be inherited, although Clang
   // and GCC since version 7.0 inherit them.
   // https://stackoverflow.com/a/51854172
   DistributedVectorView() = default;

   // initialization by base class is not a copy constructor so it has to be explicit
   template< typename Real_ >  // template catches both const and non-const qualified Element
   DistributedVectorView( const Containers::DistributedArrayView< Real_, Device, Index, Communicator >& view )
   : BaseType( view ) {}

   LocalViewType getLocalView();

   ConstLocalViewType getConstLocalView() const;

   /**
    * \brief Returns a modifiable view of the array view.
    */
   ViewType getView();

   /**
    * \brief Returns a non-modifiable view of the array view.
    */
   ConstViewType getConstView() const;

   /*
    * Usual Vector methods follow below.
    */
   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVectorView& operator=( Scalar c );

   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVectorView& operator+=( Scalar c );

   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVectorView& operator-=( Scalar c );

   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVectorView& operator*=( Scalar c );

   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVectorView& operator/=( Scalar c );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVectorView& operator=( const Vector& vector );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVectorView& operator+=( const Vector& vector );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVectorView& operator-=( const Vector& vector );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVectorView& operator*=( const Vector& vector );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVectorView& operator/=( const Vector& vector );

   template< Algorithms::ScanType Type = Algorithms::ScanType::Inclusive >
   void scan( IndexType begin = 0, IndexType end = 0 );
};

// Enable expression templates for DistributedVector
namespace Expressions {
   template< typename Real, typename Device, typename Index, typename Communicator >
   struct HasEnabledDistributedExpressionTemplates< DistributedVectorView< Real, Device, Index, Communicator > >
   : std::true_type
   {};
} // namespace Expressions

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/DistributedVectorView.hpp>
