/***************************************************************************
                          DistributedVector.h  -  description
                             -------------------
    begin                : Sep 7, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/Containers/DistributedArray.h>
#include <TNL/Containers/DistributedVectorView.h>

namespace TNL {
namespace Containers {

template< typename Real,
          typename Device = Devices::Host,
          typename Index = int,
          typename Communicator = Communicators::MpiCommunicator >
class DistributedVector
: public DistributedArray< Real, Device, Index, Communicator >
{
   using CommunicationGroup = typename Communicator::CommunicationGroup;
   using BaseType = DistributedArray< Real, Device, Index, Communicator >;
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
    * \brief A template which allows to quickly obtain a \ref Vector type with changed template parameters.
    */
   template< typename _Real,
             typename _Device = Device,
             typename _Index = Index,
             typename _Communicator = Communicator >
   using Self = DistributedVector< _Real, _Device, _Index, _Communicator >;


   // inherit all constructors and assignment operators from Array
   using BaseType::DistributedArray;
#if !defined(__CUDACC_VER_MAJOR__) || __CUDACC_VER_MAJOR__ < 11
   using BaseType::operator=;
#endif

   DistributedVector() = default;

   /**
    * \brief Copy constructor (makes a deep copy).
    */
   explicit DistributedVector( const DistributedVector& ) = default;

   /**
    * \brief Default move constructor.
    */
   DistributedVector( DistributedVector&& ) = default;

   /**
    * \brief Copy-assignment operator for copying data from another vector.
    */
   DistributedVector& operator=( const DistributedVector& ) = default;

   /**
    * \brief Move-assignment operator for acquiring data from \e rvalues.
    */
   DistributedVector& operator=( DistributedVector&& ) = default;

   /**
    * \brief Returns a modifiable view of the local part of the vector.
    */
   LocalViewType getLocalView();

   /**
    * \brief Returns a non-modifiable view of the local part of the vector.
    */
   ConstLocalViewType getConstLocalView() const;

   /**
    * \brief Returns a modifiable view of the local part of the vector,
    * including ghost values.
    */
   LocalViewType getLocalViewWithGhosts();

   /**
    * \brief Returns a non-modifiable view of the local part of the vector,
    * including ghost values.
    */
   ConstLocalViewType getConstLocalViewWithGhosts() const;

   /**
    * \brief Returns a modifiable view of the vector.
    */
   ViewType getView();

   /**
    * \brief Returns a non-modifiable view of the vector.
    */
   ConstViewType getConstView() const;

   /**
    * \brief Conversion operator to a modifiable view of the vector.
    */
   operator ViewType();

   /**
    * \brief Conversion operator to a non-modifiable view of the vector.
    */
   operator ConstViewType() const;


   /*
    * Usual Vector methods follow below.
    */
   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVector& operator=( Scalar c );

   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVector& operator+=( Scalar c );

   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVector& operator-=( Scalar c );

   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVector& operator*=( Scalar c );

   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVector& operator/=( Scalar c );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVector& operator=( const Vector& vector );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVector& operator+=( const Vector& vector );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVector& operator-=( const Vector& vector );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVector& operator*=( const Vector& vector );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVector& operator/=( const Vector& vector );

   template< Algorithms::ScanType Type = Algorithms::ScanType::Inclusive >
   void scan( IndexType begin = 0, IndexType end = 0 );
};

// Enable expression templates for DistributedVector
namespace Expressions {
   template< typename Real, typename Device, typename Index, typename Communicator >
   struct HasEnabledDistributedExpressionTemplates< DistributedVector< Real, Device, Index, Communicator > >
   : std::true_type
   {};
} // namespace Expressions

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/DistributedVector.hpp>
