// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Solvers/ODE/ExplicitSolver.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/StaticVector.h>

namespace TNL {
namespace Solvers {
namespace ODE {

template< typename Real >//, typename Aux = void >
struct ODESolverDefaultTypesGetter
{
   using RealType = Real;
   using IndexType = int;
};

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
struct ODESolverDefaultTypesGetter< Containers::Vector< Real, Device, Index, Allocator > >
{
   using RealType = Real;
   using IndexType = Index;
};

template< typename Real,
          typename Device,
          typename Index >
struct ODESolverDefaultTypesGetter< Containers::VectorView< Real, Device, Index > >
{
   using RealType = Real;
   using IndexType = Index;
};

template< int Size, typename Real >
struct ODESolverDefaultTypesGetter< TNL::Containers::StaticVector< Size, Real > >
{
   using RealType = Real;
   using IndexType = int;
};

template< typename Real,
          typename SolverMonitor = IterativeSolverMonitor< typename ODESolverDefaultTypesGetter< Real >::RealType,
                                                           typename ODESolverDefaultTypesGetter< Real >::IndexType > >
class Euler : public ExplicitSolver< Real, int, SolverMonitor >
{
   public:

      using RealType = Real;
      using IndexType  = int;
      using VectorType = Real;
      using DofVectorType = VectorType;
      using SolverMonitorType = SolverMonitor;

      __cuda_callable__
      Euler() = default;

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" );

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" );

      __cuda_callable__
      void setCFLCondition( const RealType& cfl );

      __cuda_callable__
      const RealType& getCFLCondition() const;

      __cuda_callable__
      template< typename RHSFunction >
      bool solve( VectorType& u, RHSFunction&& rhs );

   protected:
      DofVectorType k1;

      RealType cflCondition = 0.0;
};


template< int Size_,
          typename Real,
          typename SolverMonitor >
class Euler< TNL::Containers::StaticVector< Size_, Real >, SolverMonitor >
    : public ExplicitSolver< Real, int, SolverMonitor >
{
   public:

      static constexpr int Size = Size_;
      using RealType = Real;
      using IndexType  = int;
      using VectorType = TNL::Containers::StaticVector< Size, Real >;
      using DofVectorType = VectorType;
      using SolverMonitorType = SolverMonitor;

      __cuda_callable__
      Euler() = default;

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" );

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" );

      __cuda_callable__
      void setCFLCondition( const RealType& cfl );

      __cuda_callable__
      const RealType& getCFLCondition() const;

      __cuda_callable__
      template< typename RHSFunction >
      bool solve( VectorType& u, RHSFunction&& rhs );

   protected:
      DofVectorType k1;

      RealType cflCondition = 0.0;
};


template< typename Real,
          typename Device,
          typename Index,
          typename Allocator,
          typename SolverMonitor >
class Euler< Containers::Vector< Real, Device, Index, Allocator >, SolverMonitor > : public ExplicitSolver< Real, Index, SolverMonitor >
{
public:
      using RealType = Real;
      using DeviceType = Device;
      using IndexType  = Index;
      using VectorType = Containers::Vector< Real, Device, Index, Allocator >;
      using DofVectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
      using SolverMonitorType = SolverMonitor;

      Euler() = default;

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" );

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" );

   const RealType&
   getCFLCondition() const;

   bool
   solve( DofVectorPointer& u );

      template< typename RHSFunction >
      bool solve( VectorType& u, RHSFunction&& rhs );

   protected:
      DofVectorType _k1;

      RealType cflCondition = 0.0;
};

template< typename Real,
          typename Device,
          typename Index,
          typename SolverMonitor >
class Euler< Containers::VectorView< Real, Device, Index >, SolverMonitor > : public Euler< Containers::Vector< Real, Device, Index >, SolverMonitor >
{};

} // namespace ODE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/ODE/Euler.hpp>
