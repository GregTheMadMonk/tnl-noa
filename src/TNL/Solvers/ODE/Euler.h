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

template< typename DofsType, typename Aux = void >
struct ODESolverDefaultTypesGetter
{
   using RealType = typename DofsType::RealType;
   using IndexType = typename DofsType::IndexType;
};

template< int Size, typename Real >
struct ODESolverDefaultTypesGetter< TNL::Containers::StaticVector< Size, Real > >
{
   using RealType = Real;
   using IndexType = int;
};

template< typename Real >
struct ODESolverDefaultTypesGetter< Real, std::enable_if< std::is_arithmetic< Real >::value > >
{
   using RealType = Real;
   using IndexType = int;
};

template< typename Vector,
          typename SolverMonitor = IterativeSolverMonitor< typename ODESolverDefaultTypesGetter< Vector >::RealType,
                                                           typename ODESolverDefaultTypesGetter< Vector >::IndexType > >
class Euler;

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


template< typename Vector,
          typename SolverMonitor >
class Euler : public ExplicitSolver< typename Vector::RealType, typename Vector::IndexType, SolverMonitor >
{
public:
   using RealType = typename Vector::RealType;
   using DeviceType = typename Vector::DeviceType;
   using IndexType  = typename Vector::IndexType;
   using VectorType = Vector;
   using DofVectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using SolverMonitorType = SolverMonitor;

   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" );

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

}  // namespace ODE
}  // namespace Solvers
}  // namespace TNL

#include <TNL/Solvers/ODE/Euler.hpp>
