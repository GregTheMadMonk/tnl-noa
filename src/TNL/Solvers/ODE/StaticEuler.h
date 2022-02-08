// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Solvers/ODE/StaticExplicitSolver.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/StaticVector.h>

namespace TNL {
namespace Solvers {
namespace ODE {

template< typename Real >
class StaticEuler : public StaticExplicitSolver< Real, int >
{
   public:

      using RealType = Real;
      using IndexType  = int;
      using VectorType = Real;
      using DofVectorType = VectorType;
      using SolverMonitorType = SolverMonitor;

      __cuda_callable__
      StaticEuler() = default;

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
          typename Real >
class StaticEuler< TNL::Containers::StaticVector< Size_, Real > >
    : public StaticExplicitSolver< Real, int >
{
   public:

      static constexpr int Size = Size_;
      using RealType = Real;
      using IndexType  = int;
      using VectorType = TNL::Containers::StaticVector< Size, Real >;
      using DofVectorType = VectorType;

      __cuda_callable__
      StaticEuler() = default;

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

} // namespace ODE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/ODE/StaticEuler.hpp>
