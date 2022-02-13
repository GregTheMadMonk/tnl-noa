// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <math.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Solvers/ODE/StaticExplicitSolver.h>

namespace TNL {
namespace Solvers {
namespace ODE {

template< class Real >
class StaticMerson : public StaticExplicitSolver< Real, int >
{
   public:

      using RealType = Real;
      using IndexType  = int;
      using VectorType = Real;
      using DofVectorType = VectorType;

      __cuda_callable__
      StaticMerson() = default;

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" );

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" );

      __cuda_callable__
      void setAdaptivity( const RealType& a );

      __cuda_callable__
      const RealType& getAdaptivity() const;

      template< typename RHSFunction, typename... Args >
      __cuda_callable__
      bool solve( VectorType& u, RHSFunction&& rhs, Args... args );

   protected:

      DofVectorType k1, k2, k3, k4, k5, kAux;

      /****
       * This controls the accuracy of the solver
       */
      RealType adaptivity = 0.00001;
};

template< int Size_, class Real >
class StaticMerson< Containers::StaticVector< Size_, Real > >
   : public StaticExplicitSolver< Real, int >
{
   public:

      using RealType = Real;
      using IndexType  = int;
      using VectorType = Containers::StaticVector< Size_, Real >;
      using DofVectorType = VectorType;

      __cuda_callable__
      StaticMerson() = default;

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" );

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" );

      __cuda_callable__
      void setAdaptivity( const RealType& a );

      __cuda_callable__
      const RealType& getAdaptivity() const;

      template< typename RHSFunction, typename... Args >
      __cuda_callable__
      bool solve( VectorType& u, RHSFunction&& rhs, Args... args );

   protected:

      DofVectorType k1, k2, k3, k4, k5, kAux;

      /****
       * This controls the accuracy of the solver
       */
      RealType adaptivity = 0.00001;
};


} // namespace ODE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/ODE/StaticMerson.hpp>
