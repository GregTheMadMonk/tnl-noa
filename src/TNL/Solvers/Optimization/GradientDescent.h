// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Solvers/IterativeSolver.h>

namespace TNL {
   namespace Solvers {
      namespace Optimization {

template< typename Vector, typename SolverMonitor =  IterativeSolverMonitor< typename Vector::RealType, typename Vector::IndexType > >
class GradientDescent : public IterativeSolver< typename Vector::RealType, typename Vector::IndexType, SolverMonitor >
{
public:
   using RealType = typename Vector::RealType;
   using DeviceType = typename Vector::DeviceType;
   using IndexType = typename Vector::IndexType;
   using VectorType = Vector;
   using VectorView = typename Vector::ViewType;

   GradientDescent() = default;

   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" );

   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

   void
   setRelaxation( const RealType& lambda );

   const RealType&
   getRelaxation() const;

   template< typename GradientGetter >
   bool
   solve( VectorView& w, GradientGetter&& getGradient );

protected:

   RealType relaxation = 1.0;

   VectorType aux;

};

      } //namespace Optimization
   } //namespace Solvers
} //namespace TNL

#include <TNL/Solvers/Optimization/GradientDescent.hpp>
