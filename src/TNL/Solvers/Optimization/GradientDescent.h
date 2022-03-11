/***************************************************************************
                          Timer.h  -  description
                             -------------------
    begin                : Mar 14, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

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
