/***************************************************************************
                          Merson.h  -  description
                             -------------------
    begin                : 2007/06/16
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <math.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Solvers/ODE/ExplicitSolver.h>

namespace TNL {
namespace Solvers {
namespace ODE {

template< class Problem,
          typename SolverMonitor = IterativeSolverMonitor< typename Problem::RealType, typename Problem::IndexType > >
class Merson : public ExplicitSolver< Problem, SolverMonitor >
{
   public:

      using ProblemType = Problem;
      using DofVectorType = typename Problem::DofVectorType;
      using RealType = typename Problem::RealType;
      using DeviceType = typename Problem::DeviceType;
      using IndexType = typename Problem::IndexType;
      using DofVectorPointer = Pointers::SharedPointer< DofVectorType, DeviceType >;
      using SolverMonitorType = SolverMonitor;

      Merson();

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" );

      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );

      void setAdaptivity( const RealType& a );

      bool solve( DofVectorPointer& u );

   protected:

      void writeGrids( const DofVectorPointer& u );

      DofVectorPointer _k1, _k2, _k3, _k4, _k5, _kAux;

      /****
       * This controls the accuracy of the solver
       */
      RealType adaptivity;

      Containers::Vector< RealType, DeviceType, IndexType > openMPErrorEstimateBuffer;
};

} // namespace ODE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/ODE/Merson.hpp>
