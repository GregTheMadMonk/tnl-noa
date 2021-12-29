/***************************************************************************
                          IterativeSolverMonitor.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Solvers/SolverMonitor.h>

namespace TNL {
   namespace Solvers {

/**
 * \brief Object for monitoring convergence of iterative solvers.
 *
 * \tparam Real is a type of the floating-point arithmetics.
 * \tparam Index is an indexing type.
 */
template< typename Real, typename Index>
class IterativeSolverMonitor : public SolverMonitor
{
   public:

      /**
       * \brief A type of the floating-point arithmetics.
       */
      using RealType = Real;

      /**
       * \brief A type for indexing.
       */
      using IndexType = Index;

      /**
       * \brief Construct with no parameters.
       */
      IterativeSolverMonitor();

      /**
       * \brief This method can be used for naming a stage of the monitored solver.
       *
       * The stage name can be used to differ between various stages of iterative solvers.
       *
       * \param stage is name of the solver stage.
       */
      void setStage( const std::string& stage );

      /**
       * \brief Set the time of the simulated evolution if it is time dependent.
       *
       * This can be used for example when solving parabolic or hyperbolic PDEs.
       *
       * \param time time of the simulated evolution.
       */
      void setTime( const RealType& time );

      /**
       * \brief Set the time step for time dependent iterative solvers.
       *
       * \param timeStep time step of the time dependent iterative solver.
       */
      void setTimeStep( const RealType& timeStep );

      /**
       * \brief Set number of the current iteration.
       *
       * \param iterations is number of the current iteration.
       */
      void setIterations( const IndexType& iterations );

      /**
       * \brief Set residue of the current approximation of the solution.
       *
       * \param residue is a residue of the current approximation of the solution.
       */
      void setResidue( const RealType& residue );

      /**
       * \brief Set up the verbosity of the monitor.
       *
       * \param verbose is the new value of the verbosity of the monitor.
       */
      void setVerbose( const IndexType& verbose );

      /**
       * \brief Set the number of nodes of the numerical mesh or lattice.
       *
       * This can be used to compute the number of nodes processed per one second.
       *
       * \param nodes is number of nodes of the numerical mesh or lattice.
       */
      void setNodesPerIteration( const IndexType& nodes );

      /**
       * \brief Causes that the monitor prints out the status of the solver.
       */
      virtual void refresh();

   protected:

      int getLineWidth();

      std::string stage, saved_stage;

      std::atomic_bool saved, attributes_changed;

      RealType time, saved_time, timeStep, saved_timeStep, residue, saved_residue, elapsed_time_before_refresh, last_mlups;
      //TODO: Move MLUPS to LBM solver only i.e create solver monitor for LBM

      IndexType iterations, saved_iterations, iterations_before_refresh;

      IndexType verbose;

      IndexType nodesPerIteration;
};

   } // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/IterativeSolverMonitor.hpp>
