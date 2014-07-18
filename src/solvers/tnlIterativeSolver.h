/***************************************************************************
                          tnlIterativeSolver.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLITERATIVESOLVER_H_
#define TNLITERATIVESOLVER_H_

#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <solvers/tnlIterativeSolverMonitor.h>

template< typename Real, typename Index >
class tnlIterativeSolver
{
   public:

   tnlIterativeSolver();

   static void configSetup( tnlConfigDescription& config,
                            const tnlString& prefix = "" );

   bool init( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );

   void setMaxIterations( const Index& maxIterations );

   const Index& getMaxIterations() const;

   void setMinIterations( const Index& minIterations );

   const Index& getMinIterations() const;

   void resetIterations();

   bool nextIteration();

   const Index& getIterations() const;

   void setConvergenceResidue( const Real& convergenceResidue );

   const Real& getConvergenceResidue() const;

   void setDivergenceResidue( const Real& divergenceResidue );

   const Real& getDivergenceResidue() const;

   void setResidue( const Real& residue );

   const Real& getResidue() const;

   void setRefreshRate( const Index& refreshRate );

   void setSolverMonitor( tnlIterativeSolverMonitor< Real, Index >& solverMonitor );

   void refreshSolverMonitor();

   protected:

   Index maxIterations;

   Index minIterations;

   Index currentIteration;

   Real convergenceResidue;

   /****
    * If the current residue is over divergenceResidue the solver is stopped.
    */
   Real divergenceResidue;

   Real currentResidue;

   tnlIterativeSolverMonitor< Real, Index >* solverMonitor;

   Index refreshRate;
};

#include <implementation/solvers/tnlIterativeSolver_impl.h>

#endif /* TNLITERATIVESOLVER_H_ */
