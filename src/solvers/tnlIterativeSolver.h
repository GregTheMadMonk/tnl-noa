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

#include <solvers/tnlIterativeSolverMonitor.h>

template< typename Real, typename Index >
class tnlIterativeSolver
{
   public:

   tnlIterativeSolver();

   void setMaxIterations( const Index& maxIterations );

   const Index& getMaxIterations() const;

   void resetIterations();

   bool nextIteration();

   const Index& getIterations() const;

   void setMaxResidue( const Real& maxResidue );

   const Real& getMaxResidue() const;

   void setMinResidue( const Real& minResidue );

   const Real& getMinResidue() const;

   void setResidue( const Real& residue );

   const Real& getResidue() const;

   void setRefreshRate( const Index& refreshRate );

   void setSolverMonitor( tnlIterativeSolverMonitor< Real, Index >& solverMonitor );

   void refreshSolverMonitor();

   protected:

   Index maxIterations;

   Index currentIteration;

   Real maxResidue;

   /****
    * If the current residue is over minResidue the solver is stopped.
    */
   Real minResidue;

   Real currentResidue;

   tnlIterativeSolverMonitor< Real, Index >* solverMonitor;

   Index refreshRate;
};

#include <implementation/solvers/tnlIterativeSolver_impl.h>

#endif /* TNLITERATIVESOLVER_H_ */
