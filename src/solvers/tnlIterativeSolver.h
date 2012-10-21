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

#include <solvers/tnlSolverMonitor.h>

template< typename Real, typename Index >
class tnlIterativeSolver
{
   public:

   tnlIterativeSolver();

   void setMaxIterations( const Index& maxIterations );

   const Index& getMaxIterations() const;

   void resetIterations();

   void nextIteration();

   const Index& getIterations() const;

   void setMaxResidue( const Real& maxResidue );

   const Real& getMaxResidue() const;

   void setResidue( const Real& residue );

   const Real& getResidue() const;

   void setRefreshRate( const Index& refreshRate );

   void setSolverMonitor( tnlSolverMonitor< Real, Index >& solverMonitor );

   void refreshSolverMonitor();

   protected:

   Index maxIterations;

   Index currentIteration;

   Real maxResidue;

   Real currentResidue;

   tnlSolverMonitor< Real, Index >* solverMonitor;

   Index refreshRate;
};

template< typename Real, typename Index >
tnlIterativeSolver< Real, Index> :: tnlIterativeSolver()
: maxIterations( 0 ),
  currentIteration( 0 ),
  maxResidue( 0 ),
  currentResidue( 0 ),
  solverMonitor( 0 ),
  refreshRate( 1 )
{
};

template< typename Real, typename Index >
void tnlIterativeSolver< Real, Index> :: setMaxIterations( const Index& maxIterations )
{
   this -> maxIterations = maxIterations;
}

template< typename Real, typename Index >
const Index& tnlIterativeSolver< Real, Index> :: getMaxIterations() const
{
   return this -> maxIterations;
}

template< typename Real, typename Index >
void tnlIterativeSolver< Real, Index> :: resetIterations()
{
   this -> currentIteration = 0;
}

template< typename Real, typename Index >
void tnlIterativeSolver< Real, Index> :: nextIteration()
{
   if( this -> solverMonitor &&
       this -> currentIteration % this -> refreshRate == 0 )
      solverMonitor -> refresh();
   this -> currentIteration ++;
}

template< typename Real, typename Index >
const Index& tnlIterativeSolver< Real, Index> :: getIterations() const
{
   return this -> currentIteration;
}

template< typename Real, typename Index >
void tnlIterativeSolver< Real, Index> :: setMaxResidue( const Real& maxResidue )
{
   this -> maxResidue = maxResidue;
}

template< typename Real, typename Index >
const Real& tnlIterativeSolver< Real, Index> :: getMaxResidue() const
{
   return this -> maxResidue;
}

template< typename Real, typename Index >
void tnlIterativeSolver< Real, Index> :: setResidue( const Real& residue )
{
   this -> currentResidue = residue;
}

template< typename Real, typename Index >
const Real& tnlIterativeSolver< Real, Index> :: getResidue() const
{
   return this -> currentResidue;
}

template< typename Real, typename Index >
void tnlIterativeSolver< Real, Index> :: setRefreshRate( const Index& refreshRate )
{
   this -> refreshRate = refreshRate;
}

template< typename Real, typename Index >
void tnlIterativeSolver< Real, Index> :: setSolverMonitor( tnlSolverMonitor< Real, Index >& solverMonitor )
{
   this -> solverMonitor = &solverMonitor;
}

template< typename Real, typename Index >
void tnlIterativeSolver< Real, Index> :: refreshSolverMonitor()
{
   if( this -> solverMonitor )
      this -> solverMonitor -> refresh();
}

#endif /* TNLITERATIVESOLVER_H_ */
