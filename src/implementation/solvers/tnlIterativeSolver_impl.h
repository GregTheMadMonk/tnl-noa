/***************************************************************************
                          tnlIterativeSolver_impl.h  -  description
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

#ifndef TNLITERATIVESOLVER_IMPL_H_
#define TNLITERATIVESOLVER_IMPL_H_

#include <float.h>

template< typename Real, typename Index >
tnlIterativeSolver< Real, Index> :: tnlIterativeSolver()
: maxIterations( 0 ),
  currentIteration( 0 ),
  maxResidue( 0 ),
  minResidue( DBL_MAX ),
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
bool tnlIterativeSolver< Real, Index> :: nextIteration()
{
   if( this -> solverMonitor &&
       this -> currentIteration % this -> refreshRate == 0 )
      solverMonitor -> refresh();
   this -> currentIteration ++;
   if( this -> getResidue() > this -> getMinResidue() && this -> currentIteration > 10 )
      return false;
   return true;
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
void tnlIterativeSolver< Real, Index> :: setMinResidue( const Real& minResidue )
{
   this -> minResidue = minResidue;
}

template< typename Real, typename Index >
const Real& tnlIterativeSolver< Real, Index> :: getMinResidue() const
{
   return this -> minResidue;
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
void tnlIterativeSolver< Real, Index> :: setSolverMonitor( tnlIterativeSolverMonitor< Real, Index >& solverMonitor )
{
   this -> solverMonitor = &solverMonitor;
}

template< typename Real, typename Index >
void tnlIterativeSolver< Real, Index> :: refreshSolverMonitor()
{
   if( this -> solverMonitor )
   {
      this -> solverMonitor -> setIterations( this -> getIterations() );
      this -> solverMonitor -> setResidue( this -> getResidue() );
      this -> solverMonitor -> refresh();
   }
}


#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

extern template class tnlIterativeSolver< float,  int >;
extern template class tnlIterativeSolver< double, int >;
extern template class tnlIterativeSolver< float,  long int >;
extern template class tnlIterativeSolver< double, long int >;

#ifdef HAVE_CUDA
extern template class tnlIterativeSolver< float,  int >;
extern template class tnlIterativeSolver< double, int >;
extern template class tnlIterativeSolver< float,  long int >;
extern template class tnlIterativeSolver< double, long int >;
#endif

#endif

#endif /* TNLITERATIVESOLVER_IMPL_H_ */
