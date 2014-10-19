/***************************************************************************
                          tnlSemiImplicitTimeStepper_impl.h  -  description
                             -------------------
    begin                : Oct 4, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLSEMIIMPLICITTIMESTEPPER_IMPL_H_
#define TNLSEMIIMPLICITTIMESTEPPER_IMPL_H_

#include <core/mfuncs.h>

template< typename Problem,
          typename LinearSystemSolver >
tnlSemiImplicitTimeStepper< Problem, LinearSystemSolver >::
tnlSemiImplicitTimeStepper()
: problem( 0 ),
  linearSystemSolver( 0 ),
  timeStep( 0 )
{
};

template< typename Problem,
          typename LinearSystemSolver >
void
tnlSemiImplicitTimeStepper< Problem, LinearSystemSolver >::
configSetup( tnlConfigDescription& config,
             const tnlString& prefix )
{
}

template< typename Problem,
          typename LinearSystemSolver >
bool
tnlSemiImplicitTimeStepper< Problem, LinearSystemSolver >::
setup( const tnlParameterContainer& parameters,
      const tnlString& prefix )
{
   return true;
}

template< typename Problem,
          typename LinearSystemSolver >
bool
tnlSemiImplicitTimeStepper< Problem, LinearSystemSolver >::
init( const MeshType& mesh )
{
   return this->problem->setupLinearSystem( mesh, this->matrix );
}

template< typename Problem,
          typename LinearSystemSolver >
void
tnlSemiImplicitTimeStepper< Problem, LinearSystemSolver >::
setProblem( ProblemType& problem )
{
   this -> problem = &problem;
};

template< typename Problem,
          typename LinearSystemSolver >
Problem*
tnlSemiImplicitTimeStepper< Problem, LinearSystemSolver >::
getProblem() const
{
    return this -> problem;
};

template< typename Problem,
          typename LinearSystemSolver >
void
tnlSemiImplicitTimeStepper< Problem, LinearSystemSolver >::
setSolver( LinearSystemSolver& linearSystemSolver )
{
   this->linearSystemSolver = &linearSystemSolver;
}
template< typename Problem,
          typename LinearSystemSolver >
LinearSystemSolver*
tnlSemiImplicitTimeStepper< Problem, LinearSystemSolver >::
getSolver() const
{
   return this->linearSystemSolver;
}

template< typename Problem,
          typename LinearSystemSolver >
bool
tnlSemiImplicitTimeStepper< Problem, LinearSystemSolver >::
setTimeStep( const RealType& timeStep )
{
   if( timeStep <= 0.0 )
   {
      cerr << "Time step for tnlSemiImplicitTimeStepper must be positive. " << endl;
      return false;
   }
   this->timeStep = timeStep;
};

template< typename Problem,
          typename LinearSystemSolver >
bool
tnlSemiImplicitTimeStepper< Problem, LinearSystemSolver >::
solve( const RealType& time,
       const RealType& stopTime,
       const MeshType& mesh,
       DofVectorType& dofVector )
{
   tnlAssert( this->problem != 0, );
   RealType t = time;
   while( t < stopTime )
   {
      RealType currentTau = Min( this->timeStep, stopTime - t );
      this->problem->assemblyLinearSystem( t,
                                           currentTau,
                                           mesh,
                                           dofVector,
                                           this->matrix,
                                           this->rightHandSide );
      t += currentTau;
   }
}

#endif /* TNLSEMIIMPLICITTIMESTEPPER_IMPL_H_ */
