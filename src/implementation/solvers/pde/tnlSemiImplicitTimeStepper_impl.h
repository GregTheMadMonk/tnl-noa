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
   config.addEntry< bool >( "verbose", "Verbose mode.", true );
}

template< typename Problem,
          typename LinearSystemSolver >
bool
tnlSemiImplicitTimeStepper< Problem, LinearSystemSolver >::
setup( const tnlParameterContainer& parameters,
      const tnlString& prefix )
{
   this->verbose = parameters.GetParameter< bool >( "verbose" );
   return true;
}

template< typename Problem,
          typename LinearSystemSolver >
bool
tnlSemiImplicitTimeStepper< Problem, LinearSystemSolver >::
init( const MeshType& mesh )
{
   if( ! this->problem->setupLinearSystem( mesh, this->matrix ) )
      return false;
   if( this->matrix.getRows() == 0 || this->matrix.getColumns() == 0 )
   {
      cerr << "The matrix for the semi-implicit time stepping was not set correctly." << endl;
      if( ! this->matrix.getRows() )
         cerr << "The matrix dimensions are set to 0 rows." << endl;
      if( ! this->matrix.getColumns() )
         cerr << "The matrix dimensions are set to 0 columns." << endl;
      cerr << "Please check the method 'setupLinearSystem' in your solver." << endl;
      return false;
   }
   if( ! this->rightHandSide.setSize( this->matrix.getRows() ) )
      return false;
   return true;
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
   this->linearSystemSolver->setMatrix( this->matrix );
   while( t < stopTime )
   {
      RealType currentTau = Min( this->timeStep, stopTime - t );

      if( verbose )
         cout << "                                                                  Assembling the linear system ... \r" << flush;
      this->problem->assemblyLinearSystem( t,
                                           currentTau,
                                           mesh,
                                           dofVector,
                                           this->matrix,
                                           this->rightHandSide );
      cout << "u = " << dofVector << endl;
      cout << "b = " << this->rightHandSide << endl;
      if( verbose )
         cout << "                                                                  Solving the linear system for time " << t << "             \r" << flush;
      if( ! this->linearSystemSolver->solve( this->rightHandSide, dofVector ) )
      {
         cerr << "The linear system solver did not converge." << endl;
         return false;
      }
      DofVectorType aux;
      aux.setLike( dofVector );
      this->matrix.vectorProduct( dofVector, aux );
      cout << "tau = " << currentTau << endl;
      cout << "u = " << dofVector << endl;
      cout << "Ax = " << aux << endl;
      cout << "b = " << this->rightHandSide << endl;
      t += currentTau;
   }
   return true;
}

#endif /* TNLSEMIIMPLICITTIMESTEPPER_IMPL_H_ */
