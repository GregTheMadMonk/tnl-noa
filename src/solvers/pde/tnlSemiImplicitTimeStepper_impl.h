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

#include "tnlSemiImplicitTimeStepper.h"

template< typename Problem,
          typename LinearSystemSolver >
tnlSemiImplicitTimeStepper< Problem, LinearSystemSolver >::
tnlSemiImplicitTimeStepper()
: problem( 0 ),
  linearSystemSolver( 0 ),
  timeStep( 0 ),
  allIterations( 0 )
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
   this->verbose = parameters.getParameter< bool >( "verbose" );
   return true;
}

template< typename Problem,
          typename LinearSystemSolver >
bool
tnlSemiImplicitTimeStepper< Problem, LinearSystemSolver >::
init( const MeshType& mesh )
{
   cout << "Setting up the linear system...";
   if( ! this->problem->setupLinearSystem( mesh, this->matrix ) )
      return false;
   cout << " [ OK ]" << endl;
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

   this->preIterateTimer.reset();
   this->linearSystemAssemblerTimer.reset();
   this->preconditionerUpdateTimer.reset();
   this->linearSystemSolverTimer.reset();
   this->postIterateTimer.reset();

   this->allIterations = 0;
   return true;
}

template< typename Problem,
          typename LinearSystemSolver >
void
tnlSemiImplicitTimeStepper< Problem, LinearSystemSolver >::
setProblem( ProblemType& problem )
{
   this->problem = &problem;
};

template< typename Problem,
          typename LinearSystemSolver >
Problem*
tnlSemiImplicitTimeStepper< Problem, LinearSystemSolver >::
getProblem() const
{
    return this->problem;
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
   return true;
};

template< typename Problem,
          typename LinearSystemSolver >
bool
tnlSemiImplicitTimeStepper< Problem, LinearSystemSolver >::
solve( const RealType& time,
       const RealType& stopTime,
       const MeshType& mesh,
       DofVectorType& dofVector,
       MeshDependentDataType& meshDependentData )
{
   tnlAssert( this->problem != 0, );
   RealType t = time;
   this->linearSystemSolver->setMatrix( this->matrix );
   PreconditionerType preconditioner;
   tnlSolverStarterSolverPreconditionerSetter< LinearSystemSolverType, PreconditionerType >
       ::run( *(this->linearSystemSolver), preconditioner );

   while( t < stopTime )
   {
      RealType currentTau = Min( this->timeStep, stopTime - t );

      this->preIterateTimer.start();
      if( ! this->problem->preIterate( t,
                                       currentTau,
                                       mesh,
                                       dofVector,
                                       meshDependentData ) )
      {
         cerr << endl << "Preiteration failed." << endl;
         return false;
      }
      this->preIterateTimer.stop();

      if( verbose )
         cout << "                                                                  Assembling the linear system ... \r" << flush;

      this->linearSystemAssemblerTimer.start();
      this->problem->assemblyLinearSystem( t,
                                           currentTau,
                                           mesh,
                                           dofVector,
                                           this->matrix,
                                           this->rightHandSide,
                                           meshDependentData );
      this->linearSystemAssemblerTimer.stop();

      if( verbose )
         cout << "                                                                  Solving the linear system for time " << t + currentTau << "             \r" << flush;

      this->preconditionerUpdateTimer.start();
      preconditioner.update( this->matrix );
      this->preconditionerUpdateTimer.stop();

      this->linearSystemSolverTimer.start();
      if( ! this->linearSystemSolver->template solve< DofVectorType, tnlLinearResidueGetter< MatrixType, DofVectorType > >( this->rightHandSide, dofVector ) )
      {
         cerr << endl << "The linear system solver did not converge." << endl;
         return false;
      }
      this->linearSystemSolverTimer.stop();
      this->allIterations += this->linearSystemSolver->getIterations();

      //if( verbose )
      //   cout << endl;

      this->postIterateTimer.start();
      if( ! this->problem->postIterate( t,
                                        currentTau,
                                        mesh,
                                        dofVector,
                                        meshDependentData ) )
      {
         cerr << endl << "Postiteration failed." << endl;
         return false;
      }
      this->postIterateTimer.stop();

      t += currentTau;
   }
   return true;
}

template< typename Problem,
          typename LinearSystemSolver >
bool
tnlSemiImplicitTimeStepper< Problem, LinearSystemSolver >::
writeEpilog( tnlLogger& logger )
{
   logger.writeParameter< long long int >( "Iterations count:", this->allIterations );
   logger.writeParameter< const char* >( "Pre-iterate time:", "" );
   this->preIterateTimer.writeLog( logger, 1 );
   logger.writeParameter< const char* >( "Linear system assembler time:", "" );
   this->linearSystemAssemblerTimer.writeLog( logger, 1 );
   logger.writeParameter< const char* >( "Preconditioner update time:", "" );
   this->preconditionerUpdateTimer.writeLog( logger, 1 );
   logger.writeParameter< const char* >( "Linear system solver time:", "" );
   this->linearSystemSolverTimer.writeLog( logger, 1 );
   logger.writeParameter< const char* >( "Post-iterate time:", "" );
   this->postIterateTimer.writeLog( logger, 1 );
   return true;
}

#endif /* TNLSEMIIMPLICITTIMESTEPPER_IMPL_H_ */
