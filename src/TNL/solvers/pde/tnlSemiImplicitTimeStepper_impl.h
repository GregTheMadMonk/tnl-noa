/***************************************************************************
                          tnlSemiImplicitTimeStepper_impl.h  -  description
                             -------------------
    begin                : Oct 4, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/core/mfuncs.h>
#include "tnlSemiImplicitTimeStepper.h"

namespace TNL {

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
  std::cout << "Setting up the linear system...";
   if( ! this->problem->setupLinearSystem( mesh, this->matrix ) )
      return false;
  std::cout << " [ OK ]" << std::endl;
   if( this->matrix.getRows() == 0 || this->matrix.getColumns() == 0 )
   {
      std::cerr << "The matrix for the semi-implicit time stepping was not set correctly." << std::endl;
      if( ! this->matrix.getRows() )
         std::cerr << "The matrix dimensions are set to 0 rows." << std::endl;
      if( ! this->matrix.getColumns() )
         std::cerr << "The matrix dimensions are set to 0 columns." << std::endl;
      std::cerr << "Please check the method 'setupLinearSystem' in your solver." << std::endl;
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
      std::cerr << "Time step for tnlSemiImplicitTimeStepper must be positive. " << std::endl;
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
      RealType currentTau = min( this->timeStep, stopTime - t );

      this->preIterateTimer.start();
      if( ! this->problem->preIterate( t,
                                       currentTau,
                                       mesh,
                                       dofVector,
                                       meshDependentData ) )
      {
         std::cerr << std::endl << "Preiteration failed." << std::endl;
         return false;
      }
      this->preIterateTimer.stop();

      if( verbose )
        std::cout << "                                                                  Assembling the linear system ... \r" << std::flush;

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
        std::cout << "                                                                  Solving the linear system for time " << t + currentTau << "             \r" << std::flush;

      this->preconditionerUpdateTimer.start();
      preconditioner.update( this->matrix );
      this->preconditionerUpdateTimer.stop();

      this->linearSystemSolverTimer.start();
      if( ! this->linearSystemSolver->template solve< DofVectorType, tnlLinearResidueGetter< MatrixType, DofVectorType > >( this->rightHandSide, dofVector ) )
      {
         std::cerr << std::endl << "The linear system solver did not converge." << std::endl;
         return false;
      }
      this->linearSystemSolverTimer.stop();
      this->allIterations += this->linearSystemSolver->getIterations();

      //if( verbose )
      //  std::cout << std::endl;

      this->postIterateTimer.start();
      if( ! this->problem->postIterate( t,
                                        currentTau,
                                        mesh,
                                        dofVector,
                                        meshDependentData ) )
      {
         std::cerr << std::endl << "Postiteration failed." << std::endl;
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

} // namespace TNL
