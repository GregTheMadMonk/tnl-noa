/***************************************************************************
                          tnlExplicitTimeStepper_impl.h  -  description
                             -------------------
    begin                : Jan 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLEXPLICITTIMESTEPPER_IMPL_H_
#define TNLEXPLICITTIMESTEPPER_IMPL_H_

#include "tnlExplicitTimeStepper.h"


template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
tnlExplicitTimeStepper< Problem, OdeSolver >::
tnlExplicitTimeStepper()
: odeSolver( 0 ),
  problem( 0 ),
  timeStep( 0 ),
  allIterations( 0 )
{
};

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
void
tnlExplicitTimeStepper< Problem, OdeSolver >::
configSetup( tnlConfigDescription& config,
             const tnlString& prefix )
{
}

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
bool
tnlExplicitTimeStepper< Problem, OdeSolver >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   return true;
}

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
bool
tnlExplicitTimeStepper< Problem, OdeSolver >::
init( const MeshType& mesh )
{
   this->explicitUpdaterTimer.reset();
   this->mainTimer.reset();
   this->preIterateTimer.reset();
   this->postIterateTimer.reset();
   return true;
}

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
void
tnlExplicitTimeStepper< Problem, OdeSolver >::
setSolver( typename tnlExplicitTimeStepper< Problem, OdeSolver >::OdeSolverType& odeSolver )
{
   this->odeSolver = &odeSolver;
};

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
void
tnlExplicitTimeStepper< Problem, OdeSolver >::
setProblem( ProblemType& problem )
{
   this->problem = &problem;
};

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
Problem*
tnlExplicitTimeStepper< Problem, OdeSolver >::
getProblem() const
{
    return this->problem;
};

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
bool
tnlExplicitTimeStepper< Problem, OdeSolver >::
setTimeStep( const RealType& timeStep )
{
   if( timeStep <= 0.0 )
   {
      cerr << "Tau for tnlExplicitTimeStepper must be positive. " << endl;
      return false;
   }
   this->timeStep = timeStep;
   return true;
};

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
bool
tnlExplicitTimeStepper< Problem, OdeSolver >::
solve( const RealType& time,
       const RealType& stopTime,
       const MeshType& mesh,
       DofVectorType& dofVector,
       MeshDependentDataType& meshDependentData )
{
   tnlAssert( this->odeSolver, );
   mainTimer.start();
   this->odeSolver->setTau( this->timeStep );
   this->odeSolver->setProblem( * this );
   this->odeSolver->setTime( time );
   this->odeSolver->setStopTime( stopTime );
   if( this->odeSolver->getMinIterations() )
      this->odeSolver->setMaxTau( ( stopTime - time ) / ( typename OdeSolver< Problem >::RealType ) this->odeSolver->getMinIterations() );
   this->mesh = &mesh;
   this->meshDependentData = &meshDependentData;
   if( ! this->odeSolver->solve( dofVector ) )
      return false;
   this->problem->setExplicitBoundaryConditions( stopTime, *( this->mesh ), dofVector, *( this->meshDependentData ) );
   mainTimer.stop();
   this->allIterations += this->odeSolver->getIterations();
   return true;
}

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
void
tnlExplicitTimeStepper< Problem, OdeSolver >::
getExplicitRHS( const RealType& time,
                const RealType& tau,
                DofVectorType& u,
                DofVectorType& fu )
{
   this->preIterateTimer.start();
   if( ! this->problem->preIterate( time,
                                    tau,
                                    *( this->mesh),
                                    u,
                                    *( this->meshDependentData ) ) )
   {
      cerr << endl << "Preiteration failed." << endl;
      return;
      //return false; // TODO: throw exception
   }
   this->preIterateTimer.stop();
   this->explicitUpdaterTimer.start();
   this->problem->setExplicitBoundaryConditions( time, *( this->mesh ), u, *( this->meshDependentData ) );
   this->problem->getExplicitRHS( time, tau, *( this->mesh ), u, fu, *( this->meshDependentData ) );
   this->explicitUpdaterTimer.stop();
   this->postIterateTimer.start();
   if( ! this->problem->postIterate( time,
                                     tau,
                                     *( this->mesh ),
                                     u,
                                     *( this->meshDependentData ) ) )
   {
      cerr << endl << "Postiteration failed." << endl;
      return;
      //return false; // TODO: throw exception
   }
   this->postIterateTimer.stop();
}

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
bool
tnlExplicitTimeStepper< Problem, OdeSolver >::
writeEpilog( tnlLogger& logger )
{
   logger.writeParameter< long long int >( "Iterations count:", this->allIterations );
   logger.writeParameter< const char* >( "Pre-iterate time:", "" );
   this->preIterateTimer.writeLog( logger, 1 );
   logger.writeParameter< const char* >( "Explicit update computation:", "" );
   this->explicitUpdaterTimer.writeLog( logger, 1 );
   logger.writeParameter< const char* >( "Explicit time stepper time:", "" );
   this->mainTimer.writeLog( logger, 1 );
   logger.writeParameter< const char* >( "Post-iterate time:", "" );
   this->postIterateTimer.writeLog( logger, 1 );
   return true;
}

#endif /* TNLEXPLICITTIMESTEPPER_IMPL_H_ */
