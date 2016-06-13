/***************************************************************************
                          tnlPDESolver_impl.h  -  description
                             -------------------
    begin                : Jan 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLPDESOLVER_IMPL_H_
#define TNLPDESOLVER_IMPL_H_

#include "tnlPDESolver.h"


template< typename Problem,
          typename TimeStepper >
tnlPDESolver< Problem, TimeStepper >::
tnlPDESolver()
: timeStepper( 0 ),
  initialTime( 0.0 ),
  finalTime( 0.0 ),
  snapshotPeriod( 0.0 ),
  timeStep( 1.0 ),
  timeStepOrder( 0.0 ),
  problem( 0 ),
  ioTimer( 0 ),
  computeTimer( 0 )
{
}

template< typename Problem,
          typename TimeStepper >
void
tnlPDESolver< Problem, TimeStepper >::
configSetup( tnlConfigDescription& config,
             const tnlString& prefix )
{
   config.addEntry< tnlString >( prefix + "initial-condition", "File name with the initial condition.", "init.tnl" );
   config.addRequiredEntry< double >( prefix + "final-time", "Stop time of the time dependent problem." );
   config.addEntry< double >( prefix + "initial-time", "Initial time of the time dependent problem.", 0 );
   config.addRequiredEntry< double >( prefix + "snapshot-period", "Time period for writing the problem status.");
   config.addEntry< double >( "time-step", "The time step for the time discretisation.", 1.0 );
   config.addEntry< double >( "time-step-order", "The time step is set to time-step*pow( space-step, time-step-order).", 0.0 );
}

template< typename Problem,
          typename TimeStepper >
bool
tnlPDESolver< Problem, TimeStepper >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   /****
    * Load the mesh from the mesh file
    */
   const tnlString& meshFile = parameters.getParameter< tnlString >( "mesh" );
   cout << "Loading a mesh from the file " << meshFile << "...";
   if( ! this->meshPointer->load( meshFile ) )
   {
      cerr << endl;
      cerr << "I am not able to load the mesh from the file " << meshFile << "." << endl;
      cerr << " You may create it with tools like tnl-grid-setup or tnl-mesh-convert." << endl;
      return false;
   }
   cout << " [ OK ] " << endl;

   /****
    * Set DOFs (degrees of freedom)
    */
   tnlAssert( problem->getDofs( this->meshPointer ) != 0, );
   cout << "Allocating dofs ... ";
   if( ! this->dofsPointer->setSize( problem->getDofs( this->meshPointer ) ) )
   {
      cerr << endl;
      cerr << "I am not able to allocate DOFs (degrees of freedom)." << endl;
      return false;
   }
   cout << " [ OK ]" << endl;
   this->dofsPointer->setValue( 0.0 );
   this->problem->bindDofs( this->meshPointer, this->dofsPointer );
   
   /****
    * Set mesh dependent data
    */
   this->problem->setMeshDependentData( *this->meshPointer, this->meshDependentData );
   this->problem->bindMeshDependentData( *this->meshPointer, this->meshDependentData );
   
   /***
    * Set-up the initial condition
    */
   cout << "Setting up the initial condition ... ";
   typedef typename Problem :: DofVectorType DofVectorType;
   if( ! this->problem->setInitialCondition( parameters, meshPointer, this->dofsPointer, this->meshDependentData ) )
      return false;
   cout << " [ OK ]" << endl;

   /****
    * Initialize the time discretisation
    */
   this->setFinalTime( parameters.getParameter< double >( "final-time" ) );
   this->setInitialTime( parameters.getParameter< double >( "initial-time" ) );
   this->setSnapshotPeriod( parameters.getParameter< double >( "snapshot-period" ) );
   this->setTimeStep( parameters.getParameter< double >( "time-step") );
   this->setTimeStepOrder( parameters.getParameter< double >( "time-step-order" ) );
   return true;
}

template< typename Problem,
          typename TimeStepper >
bool
tnlPDESolver< Problem, TimeStepper >::
writeProlog( tnlLogger& logger,
             const tnlParameterContainer& parameters )
{
   logger.writeHeader( problem->getPrologHeader() );
   problem->writeProlog( logger, parameters );
   logger.writeSeparator();
   meshPointer->writeProlog( logger );
   logger.writeSeparator();
   logger.writeParameter< tnlString >( "Time discretisation:", "time-discretisation", parameters );
   logger.writeParameter< double >( "Initial time step:", this->timeStep * pow( meshPointer->getSmallestSpaceStep(), this->timeStepOrder ) );
   logger.writeParameter< double >( "Initial time:", "initial-time", parameters );
   logger.writeParameter< double >( "Final time:", "final-time", parameters );
   logger.writeParameter< double >( "Snapshot period:", "snapshot-period", parameters );
   const tnlString& solverName = parameters. getParameter< tnlString >( "discrete-solver" );
   logger.writeParameter< tnlString >( "Discrete solver:", "discrete-solver", parameters );
   if( solverName == "merson" )
      logger.writeParameter< double >( "Adaptivity:", "merson-adaptivity", parameters, 1 );
   if( solverName == "sor" )
      logger.writeParameter< double >( "Omega:", "sor-omega", parameters, 1 );
   if( solverName == "gmres" )
      logger.writeParameter< int >( "Restarting:", "gmres-restarting", parameters, 1 );
   logger.writeParameter< double >( "Convergence residue:", "convergence-residue", parameters );
   logger.writeParameter< double >( "Divergence residue:", "divergence-residue", parameters );
   logger.writeParameter< int >( "Maximal number of iterations:", "max-iterations", parameters );
   logger.writeParameter< int >( "Minimal number of iterations:", "min-iterations", parameters );
   logger.writeSeparator();
   logger.writeParameter< tnlString >( "Real type:", "real-type", parameters, 0 );
   logger.writeParameter< tnlString >( "Index type:", "index-type", parameters, 0 );
   logger.writeParameter< tnlString >( "Device:", "device", parameters, 0 );
   logger.writeSeparator();
   logger.writeSystemInformation( parameters );
   logger.writeSeparator();
   logger.writeCurrentTime( "Started at:" );
   logger.writeSeparator();
   return true;
}

template< typename Problem,
          typename TimeStepper >
void
tnlPDESolver< Problem, TimeStepper >::
setTimeStepper( TimeStepper& timeStepper )
{
   this->timeStepper = &timeStepper;
}

template< typename Problem,
          typename TimeStepper >
void
tnlPDESolver< Problem, TimeStepper >::
setProblem( ProblemType& problem )
{
   this->problem = &problem;
}

template< typename Problem,
          typename TimeStepper >
void
tnlPDESolver< Problem, TimeStepper >::
setInitialTime( const RealType& initialTime )
{
   this->initialTime = initialTime;
}

template< typename Problem,
          typename TimeStepper >
const typename TimeStepper :: RealType&
tnlPDESolver< Problem, TimeStepper >::
getInitialTime() const
{
   return this->initialTime;
}


template< typename Problem,
          typename TimeStepper >
bool
tnlPDESolver< Problem, TimeStepper >::
setFinalTime( const RealType& finalTime )
{
   if( finalTime <= this->initialTime )
   {
      cerr << "Final time for tnlPDESolver must larger than the initial time which is now " << this->initialTime << "." << endl;
      return false;
   }
   this->finalTime = finalTime;
   return true;
}

template< typename Problem,
          typename TimeStepper >
const typename TimeStepper :: RealType&
tnlPDESolver< Problem, TimeStepper >::
getFinalTime() const
{
   return this->finalTime;
}

template< typename Problem,
          typename TimeStepper >
bool
tnlPDESolver< Problem, TimeStepper >::
setSnapshotPeriod( const RealType& period )
{
   if( period <= 0 )
   {
      cerr << "Snapshot tau for tnlPDESolver must be positive value." << endl;
      return false;
   }
   this->snapshotPeriod = period;
   return true;
}

template< typename Problem,
          typename TimeStepper >
const typename TimeStepper::RealType&
tnlPDESolver< Problem, TimeStepper >::
getSnapshotPeriod() const
{
   return this->snapshotPeriod;
}

template< typename Problem,
          typename TimeStepper >
bool
tnlPDESolver< Problem, TimeStepper >::
setTimeStep( const RealType& timeStep )
{
   if( timeStep <= 0 )
   {
      cerr << "The time step for tnlPDESolver must be positive value." << endl;
      return false;
   }
   this->timeStep = timeStep;
   return true;
}
   
template< typename Problem,
          typename TimeStepper >
const typename TimeStepper::RealType&
tnlPDESolver< Problem, TimeStepper >::
getTimeStep() const
{
   return this->timeStep;
}

template< typename Problem,
          typename TimeStepper >
bool
tnlPDESolver< Problem, TimeStepper >::
setTimeStepOrder( const RealType& timeStepOrder )
{
   if( timeStepOrder < 0 )
   {
      cerr << "The time step order for tnlPDESolver must be zero or positive value." << endl;
      return false;
   }
   this->timeStepOrder = timeStepOrder;
   return true;
}

template< typename Problem,
          typename TimeStepper >
const typename TimeStepper::RealType&
tnlPDESolver< Problem, TimeStepper >::
getTimeStepOrder() const
{
   return this->timeStepOrder;
}

template< typename Problem, typename TimeStepper >
void tnlPDESolver< Problem, TimeStepper > :: setIoTimer( tnlTimer& ioTimer )
{
   this->ioTimer = &ioTimer;
}

template< typename Problem, typename TimeStepper >
void tnlPDESolver< Problem, TimeStepper > :: setComputeTimer( tnlTimer& computeTimer )
{
   this->computeTimer = &computeTimer;
}

template< typename Problem, typename TimeStepper >
bool
tnlPDESolver< Problem, TimeStepper >::
solve()
{
   tnlAssert( timeStepper != 0,
              cerr << "No time stepper was set in tnlPDESolver." );
   tnlAssert( problem != 0,
              cerr << "No problem was set in tnlPDESolver." );

   if( snapshotPeriod == 0 )
   {
      cerr << "No snapshot tau was set in tnlPDESolver." << endl;
      return false;
   }
   RealType t( this->initialTime );
   IndexType step( 0 );
   IndexType allSteps = ceil( ( this->finalTime - this->initialTime ) / this->snapshotPeriod );

   this->ioTimer->reset();
   this->computeTimer->reset();
   
   this->ioTimer->start();
   if( ! this->problem->makeSnapshot( t, step, meshPointer, this->dofsPointer, this->meshDependentData ) )
   {
      cerr << "Making the snapshot failed." << endl;
      return false;
   }   
   this->ioTimer->stop();
   this->computeTimer->start();

   /****
    * Initialize the time stepper
    */
   this->timeStepper->setProblem( * ( this->problem ) );
   this->timeStepper->init( this->meshPointer );
   this->timeStepper->setTimeStep( this->timeStep * pow( this->meshPointer.getData().getSmallestSpaceStep(), this->timeStepOrder ) );
   while( step < allSteps )
   {
      RealType tau = Min( this->snapshotPeriod,
                          this->finalTime - t );
      if( ! this->timeStepper->solve( t, t + tau, this->meshPointer, this->dofsPointer, this->meshDependentData ) )
         return false;
      step ++;
      t += tau;

      this->ioTimer->start();
      this->computeTimer->stop();
      if( ! this->problem->makeSnapshot( t, step, this->meshPointer, this->dofsPointer, this->meshDependentData ) )
      {
         cerr << "Making the snapshot failed." << endl;
         return false;
      }
      this->ioTimer->stop();
      this->computeTimer->start();
   }
   this->computeTimer->stop();
   return true;
}

template< typename Problem, typename TimeStepper >
bool
tnlPDESolver< Problem, TimeStepper >::
writeEpilog( tnlLogger& logger ) const
{
   return ( this->timeStepper->writeEpilog( logger ) &&
      this->problem->writeEpilog( logger ) );
}

#endif /* TNLPDESOLVER_IMPL_H_ */
