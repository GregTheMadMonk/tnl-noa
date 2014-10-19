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

template< typename Problem,
          typename TimeStepper >
tnlPDESolver< Problem, TimeStepper >::
tnlPDESolver()
: timeStepper( 0 ),
  finalTime( 0.0 ),
  snapshotPeriod( 0.0 ),
  timeStep( 1.0 ),
  timeStepOrder( 0.0 ),
  problem( 0 ),
  ioRtTimer( 0 ),
  computeRtTimer( 0 ),
  ioCpuTimer( 0 ),
  computeCpuTimer( 0 )
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
   config.addRequiredEntry< double >( prefix + "snapshot-period", "Time period for writing the problem status.");
   config.addEntry< double >( "tau", "The time step for the time discretisation.", 1.0 );
   config.addEntry< double >( "tau-order", "The time step is set to tau*pow( space-step, tau-order).", 0.0 );
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
   const tnlString& meshFile = parameters.GetParameter< tnlString >( "mesh" );
   if( ! this->mesh.load( meshFile ) )
   {
      cerr << "I am not able to load the mesh from the file " << meshFile << "." << endl;
      cerr << " You may create it with tools like tnl-grid-setup or tnl-mesh-convert." << endl;
      return false;
   }

   /****
    * Set DOFs (degrees of freedom)
    */
   tnlAssert( problem->getDofs( this->mesh ) != 0, );
   if( ! this->dofs.setSize( problem->getDofs( this->mesh ) ) ||
       ! this->auxiliaryDofs.setSize( problem->getAuxiliaryDofs( this->mesh ) ) )
   {
      cerr << "I am not able to allocate DOFs (degrees of freedom)." << endl;
      return false;
   }
   this->problem->bindDofs( mesh, this->dofs );
   this->problem->bindAuxiliaryDofs( mesh, this->auxiliaryDofs );
   
   /***
    * Set-up the initial condition
    */
   typedef typename Problem :: DofVectorType DofVectorType;
   if( ! this->problem->setInitialCondition( parameters, mesh, this->dofs ) )
      return false;

   /****
    * Initialize the time discretisation
    */
   this->setFinalTime( parameters.GetParameter< double >( "final-time" ) );
   this->setSnapshotPeriod( parameters.GetParameter< double >( "snapshot-period" ) );
   this->setTimeStep( parameters.GetParameter< double >( "time-step") );
   this->setTimeStepOrder( parameters.GetParameter< double >( "time-step-order" ) );
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
   mesh.writeProlog( logger );
   logger.writeSeparator();
   logger.writeParameter< tnlString >( "Time discretisation:", "time-discretisation", parameters );
   logger.writeParameter< double >( "Initial time step:", "time-step", parameters );
   logger.writeParameter< double >( "Final time:", "final-time", parameters );
   logger.writeParameter< double >( "Snapshot period:", "snapshot-period", parameters );
   const tnlString& solverName = parameters. GetParameter< tnlString >( "discrete-solver" );
   logger.writeParameter< tnlString >( "Discrete solver:", "discrete-solver", parameters );
   if( solverName == "merson" )
      logger.writeParameter< double >( "Adaptivity:", "merson-adaptivity", parameters, 1 );
   if( solverName == "sor" )
      logger.writeParameter< double >( "Omega:", "sor-omega", parameters, 1 );
   if( solverName == "gmres" )
      logger.writeParameter< int >( "Restarting:", "gmres-restarting", parameters, 1 );
   logger.writeSeparator();
   logger.writeParameter< tnlString >( "Real type:", "real-type", parameters, 0 );
   logger.writeParameter< tnlString >( "Index type:", "index-type", parameters, 0 );
   logger.writeParameter< tnlString >( "Device:", "device", parameters, 0 );
   logger.writeSeparator();
   logger.writeSystemInformation();
   logger.writeSeparator();
   logger.writeCurrentTime( "Started at:" );
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
bool
tnlPDESolver< Problem, TimeStepper >::
setFinalTime( const RealType& finalTime )
{
   if( finalTime <= 0 )
   {
      cerr << "Final time for tnlPDESolver must be positive value." << endl;
      return false;
   }
   this->finalTime = finalTime;
}

template< typename Problem,
          typename TimeStepper >
const typename TimeStepper :: RealType&
tnlPDESolver< Problem, TimeStepper >::
getFinalTine() const
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
   if( timeStepOrder <= 0 )
   {
      cerr << "The time step order for tnlPDESolver must be positive value." << endl;
      return false;
   }
   this->timeStepOrder = timeStepOrder;
}

template< typename Problem,
          typename TimeStepper >
const typename TimeStepper::RealType&
tnlPDESolver< Problem, TimeStepper >::
getTimeStepOrder() const
{
   return this->timeStepOrder;
}

template< typename Problem,
         typename TimeStepper >
void
tnlPDESolver< Problem, TimeStepper >::
setIoRtTimer( tnlTimerRT& ioRtTimer )
{
   this->ioRtTimer = &ioRtTimer;
}

template< typename Problem, typename TimeStepper >
void tnlPDESolver< Problem, TimeStepper > :: setComputeRtTimer( tnlTimerRT& computeRtTimer )
{
   this -> computeRtTimer = &computeRtTimer;
}

template< typename Problem, typename TimeStepper >
void tnlPDESolver< Problem, TimeStepper > :: setIoCpuTimer( tnlTimerCPU& ioCpuTimer )
{
   this -> ioCpuTimer = &ioCpuTimer;
}

template< typename Problem, typename TimeStepper >
void tnlPDESolver< Problem, TimeStepper > :: setComputeCpuTimer( tnlTimerCPU& computeCpuTimer )
{
   this -> computeCpuTimer = & computeCpuTimer;
}

template< typename Problem, typename TimeStepper >
bool tnlPDESolver< Problem, TimeStepper > :: solve()
{
   tnlAssert( timeStepper != 0,
              cerr << "No time stepper was set in tnlPDESolver with name " << this -> getName() );
   tnlAssert( problem != 0,
              cerr << "No problem was set in tnlPDESolver with name " << this -> getName() );

   if( snapshotPeriod == 0 )
   {
      cerr << "No snapshot tau was set in tnlPDESolver " << this -> getName() << "." << endl;
      return false;
   }
   RealType t( 0.0 );
   IndexType step( 0 );
   IndexType allSteps = ceil( this->finalTime / this->snapshotPeriod );
   this->timeStepper->setProblem( * ( this->problem ) );
   this->timeStepper->init( mesh );
   this->problem->bindDofs( mesh, this->dofs );
   this->problem->bindAuxiliaryDofs( mesh, this->auxiliaryDofs );

   if( ! this->problem->makeSnapshot( t, step, mesh ) )
   {
      cerr << "Making the snapshot failed." << endl;
      return false;
   }
   timeStepper->setTimeStep( this->timeStep * pow( mesh.getSmallestSpaceStep(), this->timeStepOrder ) );
   while( step < allSteps )
   {
      RealType tau = Min( this -> snapshotPeriod,
                          this -> finalTime - t );
      if( ! this->timeStepper->solve( t, t + tau, mesh, dofs ) )
         return false;
      step ++;
      t += tau;

      this->ioRtTimer->Continue();
      this->ioCpuTimer->Continue();
      this->computeRtTimer->Stop();
      this->computeCpuTimer->Stop();

      if( ! this->problem->makeSnapshot( t, step, mesh ) )
      {
         cerr << "Making the snapshot failed." << endl;
         return false;
      }

      this-> ioRtTimer->Stop();
      this-> ioCpuTimer->Stop();
      this-> computeRtTimer->Continue();
      this-> computeCpuTimer->Continue();

   }
   return true;
}

#endif /* TNLPDESOLVER_IMPL_H_ */
