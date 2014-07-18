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

template< typename Problem, typename TimeStepper >
tnlPDESolver< Problem, TimeStepper > :: tnlPDESolver()
: timeStepper( 0 ),
  finalTime( 0.0 ),
  snapshotTau( 0.0 ),
  problem( 0 ),
  ioRtTimer( 0 ),
  computeRtTimer( 0 ),
  ioCpuTimer( 0 ),
  computeCpuTimer( 0 )
{
}

template< typename Problem, typename TimeStepper >
void tnlPDESolver< Problem, TimeStepper >::configSetup( tnlConfigDescription& config,
                                                        const tnlString& prefix )
{
   config.addRequiredEntry< double >( prefix + "final-time", "Stop time of the time dependent problem." );
   config.addRequiredEntry< double >( prefix + "snapshot-period", "Time period for writing the problem status.");
}

template< typename Problem, typename TimeStepper >
bool tnlPDESolver< Problem, TimeStepper >::init( const tnlParameterContainer& parameters,
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
   if( ! this->dofs.setSize( problem->getDofs( this->mesh ) ) )
   {
      cerr << "I am not able to allocate DOFs (degrees of freedom)." << endl;
      return false;
   }


   /****
    * Initialize the time discretisation
    */
   this->setFinalTime( parameters.GetParameter< double >( "final-time" ) );
   this->setSnapshotTau( parameters.GetParameter< double >( "snapshot-period" ) );



}

template< typename Problem, typename TimeStepper >
void tnlPDESolver< Problem, TimeStepper >::setTimeStepper( TimeStepper& timeStepper )
{
   this -> timeStepper = &timeStepper;
}

template< typename Problem, typename TimeStepper >
void tnlPDESolver< Problem, TimeStepper >::setProblem( ProblemType& problem )
{
   this -> problem = &problem;
}

template< typename Problem, typename TimeStepper >
bool tnlPDESolver< Problem, TimeStepper >::setFinalTime( const RealType& finalT )
{
   if( finalT <= 0 )
   {
      cerr << "Final time for tnlPDESolver must be positive value." << endl;
      return false;
   }
   this -> finalTime = finalT;
}

template< typename Problem, typename TimeStepper >
const typename TimeStepper :: RealType& tnlPDESolver< Problem, TimeStepper > :: getFinalTine() const
{
   return this -> finalTime;
}

template< typename Problem, typename TimeStepper >
bool tnlPDESolver< Problem, TimeStepper > :: setSnapshotTau( const RealType& tau )
{
   if( tau <= 0 )
   {
      cerr << "Snapshot tau for tnlPDESolver must be positive value." << endl;
      return false;
   }
   this -> snapshotTau = tau;
}
   
template< typename Problem, typename TimeStepper >
const typename TimeStepper :: RealType& tnlPDESolver< Problem, TimeStepper > :: getSnapshotTau() const
{
   return this -> snapshotTau;
}

template< typename Problem, typename TimeStepper >
void tnlPDESolver< Problem, TimeStepper > :: setIoRtTimer( tnlTimerRT& ioRtTimer)
{
   this -> ioRtTimer = &ioRtTimer;
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

   if( snapshotTau == 0 )
   {
      cerr << "No snapshot tau was set in tnlPDESolver " << this -> getName() << "." << endl;
      return false;
   }
   RealType t( 0.0 );
   IndexType step( 0 );
   IndexType allSteps = ceil( this -> finalTime / this -> snapshotTau );
   this -> timeStepper -> setProblem( * ( this -> problem ) );
   if( ! this -> problem -> makeSnapshot( t, step ) )
   {
      cerr << "Making the snapshot failed." << endl;
      return false;
   }
   while( step < allSteps )
   {
      RealType tau = Min( this -> snapshotTau,
                          this -> finalTime - t );
      //this -> timeStepper -> setTau( tau );
      if( ! this -> timeStepper -> solve( t, t + tau ) )
         return false;
      step ++;
      t += tau;

      this -> ioRtTimer -> Continue();
      this -> ioCpuTimer -> Continue();
      this -> computeRtTimer -> Stop();
      this -> computeCpuTimer -> Stop();

      if( ! this -> problem -> makeSnapshot( t, step ) )
      {
         cerr << "Making the snapshot failed." << endl;
         return false;
      }

      this -> ioRtTimer -> Stop();
      this -> ioCpuTimer -> Stop();
      this -> computeRtTimer -> Continue();
      this -> computeCpuTimer -> Continue();

   }
   return true;
}

#endif /* TNLPDESOLVER_IMPL_H_ */
