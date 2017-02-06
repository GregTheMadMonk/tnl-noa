/***************************************************************************
                          tnlTimeIndependentPDESolver_impl.h  -  description
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

#pragma once 

#include <solvers/pde/tnlTimeIndependentPDESolver.h>

template< typename Problem >
tnlTimeIndependentPDESolver< Problem >::
tnlTimeIndependentPDESolver()
: problem( 0 ),
  computeTimer( 0 )
{
}

template< typename Problem >
void
tnlTimeIndependentPDESolver< Problem >::
configSetup( tnlConfigDescription& config,
             const String& prefix )
{
}

template< typename Problem >
bool
tnlTimeIndependentPDESolver< Problem >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   /****
    * Load the mesh from the mesh file
    */
   const String& meshFile = parameters.getParameter< String >( "mesh" );
   cout << "Loading a mesh from the file " << meshFile << "...";
   if( ! this->mesh.load( meshFile ) )
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
   TNL_ASSERT( problem->getDofs( this->mesh ) != 0, );
   cout << "Allocating dofs ... ";
   if( ! this->dofs.setSize( problem->getDofs( this->mesh ) ) )
   {
      cerr << endl;
      cerr << "I am not able to allocate DOFs (degrees of freedom)." << endl;
      return false;
   }
   cout << " [ OK ]" << endl;
   this->dofs.setValue( 0.0 );
   this->problem->bindDofs( this->mesh, this->dofs );   
   
   /****
    * Set mesh dependent data
    */
   this->problem->setMeshDependentData( this->mesh, this->meshDependentData );
   this->problem->bindMeshDependentData( this->mesh, this->meshDependentData );
   
   /***
    * Set-up the initial condition
    */
   cout << "Setting up the initial condition ... ";
   typedef typename Problem :: DofVectorType DofVectorType;
   if( ! this->problem->setInitialData( parameters, this->mesh, this->dofs, this->meshDependentData ) )
      return false;
   cout << " [ OK ]" << endl;
   
   
   return true;
}

template< typename Problem >
bool
tnlTimeIndependentPDESolver< Problem >::
writeProlog( tnlLogger& logger,
             const Config::ParameterContainer& parameters )
{
   logger.writeHeader( problem->getPrologHeader() );
   problem->writeProlog( logger, parameters );
   logger.writeSeparator();
   mesh.writeProlog( logger );
   logger.writeSeparator();
   const String& solverName = parameters. getParameter< String >( "discrete-solver" );
   logger.writeParameter< String >( "Discrete solver:", "discrete-solver", parameters );
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
   logger.writeParameter< String >( "Real type:", "real-type", parameters, 0 );
   logger.writeParameter< String >( "Index type:", "index-type", parameters, 0 );
   logger.writeParameter< String >( "Device:", "device", parameters, 0 );
   logger.writeSeparator();
   logger.writeSystemInformation( parameters );
   logger.writeSeparator();
   logger.writeCurrentTime( "Started at:" );
   logger.writeSeparator();
   return true;
}

template< typename Problem >
void
tnlTimeIndependentPDESolver< Problem >::
setProblem( ProblemType& problem )
{
   this->problem = &problem;
}

template< typename Problem >
void tnlTimeIndependentPDESolver< Problem > :: setIoTimer( tnlTimer& ioTimer )
{
  // this->ioTimer = &ioTimer;
}

template< typename Problem >
void tnlTimeIndependentPDESolver< Problem > :: setComputeTimer( tnlTimer& computeTimer )
{
   this->computeTimer = &computeTimer;
}

template< typename Problem >
bool
tnlTimeIndependentPDESolver< Problem >::
solve()
{
   TNL_ASSERT( problem != 0,
              cerr << "No problem was set in tnlPDESolver." );

   this->computeTimer->reset();
   this->computeTimer->start();
   if( ! this->problem->solve( this->mesh, this->dofs ) )
   {
      this->computeTimer->stop();
      return false;
   }
   this->computeTimer->stop();
   return true;
}

template< typename Problem >
bool
tnlTimeIndependentPDESolver< Problem >::
writeEpilog( tnlLogger& logger ) const
{
   return this->problem->writeEpilog( logger );
}
