/***************************************************************************
                          tnlExplicitINSTimeStepper_impl.h  -  description
                             -------------------
    begin                : Feb 17, 2015
    copyright            : (C) 2015 by oberhuber
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

#ifndef EXAMPLES_INCOMPRESSIBLE_NAVIER_STOKES_TNLEXPLICITINSTIMESTEPPER_IMPL_H_
#define EXAMPLES_INCOMPRESSIBLE_NAVIER_STOKES_TNLEXPLICITINSTIMESTEPPER_IMPL_H_

#include "tnlExplicitINSTimeStepper.h"
template< typename Problem,
          typename LinearSystemSolver >
tnlExplicitINSTimeStepper< Problem, LinearSystemSolver >::
tnlExplicitINSTimeStepper()
: problem( 0 ),
  timeStep( 0 )
{
};

template< typename Problem,
          typename LinearSystemSolver >
void
tnlExplicitINSTimeStepper< Problem, LinearSystemSolver >::
configSetup( tnlConfigDescription& config,
             const tnlString& prefix ) //Pridavani parametru prikazove radky
{
   config.addEntry< bool >( "verbose", "Verbose mode.", true );
}

template< typename Problem,
          typename LinearSystemSolver >
bool
tnlExplicitINSTimeStepper< Problem, LinearSystemSolver >::
setup( const tnlParameterContainer& parameters,
      const tnlString& prefix ) //Nacteni parametru prikazove radky
{
   //this->verbose = parameters.getParameter< bool >( "verbose" );
   return true;
}

template< typename Problem,
          typename LinearSystemSolver >
bool
tnlExplicitINSTimeStepper< Problem, LinearSystemSolver >::
init( const MeshType& mesh ) //Inicializace time stepperu - vytvoreni matic podle site
{
   /*cout << "Setting up the linear system...";
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
      return false;*/
   return true;
}

template< typename Problem,
          typename LinearSystemSolver >
void
tnlExplicitINSTimeStepper< Problem, LinearSystemSolver >::
setProblem( ProblemType& problem ) //Nesahej
{
   this -> problem = &problem;
};

template< typename Problem,
          typename LinearSystemSolver >
Problem*
tnlExplicitINSTimeStepper< Problem, LinearSystemSolver >::
getProblem() const
{
    return this -> problem;
};

template< typename Problem,
          typename LinearSystemSolver >
bool
tnlExplicitINSTimeStepper< Problem, LinearSystemSolver >::
setTimeStep( const RealType& timeStep )
{
   if( timeStep <= 0.0 )
   {
      cerr << "Time step for tnlExplicitINSTimeStepper must be positive. " << endl;
      return false;
   }
   this->timeStep = timeStep;
   return true;
};

template< typename Problem,
          typename LinearSystemSolver >
bool
tnlExplicitINSTimeStepper< Problem, LinearSystemSolver >::
solve( const RealType& time,
       const RealType& stopTime,
       const MeshType& mesh,
       DofVectorType& dofVector,
       DofVectorType& auxiliaryDofVector )   //Hlavni cast, kterou bude potreba menit
{
   tnlAssert( this->problem != 0, );
   RealType t = time;
   //this->_matSolver.setMatrix(this->matrix);
   while( t < stopTime )
   {

      RealType currentTau = Min( this->timeStep, stopTime - t );
	  currentTau = 0.005;

	  this->problem->diffuse(currentTau,mesh);
	  this->problem->project(mesh);
	  this->problem->advect(currentTau, mesh);
	  this->problem->project(mesh);

	  /*/if( ! this->_matSolver->template solve< DofVectorType, tnlLinearResidueGetter< typename Problem::MatrixType, DofVectorType > >( _rightHandSide, auxiliaryDofVector ) )
	  {
		 cerr << endl << "The linear system solver did not converge." << endl;
		 return false;
	  }*/
	  //this->problem->Project(auxiliaryDofVector);

	  /*if( ! this->problem->preIterate( t,
                                       currentTau,
                                       mesh,
                                       dofVector,
                                       auxiliaryDofVector ) )
      {
         cerr << endl << "Preiteration failed." << endl;
         return false;
      }
      if( verbose )
         cout << "                                                                  Assembling the linear system ... \r" << flush;
      this->problem->assemblyLinearSystem( t,
                                           currentTau,
                                           mesh,
                                           dofVector,
                                           auxiliaryDofVector,
                                           this->matrix,
                                           this->rightHandSide );
      if( verbose )
         cout << "                                                                  Solving the linear system for time " << t << "             \r" << flush;
	  if( ! this->matSolver->template solve< DofVectorType, tnlLinearResidueGetter< MatrixType, DofVectorType > >( this->rightHandSide, dofVector ) )
      {
         cerr << endl << "The linear system solver did not converge." << endl;
         return false;
      }
      //if( verbose )
      //   cout << endl;
      if( ! this->problem->postIterate( t,
                                        currentTau,
                                        mesh,
                                        dofVector,
                                        auxiliaryDofVector ) )
      {
         cerr << endl << "Postiteration failed." << endl;
         return false;
	  }*/
      t += currentTau;
   }
   return true;
}



#endif /* EXAMPLES_INCOMPRESSIBLE_NAVIER_STOKES_TNLEXPLICITINSTIMESTEPPER_IMPL_H_ */
