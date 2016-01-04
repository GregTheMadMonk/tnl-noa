/***************************************************************************
                          tnlExplicitINSTimeStepper.h  -  description
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

#ifndef EXAMPLES_INCOMPRESSIBLE_NAVIER_STOKES_TNLEXPLICITINSTIMESTEPPER_H_
#define EXAMPLES_INCOMPRESSIBLE_NAVIER_STOKES_TNLEXPLICITINSTIMESTEPPER_H_

template< typename Problem,
          typename LinearSolver >
class tnlExplicitINSTimeStepper
{
   public:

   typedef Problem ProblemType;
   typedef typename Problem::RealType RealType;
   typedef typename Problem::DeviceType DeviceType;
   typedef typename Problem::IndexType IndexType;
   typedef typename Problem::MeshType MeshType;
   typedef typename ProblemType::DofVectorType DofVectorType;

   tnlExplicitINSTimeStepper(): problem(0), timeStep(0) {}

   static void configSetup( tnlConfigDescription& config, const tnlString& prefix = "" )
   {
	   config.addEntry< bool >( "verbose", "Verbose mode.", true );
   }

   bool setup( const tnlParameterContainer& parameters,
			  const tnlString& prefix = "" )
   {
	   //this->verbose = parameters.getParameter< bool >( "verbose" );
	   return true;
   }

   bool init( const MeshType& mesh )
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

   void setProblem( ProblemType& problem ) {this -> problem = &problem;}
   ProblemType* getProblem() const {return this -> problem;}

   bool setTimeStep( const RealType& timeStep )
   {
	   if( timeStep <= 0.0 )
	   {
		  cerr << "Time step for tnlExplicitINSTimeStepper must be positive. " << endl;
		  return false;
	   }
	   this->timeStep = timeStep;
	   return true;
   }

   const RealType& getTimeStep() const;

   bool solve( const RealType& time,
               const RealType& stopTime,
               const MeshType& mesh,
               DofVectorType& dofVector,
			   DofVectorType& auxiliaryDofVector )
   {
	   tnlAssert( this->problem != 0, );
	   RealType t = time;
	   while( t < stopTime )
	   {
		  RealType currentTau = Min( this->timeStep, stopTime - t );
		  currentTau = 0.005;

		  this->problem->doStep(currentTau,mesh);

		  t += currentTau;
	   }
	   return true;
   }

   bool writeEpilog( tnlLogger& logger ) const { return true; }

   protected:

   Problem* problem;
   //LinearSolver _matSolver;
   RealType timeStep;

};

#include "tnlExplicitINSTimeStepper_impl.h"

#endif /* EXAMPLES_INCOMPRESSIBLE_NAVIER_STOKES_TNLEXPLICITINSTIMESTEPPER_H_ */
