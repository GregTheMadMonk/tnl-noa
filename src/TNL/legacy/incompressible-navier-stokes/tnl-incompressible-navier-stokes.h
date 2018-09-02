/***************************************************************************
                          tnl-incompressible-navier-stokes.h  -  description
                             -------------------
    begin                : Jan 28, 2015
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

#ifndef TNL_INCOMPRESSIBLE_NAVIER_STOKES_H_
#define TNL_INCOMPRESSIBLE_NAVIER_STOKES_H_

#include <solvers/tnlSolver.h>
#include <operators/diffusion/tnlLinearDiffusion.h>
#include "tnlIncompressibleNavierStokesProblem.h"
#include "tnlNSFastBuildConfig.h"

//typedef tnlDefaultConfigTag BuildConfig;
typedef tnlNSFastBuildConfig BuildConfig;

template< typename ConfigTag >
class tnlIncompressibleNavierStokesConfig
{
   public:
      static void configSetup( tnlConfigDescription& config )
      {
         config.addDelimiter( "Incompressible Navier-Stokes solver settings:" );
		 config.addEntry< double >( "viscosity", "Viscosity of the diffusion." );
		 config.addEntry< double >( "inletVelocity", "Maximal X velocity on the inlet." );

         /*config.addEntry< tnlString >( "boundary-conditions-type", "Choose the boundary conditions type.", "dirichlet");
            config.addEntryEnum< tnlString >( "dirichlet" );
            config.addEntryEnum< tnlString >( "neumann" );

         config.addEntry< tnlString >( "boundary-conditions-file", "File with the values of the boundary conditions.", "boundary.tnl" );
         config.addEntry< double >( "boundary-conditions-constant", "This sets a value in case of the constant boundary conditions." );
         config.addEntry< double >( "right-hand-side-constant", "This sets a constant value for the right-hand side.", 0.0 );
         config.addEntry< tnlString >( "initial-condition", "File with the initial condition.", "initial.tnl");*/
	  }
};

template< typename Mesh, typename Real = typename Mesh::RealType, typename Index = typename Mesh::IndexType >
class tnlINSBoundaryConditions{};

template< typename Mesh, typename Real = typename Mesh::RealType, typename Index = typename Mesh::IndexType >
class tnlINSRightHandSide{};

template< typename Mesh, typename Real = typename Mesh::RealType, typename Index = typename Mesh::IndexType >
class tnlIncompressibleNavierStokes
{
   public:
	  typedef Real RealType;
	  typedef typename Mesh::DeviceType DeviceType;
	  typedef Index IndexType;
};


template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter >
class tnlIncompressibleNavierStokesSetter
{
public:
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef tnlStaticVector< MeshType::Dimensions, Real > Vertex;

   static bool run( const tnlParameterContainer& parameters )
   {
      enum { Dimensions = MeshType::Dimensions };
      typedef tnlStaticVector < MeshType::Dimensions, Real > Vertex;

	  typedef tnlINSBoundaryConditions< MeshType > BoundaryConditions;
	  typedef tnlIncompressibleNavierStokes< MeshType > ApproximateOperator;
	  typedef tnlINSRightHandSide< MeshType > RightHandSide;
      typedef tnlIncompressibleNavierStokesProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Solver;
      SolverStarter solverStarter;
      return solverStarter.template run< Solver >( parameters );
   }
};

int main( int argc, char* argv[] )
{
   tnlSolver< tnlIncompressibleNavierStokesSetter, tnlIncompressibleNavierStokesConfig, BuildConfig > solver;
   if( ! solver. run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

#endif /* TNL_INCOMPRESSIBLE_NAVIER_STOKES_H_ */
