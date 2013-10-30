/***************************************************************************
                          navierStokesSolver.impl.h  -  description
                             -------------------
    begin                : Jan 13, 2013
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

#ifndef NAVIERSTOKESSOLVER_IMPL_H_
#define NAVIERSTOKESSOLVER_IMPL_H_


#include "navierStokesSolver.h"
#include <stdio.h>
#include <iostream>
#include <core/tnlString.h>
#include <core/mfilename.h>
#include <core/mfuncs.h>
#include <core/vectors/tnlSharedVector.h>
#include <solvers/ode/tnlMersonSolver.h>


#ifdef HAVE_CUDA

#include <cuda.h>

template< typename Real, typename Index >
__device__ void computeVelocityFieldCuda( const Index size,
                                          const Real R,
                                          const Real T,
                                          const Real* rho,
                                          const Real* rho_u1,
                                          const Real* rho_u2,
                                          Real* u1,
                                          Real* u2,
                                          Real* p );
#endif

template< typename Mesh, typename EulerScheme >
navierStokesSolver< Mesh, EulerScheme > :: navierStokesSolver()
: R( 0.0 ),
  T( 0.0 ),
  p_0( 0.0 )
{

   this -> mesh. setName( "navier-stokes-mesh" );
   this -> dofVector. setName( "navier-stokes-dof-vector" );
}

template< typename Mesh, typename EulerScheme >
   template< typename Geom >
bool navierStokesSolver< Mesh, EulerScheme > :: setMeshGeometry( Geom& geometry ) const
{
   return true;
}

template< typename Mesh, typename EulerScheme >
bool navierStokesSolver< Mesh, EulerScheme > :: setMeshGeometry( tnlLinearGridGeometry< 2, RealType, DeviceType, IndexType >& geometry ) const
{
   geometry. setNumberOfSegments( 3 );
   geometry. setSegmentData( 0, 0.0,  0.0,  1.0 );
   geometry. setSegmentData( 1, 0.5,  0.15, 0.85 );
   geometry. setSegmentData( 2, 1.0,  0.0,  1.0 );
   return true;
}


template< typename Mesh, typename EulerScheme >
bool navierStokesSolver< Mesh, EulerScheme > :: init( const tnlParameterContainer& parameters )
{
   cout << "Initiating solver ... " << endl;

   /****
    * Set-up problem type
    */
   const tnlString& problemName = parameters. GetParameter< tnlString >( "problem-name" );
   if( problemName == "riser" )
      problem = riser;
   if( problemName == "cavity" )
      problem = cavity;

   /****
    * Set-up the geometry
    */
   tnlTuple< 2, RealType > proportions;
   proportions. x() = parameters. GetParameter< double >( "width" );
   proportions. y() = parameters. GetParameter< double >( "height" );
   if( proportions. x() <= 0 )
   {
      cerr << "Error: width must be positive real number! It is " << proportions. x() << " now." << endl;
      return false;
   }
   if( proportions. y() <= 0 )
   {
      cerr << "Error: height must be positive real number! It is " << proportions. y() << " now." << endl;
      return false;
   }
   this -> mesh. setOrigin( tnlTuple< 2, RealType >( 0, 0 ) );
   this -> mesh. setProportions( proportions );

   /****
    * Set-up the space discretization
    */
   tnlTuple< 2, IndexType > meshes;
   meshes. x() = parameters. GetParameter< int >( "x-size" );
   meshes. y() = parameters. GetParameter< int >( "y-size" );
   if( meshes. x() <= 0 )
   {
      cerr << "Error: x-size must be positive integer number! It is " << meshes. x() << " now." << endl;
      return false;
   }
   if( meshes. y() <= 0 )
   {
      cerr << "Error: y-size must be positive integer number! It is " << meshes. y() << " now." << endl;
      return false;
   }
   this -> mesh. setDimensions( meshes. x(), meshes. y() );
   this -> setMeshGeometry( this -> mesh. getGeometry() );
   RealType hx = this -> mesh. getParametricStep(). x();
   RealType hy = this -> mesh. getParametricStep(). y();
   mesh. refresh();
   mesh. save( tnlString( "mesh.tnl" ) );

   /****
    * Set-up model coefficients
    */
   this->p_0 = parameters. GetParameter< double >( "p0" );
   navierStokesScheme.setMu( parameters. GetParameter< double >( "mu") );
   navierStokesScheme.setT( parameters. GetParameter< double >( "T") );
   navierStokesScheme.setR( parameters. GetParameter< double >( "R") );
   navierStokesScheme.setGravity( parameters. GetParameter< double >( "gravity") );
   if( ! this->boundaryConditions.init( parameters ) )
      return false;

   /****
    * Set-up grid functions
    */
   const IndexType variablesNumber = 3;

   dofVector. setSize( variablesNumber * mesh. getDofs() );
   rhsDofVector. setLike( dofVector );
   this->boundaryConditions.setMesh( this->mesh );

   /****
    * Set-up numerical scheme
    */
   pressureGradient.setFunction( navierStokesScheme.getPressure() );
   pressureGradient.bindMesh( this -> mesh );
   this->eulerScheme. bindMesh( this -> mesh );
   this->eulerScheme. setPressureGradient( this -> pressureGradient );
   this->u1Viscosity. bindMesh( this -> mesh );
   this->u1Viscosity. setFunction( this -> navierStokesScheme.getU1() );
   this->u2Viscosity. bindMesh( this -> mesh );
   this->u2Viscosity. setFunction( this -> navierStokesScheme.getU2() );
   navierStokesScheme.setAdvectionScheme( this->eulerScheme );
   navierStokesScheme.setMesh( this->mesh );
   //navierStokesScheme.setDifusionScheme( this)
   return true;
}

template< typename Mesh, typename EulerScheme >
bool navierStokesSolver< Mesh, EulerScheme > :: setInitialCondition( const tnlParameterContainer& parameters )
{
   tnlSharedVector< RealType, DeviceType, IndexType > dofs_rho, rho_u1, rho_u2;
   const IndexType& dofs = mesh. getDofs();
   dofs_rho. bind( & dofVector. getData()[ 0        ], dofs );
   rho_u1.   bind( & dofVector. getData()[     dofs ], dofs );
   rho_u2.   bind( & dofVector. getData()[ 2 * dofs ], dofs );

   dofs_rho. setValue( p_0 / ( this -> R * this -> T ) );
   rho_u1. setValue( 0.0 );
   rho_u2. setValue( 0.0 );

   const IndexType& xSize = mesh. getDimensions(). x();
   const IndexType& ySize = mesh. getDimensions(). y();
   const RealType hx = mesh. getParametricStep(). x();
   const RealType hy = mesh. getParametricStep(). y();

   for( IndexType j = 0; j < ySize; j ++ )
      for( IndexType i = 0; i < xSize; i ++ )
      {
         const IndexType c = mesh. getElementIndex( i, j );
         const RealType x = i * hx;
         const RealType y = j * hy;

         dofs_rho. setElement( c, p_0 / ( this -> R * this -> T ) );
         rho_u1. setElement( c, 0.0 );
         rho_u2. setElement( c, 0.0 );

      }
   return true;
}

template< typename Mesh, typename EulerScheme >
typename navierStokesSolver< Mesh, EulerScheme > :: DofVectorType& navierStokesSolver< Mesh, EulerScheme > :: getDofVector()
{
   return this -> dofVector;
}

template< typename Mesh, typename EulerScheme >
bool navierStokesSolver< Mesh, EulerScheme > :: makeSnapshot( const RealType& t,
                                                              const IndexType step )
{
   cout << endl << "Writing output at time " << t << " step " << step << "." << endl;
   tnlSharedVector< RealType, DeviceType, IndexType > rho, rho_u1, rho_u2,
                                                      rho_t, rho_u1_t, rho_u2_t;
   const IndexType& dofs = mesh. getDofs();
   rho.    bind( & dofVector.getData()[ 0        ], dofs );
   rho_u1. bind( & dofVector.getData()[     dofs ], dofs );
   rho_u2. bind( & dofVector.getData()[ 2 * dofs ], dofs );
   rho_t.    bind( & this->rhsDofVector.getData()[ 0        ], dofs );
   rho_u1_t. bind( & this->rhsDofVector.getData()[     dofs ], dofs );
   rho_u2_t. bind( & this->rhsDofVector.getData()[ 2 * dofs ], dofs );


   navierStokesScheme.updatePhysicalQuantities( rho, rho_u1, rho_u2 );
   tnlVector< RealType, DeviceType, IndexType > u;
   u. setLike( navierStokesScheme.getU1() );
   for( IndexType i = 0; i < navierStokesScheme.getU1().getSize(); i ++ )
   {
      const RealType& u1 = navierStokesScheme.getU1()[ i ];
      const RealType& u2 = navierStokesScheme.getU2()[ i ];
      u[ i ] = sqrt( u1 * u1 + u2 * u2 );
   }
   tnlString fileName;
   /*FileNameBaseNumberEnding( "u-1-", step, 5, ".tnl", fileName );
   if( ! this -> u1. save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "u-2-", step, 5, ".tnl", fileName );
   if( ! this -> u2. save( fileName ) )
      return false;*/
   /*FileNameBaseNumberEnding( "p-", step, 5, ".tnl", fileName );
   if( ! p. save( fileName ) )
      return false;*/
   FileNameBaseNumberEnding( "u-", step, 5, ".tnl", fileName );
   if( ! u. save( fileName ) )
      return false;
   /*FileNameBaseNumberEnding( "rho-t-", step, 5, ".tnl", fileName );
   if( ! rho_t. save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "rho-u1-t-", step, 5, ".tnl", fileName );
   if( ! rho_u1_t. save( fileName ) )
      return false;

   FileNameBaseNumberEnding( "rho-u2-t-", step, 5, ".tnl", fileName );
   if( ! rho_u2_t. save( fileName ) )
      return false;*/

   FileNameBaseNumberEnding( "rho-", step, 5, ".tnl", fileName );
   if( ! navierStokesScheme.getRho(). save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "rho-u1-", step, 5, ".tnl", fileName );
   if( ! rho_u1. save( fileName ) )
      return false;

   FileNameBaseNumberEnding( "rho-u2-", step, 5, ".tnl", fileName );
   if( ! rho_u2. save( fileName ) )
      return false;
   return true;
}

/*template< typename Mesh, typename EulerScheme >
   template< typename Vector >
void navierStokesSolver< Mesh, EulerScheme > :: updatePhysicalQuantities( const Vector& dofs_rho,
                                                                          const Vector& rho_u1,
                                                                          const Vector& rho_u2 )
{
   if( DeviceType :: getDevice() == tnlHostDevice )
   {
      const IndexType& xSize = mesh. getDimensions(). x();
      const IndexType& ySize = mesh. getDimensions(). y();

   #ifdef HAVE_OPENMP
   #pragma omp parallel for
   #endif
      for( IndexType j = 0; j < ySize; j ++ )
         for( IndexType i = 0; i < xSize; i ++ )
         {
            IndexType c = mesh. getElementIndex( i, j );
            this->rho[ c ] = dofs_rho[ c ];
            this->u1[ c ] = rho_u1[ c ] / this->rho[ c ];
            this->u2[ c ] = rho_u2[ c ] / this->rho[ c ];
            this->p[ c ] = this->rho[ c ] * this -> R * this -> T;
         }
   }
}*/

template< typename Mesh, typename EulerScheme >
void navierStokesSolver< Mesh, EulerScheme > :: GetExplicitRHS(  const RealType& time,
                                                                 const RealType& tau,
                                                                 DofVectorType& u,
                                                                 DofVectorType& fu )
{
   tnlSharedVector< RealType, DeviceType, IndexType > dofs_rho, rho_u1, rho_u2,
                                                      rho_t, rho_u1_t, rho_u2_t;

   const IndexType& dofs = mesh. getDofs();
   dofs_rho. bind( & u. getData()[ 0 ], dofs );
   rho_u1. bind( & u. getData()[ dofs ], dofs );
   rho_u2. bind( & u. getData()[ 2 * dofs ], dofs );

   eulerScheme. setRho( dofs_rho );
   eulerScheme. setRhoU1( rho_u1 );
   eulerScheme. setRhoU2( rho_u2 );

   rho_t. bind( & fu. getData()[ 0 ], dofs );
   rho_u1_t. bind( & fu. getData()[ dofs ], dofs );
   rho_u2_t. bind( & fu. getData()[ 2 * dofs ], dofs );

   navierStokesScheme.updatePhysicalQuantities( dofs_rho, rho_u1, rho_u2 );

   /****
    * Compute RHS
    */
   const IndexType& xSize = mesh. getDimensions(). x();
   const IndexType& ySize = mesh. getDimensions(). y();
   /*const RealType hx = mesh. getParametricStep(). x();
   const RealType hy = mesh. getParametricStep(). y();*/

   if( DeviceType :: getDevice() == tnlHostDevice )
   {
      this->boundaryConditions.apply( time,
                                      this->navierStokesScheme.getRho(),
                                      this->navierStokesScheme.getU1(),
                                      this->navierStokesScheme.getU2() );
      for( IndexType i = 0; i < xSize; i ++ )
      {
         const IndexType c1 = mesh.getElementIndex( i, 0 );
         const IndexType c2 = mesh.getElementIndex( i, 1 );
         const IndexType c3 = mesh.getElementIndex( i, ySize - 1 );
         const IndexType c4 = mesh.getElementIndex( i, ySize - 2 );

         dofs_rho[ c1 ] = this->navierStokesScheme.getRho()[ c1 ];
         rho_u1[ c1 ]   = this->navierStokesScheme.getRho()[ c1 ] * this->navierStokesScheme.getU1()[ c1 ];
         rho_u2[ c1 ]   = this->navierStokesScheme.getRho()[ c1 ] * this->navierStokesScheme.getU2()[ c1 ];
         dofs_rho[ c3 ] = this->navierStokesScheme.getRho()[ c3 ];
         rho_u1[ c3 ]   = this->navierStokesScheme.getRho()[ c3 ] * this->navierStokesScheme.getU1()[ c3 ];
         rho_u2[ c3 ]   = this->navierStokesScheme.getRho()[ c3 ] * this->navierStokesScheme.getU2()[ c3 ];
      }
      for( IndexType j = 0; j < ySize; j ++ )
      {
         const IndexType c1 = mesh.getElementIndex( 0, j );
         const IndexType c2 = mesh.getElementIndex( 1, j );
         const IndexType c3 = mesh.getElementIndex( xSize - 1, j );
         const IndexType c4 = mesh.getElementIndex( xSize - 2, j );

         dofs_rho[ c1 ] = this->navierStokesScheme.getRho()[ c1 ];
         rho_u1[ c1 ]   = this->navierStokesScheme.getRho()[ c1 ] * this->navierStokesScheme.getU1()[ c1 ];
         rho_u2[ c1 ]   = this->navierStokesScheme.getRho()[ c1 ] * this->navierStokesScheme.getU2()[ c1 ];
         dofs_rho[ c3 ] = this->navierStokesScheme.getRho()[ c3 ];
         rho_u1[ c3 ]   = this->navierStokesScheme.getRho()[ c3 ] * this->navierStokesScheme.getU1()[ c3 ];
         rho_u2[ c3 ]   = this->navierStokesScheme.getRho()[ c3 ] * this->navierStokesScheme.getU2()[ c3 ];

      }

   #ifdef HAVE_OPENMP
   #pragma omp parallel for
   #endif
      for( IndexType j = 0; j < ySize; j ++ )
         for( IndexType i = 0; i < xSize; i ++ )
         {
            IndexType c = mesh. getElementIndex( i, j );
            if( i == 0 || j == 0 ||
                i == xSize - 1 || j == ySize - 1 )
            {
               rho_t[ c ] = rho_u1_t[ c ] = rho_u2_t[ c ] = 0.0;
               continue;
            }

            eulerScheme. getExplicitRhs( c,
                                         rho_t[ c ],
                                         rho_u1_t[ c ],
                                         rho_u2_t[ c ],
                                         tau );
            
            //rho_u1_t[ c ] += ;
            //rho_u2_t[ c ] -= startUpCoefficient * this -> gravity * this -> rho[ c ];

            /***
             * Add the viscosity term
             */
            rho_u1_t[ c ] += this->navierStokesScheme.getMu() * u1Viscosity. getDiffusion( c );
            rho_u2_t[ c ] += this->navierStokesScheme.getMu() * u2Viscosity. getDiffusion( c );

         }
   }

   /*rhsDofVector = fu;
   makeSnapshot( 0.0, 1 );
   getchar();*/

}

template< typename Mesh, typename EulerScheme >
tnlSolverMonitor< typename navierStokesSolver< Mesh, EulerScheme > :: RealType,
                  typename navierStokesSolver< Mesh, EulerScheme > :: IndexType >* 
   navierStokesSolver< Mesh, EulerScheme > ::  getSolverMonitor()
{
   return &solverMonitor;
}

template< typename Mesh, typename EulerScheme >
tnlString navierStokesSolver< Mesh, EulerScheme > :: getTypeStatic()
{
   return tnlString( "navierStokesSolver< " ) +
          Mesh :: getTypeStatic() + " >";
}

template< typename Mesh, typename EulerScheme >
tnlString navierStokesSolver< Mesh, EulerScheme > :: getPrologHeader() const
{
   return tnlString( "Navier-Stokes Problem Solver" );
}

template< typename Mesh, typename EulerScheme >
void navierStokesSolver< Mesh, EulerScheme > :: writeProlog( tnlLogger& logger,
                                                             const tnlParameterContainer& parameters ) const
{
   logger. WriteParameter< tnlString >( "Problem name:", "problem-name", parameters );
   const tnlString& problemName = parameters. GetParameter< tnlString >( "problem-name" );
   if( problemName == "cavity" )
   {
      logger. WriteParameter< double >( "Max. inflow velocity:", "max-inflow-velocity", parameters, 1 );
   }
   logger. WriteParameter< double >( "Viscosity:", "mu", parameters );
   logger. WriteParameter< double >( "Temperature:", "T", parameters );
   logger. WriteParameter< double >( "Gass constant:", "R", parameters );
   logger. WriteParameter< double >( "Pressure:", "p0", parameters );
   logger. WriteParameter< double >( "Gravity:", "gravity", parameters );
   logger. WriteSeparator();
   logger. WriteParameter< double >( "Domain width:", mesh. getProportions(). x() - mesh. getOrigin(). x() );
   logger. WriteParameter< double >( "Domain height:", mesh. getProportions(). y() - mesh. getOrigin(). y() );
   logger. WriteSeparator();
   logger. WriteParameter< tnlString >( "Space discretisation:", "scheme", parameters );
   logger. WriteParameter< int >( "Meshes along x:", mesh. getDimensions(). x() );
   logger. WriteParameter< int >( "Meshes along y:", mesh. getDimensions(). y() );
   logger. WriteParameter< double >( "Space step along x:", mesh. getParametricStep(). x() );
   logger. WriteParameter< double >( "Space step along y:", mesh. getParametricStep(). y() );
}

#endif /* NAVIERSTOKESSOLVER_IMPL_H_ */
