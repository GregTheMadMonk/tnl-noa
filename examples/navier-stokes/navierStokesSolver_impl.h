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
#include <core/tnlSharedVector.h>
#include <solvers/ode/tnlMersonSolver.h>
#include <legacy/mesh/tnlGridOld.h>


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
: mu( 0.0 ),
  R( 0.0 ),
  T( 0.0 ),
  p_0( 0.0 ),
  gravity( 0.0 )
{
   this -> rho. setName( "rho-g" );
   this -> u1. setName( "u1-g");
   this -> u2. setName( "u2-g" );
   this -> p. setName( "p" );
   this -> mesh. setName( "navier-stokes-mesh" );
   this -> dofVector. setName( "navier-stokes-dof-vector" );
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
   this -> mesh. setDimensions( meshes. y(), meshes. x() );
   RealType hx = this -> mesh. getParametricStep(). x();
   RealType hy = this -> mesh. getParametricStep(). y();
   mesh. save( tnlString( "mesh.tnl" ) );

   /****
    * Set-up model coefficients
    */
   this -> p_0 = parameters. GetParameter< double >( "p0" );
   this -> mu = parameters. GetParameter< double >( "mu");
   this -> T = parameters. GetParameter< double >( "T" );
   this -> R = parameters. GetParameter< double >( "R" );
   this -> gravity = parameters. GetParameter< double >( "gravity" );
   this -> maxInflowVelocity = parameters. GetParameter< double >( "max-inflow-velocity" );
   this -> maxOutflowVelocity = parameters. GetParameter< double >( "max-outflow-velocity" );
   this -> startUp = parameters. GetParameter< double >( "start-up" );

   /****
    * Set-up grid functions
    */
   const IndexType variablesNumber = 3;

   rho. setSize( mesh. getDofs() );
   u1. setSize( mesh. getDofs() );
   u2. setSize( mesh. getDofs() );
   p. setSize( mesh. getDofs() );

   dofVector. setSize( variablesNumber * mesh. getDofs() );
   rhsDofVector. setLike( dofVector );

   /****
    * Set-up numerical scheme
    */
   pressureGradient. setFunction( p );
   pressureGradient. bindMesh( this -> mesh );
   this -> eulerScheme. bindMesh( this -> mesh );
   this -> eulerScheme. setPressureGradient( this -> pressureGradient );
   this -> u1Viscosity. bindMesh( this -> mesh );
   this -> u1Viscosity. setFunction( this -> u1 );
   this -> u2Viscosity. bindMesh( this -> mesh );
   this -> u2Viscosity. setFunction( this -> u2 );
   return true;
}

template< typename Mesh, typename EulerScheme >
bool navierStokesSolver< Mesh, EulerScheme > :: setInitialCondition( const tnlParameterContainer& parameters )
{
   tnlSharedVector< RealType, DeviceType, IndexType > rho, rho_u1, rho_u2;
   const IndexType& dofs = mesh. getDofs();
   rho.    bind( & dofVector. getData()[ 0        ], dofs );
   rho_u1. bind( & dofVector. getData()[     dofs ], dofs );
   rho_u2. bind( & dofVector. getData()[ 2 * dofs ], dofs );

   rho. setValue( p_0 / ( this -> R * this -> T ) );
   rho_u1. setValue( 0.0 );
   rho_u2. setValue( 0.0 );

   const IndexType& xSize = mesh. getDimensions(). x();
   const IndexType& ySize = mesh. getDimensions(). y();
   const RealType hx = mesh. getParametricStep(). x();
   const RealType hy = mesh. getParametricStep(). y();

   for( IndexType j = 0; j < ySize; j ++ )
      for( IndexType i = 0; i < xSize; i ++ )
      {
         const IndexType c = mesh. getElementIndex( j, i );
         const RealType x = i * hx;
         const RealType y = j * hy;

         rho. setElement( c, p_0 / ( this -> R * this -> T ) );
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
   rho.    bind( & dofVector. getData()[ 0        ], dofs );
   rho_u1. bind( & dofVector. getData()[     dofs ], dofs );
   rho_u2. bind( & dofVector. getData()[ 2 * dofs ], dofs );
   rho_t.    bind( & this -> rhsDofVector. getData()[ 0        ], dofs );
   rho_u1_t. bind( & this -> rhsDofVector. getData()[     dofs ], dofs );
   rho_u2_t. bind( & this -> rhsDofVector. getData()[ 2 * dofs ], dofs );


   updatePhysicalQuantities( rho, rho_u1, rho_u2 );
   tnlVector< RealType, DeviceType, IndexType > u;
   u. setLike( u1 );
   for( IndexType i = 0; i < this -> u1. getSize(); i ++ )
      u[ i ] = sqrt( u1[ i ] * u1[ i ] + u2[ i ] * u2[ i ] );
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
   FileNameBaseNumberEnding( "rho-t-", step, 5, ".tnl", fileName );
   if( ! rho_t. save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "rho-u1-t-", step, 5, ".tnl", fileName );
   if( ! rho_u1_t. save( fileName ) )
      return false;

   FileNameBaseNumberEnding( "rho-u2-t-", step, 5, ".tnl", fileName );
   if( ! rho_u2_t. save( fileName ) )
      return false;

   FileNameBaseNumberEnding( "rho-", step, 5, ".tnl", fileName );
   if( ! rho. save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "rho-u1-", step, 5, ".tnl", fileName );
   if( ! rho_u1. save( fileName ) )
      return false;

   FileNameBaseNumberEnding( "rho-u2-", step, 5, ".tnl", fileName );
   if( ! rho_u2. save( fileName ) )
      return false;
   return true;
}

template< typename Mesh, typename EulerScheme >
   template< typename Vector >
void navierStokesSolver< Mesh, EulerScheme > :: updatePhysicalQuantities( const Vector& rho,
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
            IndexType c = mesh. getElementIndex( j, i );
            u1[ c ] = rho_u1[ c ] / rho[ c ];
            u2[ c ] = rho_u2[ c ] / rho[ c ];
            p[ c ] = rho[ c ] * this -> R * this -> T;
         }
   }
#ifdef HAVE_CUDA
   if( Mesh :: DeviceType :: getDevice() == tnlCudaDevice )
   {
      const int cudaBlockSize = 256;
      const int maxGridSize = 65536;
      const int cudaBlocks = ceil( u1. getSize() / cudaBlockSize );
      const int gridNumbers = ceil( cudaBlocks / maxGridSize );
      dim3 blockSize, gridSize;
      blockSize. x = cudaBlockSize;
      for( int grid = 0; grid < gridNumbers; grid ++ )
      {

         Index size( 0 );
         if( grid < gridNumbers )
         {
            gridSize. x = maxGridSize;
            size = gridSize. x * blockSize. x;
         }
         else
         {
            gridSize. x = cudaBlocks % maxGridSize;
            size = u1. getSize() - ( gridNumbers - 1 ) * maxGridSize * cudaBlockSize;
         }
         Index startIndex = grid * maxGridSize * cudaBlockSize;
         computeVelocityFieldCuda<<< blockSize, gridSize >>>( size,
                                                              this -> R,
                                                              this -> T,
                                                              & rho. getData()[ startIndex ],
                                                              & rho_u1. getData()[ startIndex ],
                                                              & rho_u2. getData()[ startIndex ],
                                                              & u1. getData()[ startIndex ],
                                                              & u2. getData()[ startIndex ],
                                                              & p. getData()[ startIdex ] );

   }
#endif
}

template< typename Mesh, typename EulerScheme >
void navierStokesSolver< Mesh, EulerScheme > :: GetExplicitRHS(  const RealType& time,
                                                                 const RealType& tau,
                                                                 DofVectorType& u,
                                                                 DofVectorType& fu )
{
   tnlSharedVector< RealType, DeviceType, IndexType > rho, rho_u1, rho_u2,
                                                      rho_t, rho_u1_t, rho_u2_t;

   const IndexType& dofs = mesh. getDofs();
   rho. bind( & u. getData()[ 0 ], dofs );
   rho_u1. bind( & u. getData()[ dofs ], dofs );
   rho_u2. bind( & u. getData()[ 2 * dofs ], dofs );

   eulerScheme. setRho( rho );
   eulerScheme. setRhoU1( rho_u1 );
   eulerScheme. setRhoU2( rho_u2 );

   rho_t. bind( & fu. getData()[ 0 ], dofs );
   rho_u1_t. bind( & fu. getData()[ dofs ], dofs );
   rho_u2_t. bind( & fu. getData()[ 2 * dofs ], dofs );



   updatePhysicalQuantities( rho, rho_u1, rho_u2 );

   /****
    * Compute RHS
    */
   const IndexType& xSize = mesh. getDimensions(). x();
   const IndexType& ySize = mesh. getDimensions(). y();
   const RealType hx = mesh. getParametricStep(). x();
   const RealType hy = mesh. getParametricStep(). y();
   RealType startUpCoefficient( 1.0 );
   if( this -> startUp != 0.0 )
      startUpCoefficient = Min( ( RealType ) 1.0, time / this -> startUp );

   if( DeviceType :: getDevice() == tnlHostDevice )
   {
      /****
       * Set the boundary conditions.
       * Speed: DBC on inlet, NBC on outlet, 0 DBC on walls.
       * Density: NBS on inlet, DBC on outlet, NBC on walls
       */
      for( IndexType i = 0; i < xSize; i ++ )
      {
         const IndexType c1 = mesh. getElementIndex( 0, i );
         const IndexType c2 = mesh. getElementIndex( 1, i );
         const IndexType c3 = mesh. getElementIndex( ySize - 1, i );
         const IndexType c4 = mesh. getElementIndex( ySize - 2, i );

         RealType x = i * hx / mesh. getProportions(). x();

         /****
          * Boundary conditions at the bottom and the top
          */
         if( problem == cavity )
         {
            this -> u1[ c1 ] = 0;
            this -> u2[ c1 ] = 0;
            this -> u1[ c3 ] = sin( M_PI * x ) * startUpCoefficient * this -> maxOutflowVelocity;
            this -> u2[ c3 ] = 0;

            rho[ c1 ] = rho[ c2 ];
            rho[ c3 ] = rho[ c4 ];
            //rho[ c3 ] = this -> p_0 / ( this -> R * this -> T );
         }

         rho_u1[ c1 ] = rho[ c1 ] * this -> u1[ c1 ];
         rho_u2[ c1 ] = rho[ c1 ] * this -> u2[ c1 ];
         rho_u1[ c3 ] = rho[ c3 ] * this -> u1[ c3 ];
         rho_u2[ c3 ] = rho[ c3 ] * this -> u2[ c3 ];
      }
      for( IndexType j = 0; j < ySize; j ++ )
      {
         const IndexType c1 = mesh. getElementIndex( j, 0 );
         const IndexType c2 = mesh. getElementIndex( j, 1 );
         const IndexType c3 = mesh. getElementIndex( j, xSize - 1 );
         const IndexType c4 = mesh. getElementIndex( j, xSize - 2 );

         RealType y = j * hy / mesh. getProportions(). y();

         /****
          * Boundary conditions on the left and right
          */
         if( problem == cavity )
         {
            this -> u1[ c1 ] = 0;
            this -> u2[ c1 ] = 0;
            this -> u1[ c3 ] = 0;
            this -> u2[ c3 ] = 0;

            rho[ c1 ] = rho[ c2 ];
            rho[ c3 ] = rho[ c4 ];
         }
         rho_u1[ c1 ] = rho[ c1 ] * this -> u1[ c1 ];
         rho_u2[ c1 ] = rho[ c1 ] * this -> u2[ c1 ];
         rho_u1[ c3 ] = rho[ c3 ] * this -> u1[ c3 ];
         rho_u2[ c3 ] = rho[ c3 ] * this -> u2[ c3 ];

      }

      const RealType c = sqrt( this -> R * this -> T );
   #ifdef HAVE_OPENMP
   #pragma omp parallel for
   #endif
      for( IndexType j = 0; j < ySize; j ++ )
         for( IndexType i = 0; i < xSize; i ++ )
         {
            IndexType c = mesh. getElementIndex( j, i );
            if( i == 0 || j == 0 ||
                i == xSize - 1 || j == ySize - 1 )
            {
               rho_t[ c ] = rho_u1_t[ c ] = rho_u2_t[ c ] = 0.0;
               continue;
            }

            eulerScheme. getExplicitRhs( c,
                                         rho_t[ c ],
                                         rho_u1_t[ c ],
                                         rho_u2_t[ c ] );
            
            //rho_u1_t[ c ] += ;
            rho_u2_t[ c ] -= startUpCoefficient * this -> gravity * this -> rho[ c ];

            /***
             * Add the viscosity term
             */
            rho_u1_t[ c ] += this -> mu * u1Viscosity. getDiffusion( c );
            rho_u2_t[ c ] += this -> mu * u2Viscosity. getDiffusion( c );

         }
   }
   if( DeviceType :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      const int cudaBlockSize = 256;
      const int maxGridSize = 65536;
      const int cudaBlocks = ceil( u1. getSize() / cudaBlockSize );
      const int gridNumbers = ceil( cudaBlocks / maxGridSize );
      dim3 blockSize, gridSize;
      blockSize. x = cudaBlockSize;
      for( int grid = 0; grid < gridNumbers; grid ++ )
      {
         if( grid < gridNumbers )
            gridSize. x = maxGridSize;
         else
            gridSize. x = cudaBlocks % maxGridSize;

         computeRhsCuda<<< blockSize, gridSize >>>
                                ( gridIdx,
                                  xSize,
                                  ySize,
                                  hx,
                                  hy,
                                  rho. getData(),
                                  rho_u1. getData(),
                                  rho_u2. getData(),
                                  u1. getData(),
                                  u2. getData(),
                                  p. getData(),
                                  rho_t. getData(),
                                  rho_u1_t. getData(),
                                  rho_u2_t. getData() );

      }
#endif
   }

   rhsDofVector = fu;
   //makeSnapshot( 0.0, 1 );
   //getchar();

}

#ifdef HAVE_CUDA
template< typename Real, typename Index >
__device__ void computeVelocityFieldCuda( const Index size,
                                          const Real R,
                                          const Real T,
                                          const Real* rho,
                                          const Real* rho_u1,
                                          const Real* rho_u2,
                                          Real* u1,
                                          Real* u2,
                                          Real* p )
{
   int globalThreadIdx = blockIdx. x * blockDim. x + threadIdx. x;
   if( globalThreadIdx > size )
      return;

   u1[ globalThreadIdx ] = rho_u1[ globalThreadIdx ] / rho[ globalThreadIdx ];
   u2[ globalThreadIdx ] = rho_u2[ globalThreadIdx ] / rho[ globalThreadIdx ];
   p[ globalThreadIdx ] = rho[ globalThreadIdx ] * R * T;
}
#endif

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
