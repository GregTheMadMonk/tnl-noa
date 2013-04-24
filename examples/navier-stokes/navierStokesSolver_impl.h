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

template< typename Mesh >
navierStokesSolver< Mesh > :: navierStokesSolver()
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

template< typename Mesh >
bool navierStokesSolver< Mesh > :: init( const tnlParameterContainer& parameters )
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
   this -> mesh. setLowerCorner( tnlTuple< 2, RealType >( 0, 0 ) );
   this -> mesh. setUpperCorner( proportions );

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
   RealType hx = this -> mesh. getSpaceStep(). x();
   RealType hy = this -> mesh. getSpaceStep(). y();
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
   const tnlString& schemeName = parameters. GetParameter< tnlString >( "scheme" );
   if( schemeName == "lax-fridrichs" )
      this -> scheme = laxFridrichsEnm;

   return true;
}

template< typename Mesh >
bool navierStokesSolver< Mesh > :: setInitialCondition( const tnlParameterContainer& parameters )
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
   const RealType hx = mesh. getSpaceStep(). x();
   const RealType hy = mesh. getSpaceStep(). y();

   for( IndexType j = 0; j < ySize; j ++ )
      for( IndexType i = 0; i < xSize; i ++ )
      {
         const IndexType c = mesh. getNodeIndex( j, i );
         const RealType x = i * hx;
         const RealType y = j * hy;

         rho. setElement( c, p_0 / ( this -> R * this -> T ) );
         rho_u1. setElement( c, 0.0 );
         rho_u2. setElement( c, 0.0 );

      }
   return true;
}

template< typename Mesh >
typename navierStokesSolver< Mesh > :: DofVectorType& navierStokesSolver< Mesh > :: getDofVector()
{
   return this -> dofVector;
}

template< typename Mesh >
bool navierStokesSolver< Mesh > :: makeSnapshot( const RealType& t,
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
   return true;
}

template< typename Mesh >
   template< typename Vector >
void navierStokesSolver< Mesh > :: updatePhysicalQuantities( const Vector& rho,
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
            IndexType c = mesh. getNodeIndex( j, i );
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

template< typename Mesh >
void navierStokesSolver< Mesh > :: GetExplicitRHS(  const RealType& time,
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

   rho_t. bind( & fu. getData()[ 0 ], dofs );
   rho_u1_t. bind( & fu. getData()[ dofs ], dofs );
   rho_u2_t. bind( & fu. getData()[ 2 * dofs ], dofs );

   updatePhysicalQuantities( rho, rho_u1, rho_u2 );

   /****
    * Compute RHS
    */
   const IndexType& xSize = mesh. getDimensions(). x();
   const IndexType& ySize = mesh. getDimensions(). y();
   const RealType hx = mesh. getSpaceStep(). x();
   const RealType hy = mesh. getSpaceStep(). y();
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
         const IndexType c1 = mesh. getNodeIndex( 0, i );
         const IndexType c2 = mesh. getNodeIndex( 1, i );
         const IndexType c3 = mesh. getNodeIndex( ySize - 1, i );
         const IndexType c4 = mesh. getNodeIndex( ySize - 2, i );

         RealType x = i * hx / mesh. getUpperCorner(). x();

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
         const IndexType c1 = mesh. getNodeIndex( j, 0 );
         const IndexType c2 = mesh. getNodeIndex( j, 1 );
         const IndexType c3 = mesh. getNodeIndex( j, xSize - 1 );
         const IndexType c4 = mesh. getNodeIndex( j, xSize - 2 );

         RealType y = j * hy / mesh. getUpperCorner(). y();

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
            IndexType c = mesh. getNodeIndex( j, i );
            if( i == 0 || j == 0 ||
                i == xSize - 1 || j == ySize - 1 )
            {
               rho_t[ c ] = rho_u1_t[ c ] = rho_u2_t[ c ] = 0.0;
               continue;
            }

            IndexType e = mesh. getNodeIndex( j, i + 1 );
            IndexType w = mesh. getNodeIndex( j, i - 1 );
            IndexType n = mesh. getNodeIndex( j + 1, i );
            IndexType s = mesh. getNodeIndex( j - 1, i );

            const RealType& u = this -> u1[ c ];
            const RealType& v = this -> u2[ c ];
            const RealType u_sqr = u * u;
            const RealType v_sqr = v * v;
            switch( this -> scheme )
            {
               case laxFridrichsEnm:
                  this -> laxFridrichsScheme. getExplicitRhs( mesh,
                                                              c,
                                                              rho,
                                                              rho_u1,
                                                              rho_u2,
                                                              rho_t,
                                                              rho_u1_t,
                                                              rho_u2_t,
                                                              1.0 );
                  /****
                   * Remark: if fabs( u1_diff ) and fabs( u2_diff ) are used
                   * instead of u_diff in computeInterphaseFriction, higher beta
                   * can be used (up to 100 instead of 10).
                   */
                  rho_u1_t[ c ] += -( p[ e ] - p[ w ] ) / ( 2.0 * hx );
                  rho_u2_t[ c ] += -( p[ n ] - p[ s ] ) / ( 2.0 * hy );
                                          - startUpCoefficient * this -> gravity * this -> rho[ c ];
                  break;
            }

            /***
             * Add the viscosity term
             */
            rho_u1_t[ c ] += this -> mu *
                                   ( ( u1[ e ] - 2.0 * u1[ c ] + u1[ w ] ) / ( hx * hx ) +
                                     ( u1[ n ] - 2.0 * u1[ c ] + u1[ s ] ) / ( hy * hy ) );
            rho_u2_t[ c ] += this -> mu *
                                   ( ( u2[ e ] - 2.0 * u2[ c ] + u2[ w ] ) / ( hx * hx ) +
                                     ( u2[ n ] - 2.0 * u2[ c ] + u2[ s ] ) / ( hy * hy ) );

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

template< typename Mesh >
tnlSolverMonitor< typename navierStokesSolver< Mesh > :: RealType,
                  typename navierStokesSolver< Mesh > :: IndexType >* 
   navierStokesSolver< Mesh > ::  getSolverMonitor()
{
   return &solverMonitor;
}

template< typename Mesh >
tnlString navierStokesSolver< Mesh > :: getTypeStatic()
{
   return tnlString( "navierStokesSolver< " ) +
          Mesh :: getTypeStatic() + " >";
}

template< typename Mesh >
tnlString navierStokesSolver< Mesh > :: getPrologHeader() const
{
   return tnlString( "Navier-Stokes Problem Solver" );
}

template< typename Mesh >
void navierStokesSolver< Mesh > :: writeProlog( tnlLogger& logger,
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
   logger. WriteParameter< double >( "Domain width:", mesh. getUpperCorner(). x() - mesh. getLowerCorner(). x() );
   logger. WriteParameter< double >( "Domain height:", mesh. getUpperCorner(). y() - mesh. getLowerCorner(). y() );
   logger. WriteSeparator();
   logger. WriteParameter< tnlString >( "Space discretisation:", "scheme", parameters );
   logger. WriteParameter< int >( "Meshes along x:", mesh. getDimensions(). x() );
   logger. WriteParameter< int >( "Meshes along y:", mesh. getDimensions(). y() );
   logger. WriteParameter< double >( "Space step along x:", mesh. getSpaceStep(). x() );
   logger. WriteParameter< double >( "Space step along y:", mesh. getSpaceStep(). y() );
}

#endif /* NAVIERSTOKESSOLVER_IMPL_H_ */
