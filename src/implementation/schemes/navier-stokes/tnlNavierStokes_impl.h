/***************************************************************************
                          tnlNavierStokes_impl.h  -  description
                             -------------------
    begin                : Oct 22, 2013
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

#ifndef TNLNAVIERSTOKES_IMPL_H_
#define TNLNAVIERSTOKES_IMPL_H_

#include <schemes/navier-stokes/tnlNavierStokes.h>

template< typename AdvectionScheme,
          typename DiffusionScheme >
tnlNavierStokes< AdvectionScheme, DiffusionScheme >::tnlNavierStokes()
: advection( 0 ),
  diffusion( 0 )
{
}

template< typename AdvectionScheme,
          typename DiffusionScheme >
tnlString tnlNavierStokes< AdvectionScheme, DiffusionScheme >::getTypeStatic()
{
   return tnlString( "tnlNavierStokes< " ) +
          AdvectionScheme::getTypeStatic() + ", "
          DiffusionScheme::getTypeStatic() + " >";
}

template< typename AdvectionScheme,
          typename DiffusionScheme >
void tnlNavierStokes< AdvectionScheme, DiffusionScheme >::setAdvectionScheme( AdvectionSchemeType& advection )
{
   this->advection = advection;
}

template< typename AdvectionScheme,
          typename DiffusionScheme >
void tnlNavierStokes< AdvectionScheme, DiffusionScheme >::setDiffusionScheme( DiffusionSchemeType& diffusion )
{
   this->diffusion = diffusion;
}

template< typename AdvectionScheme,
          typename DiffusionScheme >
void tnlNavierStokes< AdvectionScheme, DiffusionScheme >::setBoundaryConditions( BoundaryConditionsType& boundaryConditions )
{
   this->boundaryConditions = boundaryConditions;
}

template< typename AdvectionScheme,
          typename DiffusionScheme >
void tnlNavierStokes< AdvectionScheme, DiffusionScheme > :: updatePhysicalQuantities( const Vector& rho,
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
            u1[ c ] = rho_u1[ c ] / rho[ c ];
            u2[ c ] = rho_u2[ c ] / rho[ c ];
            p[ c ] = rho[ c ] * this -> R * this -> T;
         }
   }
}

void tnlNavierStokes< AdvectionScheme, DiffusionScheme >::getExplicitRhs( const RealType& time,
                                                                          const RealType& tau,
                                                                          DofVectorType& u,
                                                                          DofVectorType& fu )
{
   tnlAssert( this->advection );
   tnlAssert( this->diffusion );
   tnlAssert( this->boundaryConditions );

   tnlSharedVector< RealType, DeviceType, IndexType > rho, rho_u1, rho_u2,
                                                      rho_t, rho_u1_t, rho_u2_t;

   const IndexType& dofs = this->advection.getMesh().getDofs();
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

#ifdef HAVE_OPENMP
  #pragma omp parallel for
  #endif
  for( IndexType j = 0; j < ySize; j ++ )
     for( IndexType i = 0; i < xSize; i ++ )
     {
        IndexType c = this->advection.getMesh().getElementIndex( i, j );
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
        rho_u1_t[ c ] += this -> mu * diffusion->getDiffusion( c );
        rho_u2_t[ c ] += this -> mu * diffusion->getDiffusion( c );

     }
}


#endif /* TNLNAVIERSTOKES_IMPL_H_ */
