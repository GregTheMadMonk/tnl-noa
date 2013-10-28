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
          typename DiffusionScheme,
          typename BoundaryConditions >
tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::tnlNavierStokes()
: advection( 0 ),
  diffusion( 0 )
{
   this->rho.setName( "navier-stokes-rho" );
   this->u1.setName( "navier-stokes-u1");
   this->u2.setName( "navier-stokes-u2" );
   this->p.setName( "navier-stokes-p" );
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
tnlString tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getTypeStatic()
{
   return tnlString( "tnlNavierStokes< " ) +
          AdvectionScheme::getTypeStatic() + ", " +
          DiffusionScheme::getTypeStatic() + " >";
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
void tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::setAdvectionScheme( AdvectionSchemeType& advection )
{
   this->advection = &advection;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
void tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::setDiffusionScheme( DiffusionSchemeType& diffusion )
{
   this->diffusion = &diffusion;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
void tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::setBoundaryConditions( BoundaryConditionsType& boundaryConditions )
{
   this->boundaryConditions = &boundaryConditions;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
void tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::setMesh( MeshType& mesh )
{
   this->mesh = &mesh;
   this->rho.setSize( this->mesh->getDofs() );
   this->u1.setSize(  this->mesh->getDofs() );
   this->u2.setSize(  this->mesh->getDofs() );
   this->p.setSize(   this->mesh->getDofs() );
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
typename tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::VectorType&
   tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getRho()
{
   return this->rho;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
const typename tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::VectorType&
   tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getRho() const
{
   return this->rho;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
typename tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::VectorType&
   tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getU1()
{
   return this->u1;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
const typename tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::VectorType&
   tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getU1() const
{
   return this->u1;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
typename tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::VectorType&
   tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getU2()
{
   return this->u2;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
const typename tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::VectorType&
   tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getU2() const
{
   return this->u2;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
typename tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::VectorType&
   tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getPressure()
{
   return this->p;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
const typename tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::VectorType&
   tnlNavierStokes< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getPressure() const
{
   return this->p;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
   template< typename Vector >
void tnlNavierStokes< AdvectionScheme,
                      DiffusionScheme,
                      BoundaryConditions > :: updatePhysicalQuantities( const Vector& dofs_rho,
                                                                        const Vector& dofs_rho_u1,
                                                                        const Vector& dofs_rho_u2 )
{
   if( DeviceType :: getDevice() == tnlHostDevice )
   {
      //const IndexType& xSize = mesh. getDimensions(). x();
      //const IndexType& ySize = mesh. getDimensions(). y();
      const IndexType size = dofs_rho.getSize();

   #ifdef HAVE_OPENMP
   #pragma omp parallel for
   #endif
      //for( IndexType j = 0; j < ySize; j ++ )
      //   for( IndexType i = 0; i < xSize; i ++ )
      for( IndexType c = 0; c < size; c++ )
         {
            //IndexType c = mesh. getElementIndex( i, j );
            this->rho[ c ] = dofs_rho[ c ];
            this->u1[ c ] = dofs_rho_u1[ c ] / dofs_rho[ c ];
            this->u2[ c ] = dofs_rho_u2[ c ] / dofs_rho[ c ];
            this->p[ c ] = dofs_rho[ c ] * this -> R * this -> T;
         }
   }
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
void tnlNavierStokes< AdvectionScheme,
                      DiffusionScheme,
                      BoundaryConditions >::getExplicitRhs( const RealType& time,
                                                            const RealType& tau,
                                                            DofVectorType& u,
                                                            DofVectorType& fu ) const
{
   tnlAssert( this->advection, );
   tnlAssert( this->diffusion, );
   tnlAssert( this->boundaryConditions, );

   tnlSharedVector< RealType, DeviceType, IndexType > rho, rho_u1, rho_u2,
                                                      rho_t, rho_u1_t, rho_u2_t;

   const IndexType& dofs = this->advection.getMesh().getDofs();
   rho. bind( & u. getData()[ 0 ], dofs );
   rho_u1. bind( & u. getData()[ dofs ], dofs );
   rho_u2. bind( & u. getData()[ 2 * dofs ], dofs );

   advection.setRho( rho );
   advection.setRhoU1( rho_u1 );
   advection.setRhoU2( rho_u2 );

   rho_t.bind( & fu. getData()[ 0 ], dofs );
   rho_u1_t.bind( & fu. getData()[ dofs ], dofs );
   rho_u2_t.bind( & fu. getData()[ 2 * dofs ], dofs );

   updatePhysicalQuantities( rho, rho_u1, rho_u2 );

   const IndexType& xSize = this->mesh->getDimensions().x();
   const IndexType& ySize = this->mesh->getDimensions().y();

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

        this->advection->getExplicitRhs( c,
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
