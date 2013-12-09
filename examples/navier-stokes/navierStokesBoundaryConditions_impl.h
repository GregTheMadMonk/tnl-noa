/***************************************************************************
                          navierStokesBoundaryConditions_impl.h  -  description
                             -------------------
    begin                : Oct 24, 2013
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

#ifndef NAVIERSTOKESBOUNDARYCONDITIONS_IMPL_H_
#define NAVIERSTOKESBOUNDARYCONDITIONS_IMPL_H_

#include "navierStokesBoundaryConditions.h"

template< typename Mesh >
navierStokesBoundaryConditions< Mesh >::navierStokesBoundaryConditions()
: mesh( 0 ),
  maxInflowVelocity( 0.0 ),
  startUp( 0.0 )
{
}

template< typename Mesh >
bool navierStokesBoundaryConditions< Mesh >::init( const tnlParameterContainer& parameters )
{
   this -> maxInflowVelocity = parameters. GetParameter< double >( "max-inflow-velocity" );
   //this -> maxOutflowVelocity = parameters. GetParameter< double >( "max-outflow-velocity" );
   this -> startUp = parameters. GetParameter< double >( "start-up" );
   //this -> T = parameters. GetParameter< double >( "T" );
   this->p0 = parameters. GetParameter< double >( "p0" );
   return true;
}

template< typename Mesh >
void navierStokesBoundaryConditions< Mesh >::setMesh( const MeshType& mesh )
{
   this->mesh = &mesh;
}

template< typename Mesh >
   template< typename Vector >
void navierStokesBoundaryConditions< Mesh >::apply( const RealType& time,
                                                    Vector& rho,
                                                    Vector& u1,
                                                    Vector& u2,
                                                    Vector& energy )
{
   /****
     * Set the boundary conditions.
     * Speed: DBC on inlet, NBC on outlet, 0 DBC on walls.
     * Density: NBS on inlet, DBC on outlet, NBC on walls
     */

   const IndexType& xSize = this->mesh->getDimensions().x();
   const IndexType& ySize = this->mesh->getDimensions().y();
   const RealType hx = this->mesh->getParametricStep().x();
   const RealType hy = this->mesh->getParametricStep().y();
   RealType startUpCoefficient( 1.0 );
   if( this -> startUp != 0.0 )
      startUpCoefficient = Min( ( RealType ) 1.0, time / this -> startUp );

   for( IndexType i = 0; i < xSize; i ++ )
   {
      const IndexType c1 = this->mesh->getElementIndex( i, 0 );
      const IndexType c2 = this->mesh->getElementIndex( i, 1 );
      const IndexType c3 = this->mesh->getElementIndex( i, ySize - 1 );
      const IndexType c4 = this->mesh->getElementIndex( i, ySize - 2 );

      RealType x = i * hx / this->mesh->getProportions().x();

      /****
       * Boundary conditions at the bottom and the top
       */
      //if( problem == cavity )
      {
         u1[ c1 ] = 0;
         u2[ c1 ] = 0;
         u1[ c3 ] = sin( M_PI * x ) * startUpCoefficient * this -> maxInflowVelocity;
         u2[ c3 ] = 0;

         rho[ c1 ] = rho[ c2 ];
         rho[ c3 ] = rho[ c4 ];
         energy[ c1 ] = energy[ c2 ];
         energy[ c3 ] = energy[ c4 ];
          //rho[ c3 ] = this -> p_0 / ( this -> R * this -> T );
      }

      /*rho_u1[ c1 ] = rho[ c1 ] * this -> u1[ c1 ];
      rho_u2[ c1 ] = rho[ c1 ] * this -> u2[ c1 ];
      rho_u1[ c3 ] = rho[ c3 ] * this -> u1[ c3 ];
      rho_u2[ c3 ] = rho[ c3 ] * this -> u2[ c3 ];*/
   }
   for( IndexType j = 0; j < ySize; j ++ )
   {
      const IndexType c1 = this->mesh->getElementIndex( 0, j );
      const IndexType c2 = this->mesh->getElementIndex( 1, j );
      const IndexType c3 = this->mesh->getElementIndex( xSize - 1, j );
      const IndexType c4 = this->mesh->getElementIndex( xSize - 2, j );

      RealType y = j * hy / this->mesh->getProportions().y();

      /****
       * Boundary conditions on the left and right
       */
      //if( problem == cavity )
      {
         u1[ c1 ] = 0;
         u2[ c1 ] = 0;
         u1[ c3 ] = 0;
         u2[ c3 ] = 0;

         rho[ c1 ] = rho[ c2 ];
         rho[ c3 ] = rho[ c4 ];
         energy[ c1 ] = energy[ c2 ];
         energy[ c3 ] = energy[ c4 ];
      }
      /*rho_u1[ c1 ] = rho[ c1 ] * this -> u1[ c1 ];
      rho_u2[ c1 ] = rho[ c1 ] * this -> u2[ c1 ];
      rho_u1[ c3 ] = rho[ c3 ] * this -> u1[ c3 ];
      rho_u2[ c3 ] = rho[ c3 ] * this -> u2[ c3 ];*/
    }
}


#endif /* NAVIERSTOKESBOUNDARYCONDITIONS_IMPL_H_ */