/***************************************************************************
                          tnlExplicitUpdater_impl.h  -  description
                             -------------------
    begin                : Jul 29, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLEXPLICITUPDATER_IMPL_H_
#define TNLEXPLICITUPDATER_IMPL_H_

#include <mesh/tnlTraversal_Grid1D.h>
#include <mesh/tnlTraversal_Grid2D.h>
#include <mesh/tnlTraversal_Grid3D.h>

template< typename Mesh,
          typename DofVector,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide >
   template< int EntityDimensions >
void
tnlExplicitUpdater< Mesh, DofVector, DifferentialOperator, BoundaryConditions, RightHandSide >::
update( const RealType& time,
        const RealType& tau,
        const Mesh& mesh,
        DifferentialOperator& differentialOperator,
        BoundaryConditions& boundaryConditions,
        RightHandSide& rightHandSide,
        DofVector& u,
        DofVector& fu ) const
{
   TraversalUserData userData( time, tau, differentialOperator, boundaryConditions, rightHandSide, u, fu );
   TraversalBoundaryEntitiesProcessor boundaryEntitiesProcessor;
   TraversalInteriorEntitiesProcessor interiorEntitiesProcessor;
   tnlTraversal< MeshType, EntityDimensions > meshTraversal;
   meshTraversal.template processEntities< TraversalUserData,
                                           TraversalBoundaryEntitiesProcessor,
                                           TraversalInteriorEntitiesProcessor >
                                          ( mesh,
                                            userData,
                                            boundaryEntitiesProcessor,
                                            interiorEntitiesProcessor );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename DofVector,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide >
   template< int EntityDimensions >
void
tnlExplicitUpdater< tnlGrid< Dimensions, Real, Device, Index >, DofVector, DifferentialOperator, BoundaryConditions, RightHandSide >::
update( const RealType& time,
        const RealType& tau,
        const tnlGrid< Dimensions, Real, Device, Index >& mesh,
        DifferentialOperator& differentialOperator,
        BoundaryConditions& boundaryConditions,
        RightHandSide& rightHandSide,
        DofVector& u,
        DofVector& fu ) const
{
   TraversalUserData userData( time, tau, differentialOperator, boundaryConditions, rightHandSide, u, fu );
   TraversalBoundaryEntitiesProcessor boundaryEntitiesProcessor;
   TraversalInteriorEntitiesProcessor interiorEntitiesProcessor;
   tnlTraversal< MeshType, EntityDimensions > meshTraversal;
   meshTraversal.template processEntities< TraversalUserData,
                                           TraversalBoundaryEntitiesProcessor,
                                           TraversalInteriorEntitiesProcessor >
                                          ( mesh,
                                            userData,
                                            boundaryEntitiesProcessor,
                                            interiorEntitiesProcessor );
}


#endif /* TNLEXPLICITUPDATER_IMPL_H_ */
