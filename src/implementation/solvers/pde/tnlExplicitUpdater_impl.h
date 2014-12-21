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
        const Mesh& mesh,
        DifferentialOperator& differentialOperator,
        BoundaryConditions& boundaryConditions,
        RightHandSide& rightHandSide,
        DofVector& u,
        DofVector& fu ) const
{
   if( DeviceType::DeviceType == tnlHostDevice )
   {
      TraversalUserData userData( time, differentialOperator, boundaryConditions, rightHandSide, u, fu );
      tnlTraversal< MeshType, EntityDimensions > meshTraversal;
      meshTraversal.template processBoundaryEntities< TraversalUserData,
                                                      TraversalBoundaryEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
      meshTraversal.template processInteriorEntities< TraversalUserData,
                                                      TraversalInteriorEntitiesProcessor >
                                                    ( mesh,
                                                      userData );

   }
   if( DeviceType::DeviceType == tnlCudaDevice )
   {
      RealType* kernelTime = tnlCuda::passToDevice( time );
      DifferentialOperator* kernelDifferentialOperator = tnlCuda::passToDevice( differentialOperator );
      BoundaryConditions* kernelBoundaryConditions = tnlCuda::passToDevice( boundaryConditions );
      RightHandSide* kernelRightHandSide = tnlCuda::passToDevice( rightHandSide );
      DofVector* kernelU = tnlCuda::passToDevice( u );
      DofVector* kernelFu = tnlCuda::passToDevice( fu );
      TraversalUserData userData( *kernelTime, *kernelDifferentialOperator, *kernelBoundaryConditions, *kernelRightHandSide, *kernelU, *kernelFu );
      checkCudaDevice;
      tnlTraversal< MeshType, EntityDimensions > meshTraversal;
      meshTraversal.template processBoundaryEntities< TraversalUserData,
                                                      TraversalBoundaryEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
      meshTraversal.template processInteriorEntities< TraversalUserData,
                                                      TraversalInteriorEntitiesProcessor >
                                                    ( mesh,
                                                      userData );

      checkCudaDevice;
      tnlCuda::freeFromDevice( kernelTime );
      tnlCuda::freeFromDevice( kernelDifferentialOperator );
      tnlCuda::freeFromDevice( kernelBoundaryConditions );
      tnlCuda::freeFromDevice( kernelRightHandSide );
      tnlCuda::freeFromDevice( kernelU );
      tnlCuda::freeFromDevice( kernelFu );
      checkCudaDevice;
   }
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
        const tnlGrid< Dimensions, Real, Device, Index >& mesh,
        const DifferentialOperator& differentialOperator,
        const BoundaryConditions& boundaryConditions,
        const RightHandSide& rightHandSide,
        DofVector& u,
        DofVector& fu ) const
{

   if( DeviceType::DeviceType == tnlHostDevice )
   {
      TraversalUserData userData( time, differentialOperator, boundaryConditions, rightHandSide, u, fu );
      tnlTraversal< MeshType, EntityDimensions > meshTraversal;
      meshTraversal.template processBoundaryEntities< TraversalUserData,
                                                      TraversalBoundaryEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
      meshTraversal.template processInteriorEntities< TraversalUserData,
                                                      TraversalInteriorEntitiesProcessor >
                                                    ( mesh,
                                                      userData );

   }
   if( DeviceType::DeviceType == tnlCudaDevice )
   {

      RealType* kernelTime = tnlCuda::passToDevice( time );
      DifferentialOperator* kernelDifferentialOperator = tnlCuda::passToDevice( differentialOperator );
      BoundaryConditions* kernelBoundaryConditions = tnlCuda::passToDevice( boundaryConditions );
      RightHandSide* kernelRightHandSide = tnlCuda::passToDevice( rightHandSide );
      DofVector* kernelU = tnlCuda::passToDevice( u );
      DofVector* kernelFu = tnlCuda::passToDevice( fu );
      checkCudaDevice;
      TraversalUserData userData( *kernelTime, *kernelDifferentialOperator, *kernelBoundaryConditions, *kernelRightHandSide, *kernelU, *kernelFu );
      tnlTraversal< MeshType, EntityDimensions > meshTraversal;
      meshTraversal.template processBoundaryEntities< TraversalUserData,
                                                      TraversalBoundaryEntitiesProcessor >
                                                   ( mesh,
                                                     userData );
      meshTraversal.template processInteriorEntities< TraversalUserData,
                                                      TraversalInteriorEntitiesProcessor >
                                                   ( mesh,
                                                     userData );

      checkCudaDevice;
      tnlCuda::freeFromDevice( kernelTime );
      tnlCuda::freeFromDevice( kernelDifferentialOperator );
      tnlCuda::freeFromDevice( kernelBoundaryConditions );
      tnlCuda::freeFromDevice( kernelRightHandSide );
      tnlCuda::freeFromDevice( kernelU );
      tnlCuda::freeFromDevice( kernelFu );
      checkCudaDevice;
   }
}


#endif /* TNLEXPLICITUPDATER_IMPL_H_ */