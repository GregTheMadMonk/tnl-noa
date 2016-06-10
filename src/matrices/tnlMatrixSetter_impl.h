/***************************************************************************
                          tnlMatrixSetter_impl.h  -  description
                             -------------------
    begin                : Oct 11, 2014
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

#pragma once

#include <mesh/tnlTraverser.h>

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename CompressedRowsLengthsVector >
   template< typename EntityType >
void
tnlMatrixSetter< Mesh, DifferentialOperator, BoundaryConditions, CompressedRowsLengthsVector >::
getCompressedRowsLengths( const MeshPointer& meshPointer,
                          DifferentialOperatorPointer& differentialOperatorPointer,
                          BoundaryConditionsPointer& boundaryConditionsPointer,
                          CompressedRowsLengthsVectorPointer& rowLengthsPointer ) const
{
   //if( std::is_same< DeviceType, tnlHost >::value )
   {
      TraversalUserData userData( differentialOperatorPointer, boundaryConditionsPointer, rowLengthsPointer );
      tnlTraverser< MeshType, EntityType > meshTraversal;
      meshTraversal.template processBoundaryEntities< TraversalUserData,
                                                      TraversalBoundaryEntitiesProcessor >
                                                    ( meshPointer,
                                                      userData );
      meshTraversal.template processInteriorEntities< TraversalUserData,
                                                      TraversalInteriorEntitiesProcessor >
                                                    ( meshPointer,
                                                      userData );
   }
   /*if( std::is_same< DeviceType, tnlCuda >::value )
   {
      DifferentialOperator* kernelDifferentialOperator = tnlCuda::passToDevice( differentialOperator );
      BoundaryConditions* kernelBoundaryConditions = tnlCuda::passToDevice( boundaryConditions );
      CompressedRowsLengthsVector* kernelCompressedRowsLengths = tnlCuda::passToDevice( rowLengths );
      TraversalUserData userData( *kernelDifferentialOperator, *kernelBoundaryConditions, *kernelCompressedRowsLengths );
      checkCudaDevice;
      tnlTraverser< MeshType, EntityType > meshTraversal;
      meshTraversal.template processBoundaryEntities< TraversalUserData,
                                                      TraversalBoundaryEntitiesProcessor >
                                                    ( meshPointer,
                                                      userData );
      meshTraversal.template processInteriorEntities< TraversalUserData,
                                                      TraversalInteriorEntitiesProcessor >
                                                    ( meshPointer,
                                                      userData );

      checkCudaDevice;
      tnlCuda::freeFromDevice( kernelDifferentialOperator );
      tnlCuda::freeFromDevice( kernelBoundaryConditions );
      tnlCuda::freeFromDevice( kernelCompressedRowsLengths );
      checkCudaDevice;
   }*/
}

/*
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename CompressedRowsLengthsVector >
   template< typename EntityType >
void
tnlMatrixSetter< tnlGrid< Dimensions, Real, Device, Index >, DifferentialOperator, BoundaryConditions, CompressedRowsLengthsVector >::
getCompressedRowsLengths( const MeshPointer& meshPointer,
                          const DifferentialOperator& differentialOperator,
                          const BoundaryConditions& boundaryConditions,
                          CompressedRowsLengthsVector& rowLengths ) const
{
   if( DeviceType::DeviceType == ( int ) tnlHostDevice )
   {
      TraversalUserData userData( differentialOperator, boundaryConditions, rowLengths );
      tnlTraverser< MeshPointer, EntityType > meshTraversal;
      meshTraversal.template processBoundaryEntities< TraversalUserData,
                                                      TraversalBoundaryEntitiesProcessor >
                                                    ( meshPointer,
                                                      userData );
      meshTraversal.template processInteriorEntities< TraversalUserData,
                                                      TraversalInteriorEntitiesProcessor >
                                                    ( meshPointer,
                                                      userData );
   }
   if( DeviceType::DeviceType == ( int ) tnlCudaDevice )
   {
      DifferentialOperator* kernelDifferentialOperator = tnlCuda::passToDevice( differentialOperator );
      BoundaryConditions* kernelBoundaryConditions = tnlCuda::passToDevice( boundaryConditions );
      CompressedRowsLengthsVector* kernelCompressedRowsLengths = tnlCuda::passToDevice( rowLengths );
      TraversalUserData userData( *kernelDifferentialOperator, *kernelBoundaryConditions, *kernelCompressedRowsLengths );
      checkCudaDevice;
      tnlTraverser< MeshPointer, EntityType > meshTraversal;
      meshTraversal.template processBoundaryEntities< TraversalUserData,
                                                      TraversalBoundaryEntitiesProcessor >
                                                    ( meshPointer,
                                                      userData );
      meshTraversal.template processInteriorEntities< TraversalUserData,
                                                      TraversalInteriorEntitiesProcessor >
                                                    ( meshPointer,
                                                      userData );

      checkCudaDevice;
      tnlCuda::freeFromDevice( kernelDifferentialOperator );
      tnlCuda::freeFromDevice( kernelBoundaryConditions );
      tnlCuda::freeFromDevice( kernelCompressedRowsLengths );
      checkCudaDevice;
   }
}
*/

