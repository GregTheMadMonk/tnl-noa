/***************************************************************************
                          tnlMatrixSetter_impl.h  -  description
                             -------------------
    begin                : Oct 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/mesh/tnlTraverser.h>

namespace TNL {

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename CompressedRowsLengthsVector >
   template< typename EntityType >
void
tnlMatrixSetter< Mesh, DifferentialOperator, BoundaryConditions, CompressedRowsLengthsVector >::
getCompressedRowsLengths( const Mesh& mesh,
               DifferentialOperator& differentialOperator,
               BoundaryConditions& boundaryConditions,
               CompressedRowsLengthsVector& rowLengths ) const
{
   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      TraversalUserData userData( differentialOperator, boundaryConditions, rowLengths );
      tnlTraverser< MeshType, EntityType > meshTraversal;
      meshTraversal.template processBoundaryEntities< TraversalUserData,
                                                      TraversalBoundaryEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
      meshTraversal.template processInteriorEntities< TraversalUserData,
                                                      TraversalInteriorEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
   }
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
      DifferentialOperator* kernelDifferentialOperator = Devices::Cuda::passToDevice( differentialOperator );
      BoundaryConditions* kernelBoundaryConditions = Devices::Cuda::passToDevice( boundaryConditions );
      CompressedRowsLengthsVector* kernelCompressedRowsLengths = Devices::Cuda::passToDevice( rowLengths );
      TraversalUserData userData( *kernelDifferentialOperator, *kernelBoundaryConditions, *kernelCompressedRowsLengths );
      checkCudaDevice;
      tnlTraverser< MeshType, EntityType > meshTraversal;
      meshTraversal.template processBoundaryEntities< TraversalUserData,
                                                      TraversalBoundaryEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
      meshTraversal.template processInteriorEntities< TraversalUserData,
                                                      TraversalInteriorEntitiesProcessor >
                                                    ( mesh,
                                                      userData );

      checkCudaDevice;
      Devices::Cuda::freeFromDevice( kernelDifferentialOperator );
      Devices::Cuda::freeFromDevice( kernelBoundaryConditions );
      Devices::Cuda::freeFromDevice( kernelCompressedRowsLengths );
      checkCudaDevice;
   }
}

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
getCompressedRowsLengths( const MeshType& mesh,
               const DifferentialOperator& differentialOperator,
               const BoundaryConditions& boundaryConditions,
               CompressedRowsLengthsVector& rowLengths ) const
{
   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      TraversalUserData userData( differentialOperator, boundaryConditions, rowLengths );
      tnlTraverser< MeshType, EntityType > meshTraversal;
      meshTraversal.template processBoundaryEntities< TraversalUserData,
                                                      TraversalBoundaryEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
      meshTraversal.template processInteriorEntities< TraversalUserData,
                                                      TraversalInteriorEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
   }
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
      DifferentialOperator* kernelDifferentialOperator = Devices::Cuda::passToDevice( differentialOperator );
      BoundaryConditions* kernelBoundaryConditions = Devices::Cuda::passToDevice( boundaryConditions );
      CompressedRowsLengthsVector* kernelCompressedRowsLengths = Devices::Cuda::passToDevice( rowLengths );
      TraversalUserData userData( *kernelDifferentialOperator, *kernelBoundaryConditions, *kernelCompressedRowsLengths );
      checkCudaDevice;
      tnlTraverser< MeshType, EntityType > meshTraversal;
      meshTraversal.template processBoundaryEntities< TraversalUserData,
                                                      TraversalBoundaryEntitiesProcessor >
                                                    ( mesh,
                                                      userData );
      meshTraversal.template processInteriorEntities< TraversalUserData,
                                                      TraversalInteriorEntitiesProcessor >
                                                    ( mesh,
                                                      userData );

      checkCudaDevice;
      Devices::Cuda::freeFromDevice( kernelDifferentialOperator );
      Devices::Cuda::freeFromDevice( kernelBoundaryConditions );
      Devices::Cuda::freeFromDevice( kernelCompressedRowsLengths );
      checkCudaDevice;
   }
}

} // namespace TNL
