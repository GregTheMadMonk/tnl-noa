/***************************************************************************
                          ExplicitUpdater_impl.h  -  description
                             -------------------
    begin                : Jul 29, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>
#include <TNL/Meshes/GridDetails/Traverser_Grid1D.h>
#include <TNL/Meshes/GridDetails/Traverser_Grid2D.h>
#include <TNL/Meshes/GridDetails/Traverser_Grid3D.h>

#include "ExplicitUpdater.h"


namespace TNL {
namespace Solvers {
namespace PDE {   

template< typename Mesh,
          typename MeshFunction,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide >
   template< typename EntityType >
void
ExplicitUpdater< Mesh, MeshFunction, DifferentialOperator, BoundaryConditions, RightHandSide >::
update( const RealType& time,
        const MeshPointer& meshPointer,
        const DifferentialOperatorPointer& differentialOperatorPointer,        
        const BoundaryConditionsPointer& boundaryConditionsPointer,
        const RightHandSidePointer& rightHandSidePointer,
        MeshFunctionPointer& uPointer,
        MeshFunctionPointer& fuPointer ) const
{
   static_assert( std::is_same< MeshFunction,
                                Vectors::Vector< typename MeshFunction::RealType,
                                           typename MeshFunction::DeviceType,
                                           typename MeshFunction::IndexType > >::value != true,
      "Error: I am getting tnlVector instead of MeshFunction or similar object. You might forget to bind DofVector into MeshFunction in you method getExplicitRHS."  );
   {
      TraverserUserData userData( time,
                                  differentialOperatorPointer.template getData< DeviceType >(),
                                  boundaryConditionsPointer.template getData< DeviceType >(),
                                  rightHandSidePointer.template getData< DeviceType >(),
                                  uPointer.template modifyData< DeviceType >(),
                                  fuPointer.template modifyData< DeviceType >(),
                                  fuPointer.template modifyData< Devices::Host >() );
      Meshes::Traverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserBoundaryEntitiesProcessor >
                                                    ( meshPointer,
                                                      userData );
      meshTraverser.template processInteriorEntities< TraverserUserData,
                                                      TraverserInteriorEntitiesProcessor >
                                                    ( meshPointer,
                                                      userData );

   }
}

} // namespace PDE
} // namespace Solvers
} // namespace TNL
