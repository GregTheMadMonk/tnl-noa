/***************************************************************************
                          LinearSystemAssembler_impl.h  -  description
                             -------------------
    begin                : Oct 12, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>
#include <TNL/Meshes/GridDetails/Traverser_Grid1D.h>
#include <TNL/Meshes/GridDetails/Traverser_Grid2D.h>
#include <TNL/Meshes/GridDetails/Traverser_Grid3D.h>

namespace TNL {
namespace Solvers {
namespace PDE {   

template< typename Mesh,
          typename MeshFunction,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename TimeDiscretisation,
          typename Matrix,
          typename DofVector >
   template< typename EntityType >
void
LinearSystemAssembler< Mesh, MeshFunction, DifferentialOperator, BoundaryConditions, RightHandSide, TimeDiscretisation, Matrix, DofVector >::
assembly( const RealType& time,
          const RealType& tau,
          const MeshPointer& meshPointer,
          const DifferentialOperatorPointer& differentialOperatorPointer,
          const BoundaryConditionsPointer& boundaryConditionsPointer,
          const RightHandSidePointer& rightHandSidePointer,
          const MeshFunctionPointer& uPointer,
          MatrixPointer& matrixPointer,
          DofVectorPointer& bPointer )
{
      static_assert( std::is_same< MeshFunction,
                                Containers::Vector< typename MeshFunction::RealType,
                                           typename MeshFunction::DeviceType,
                                           typename MeshFunction::IndexType > >::value != true,
      "Error: I am getting Vector instead of MeshFunction or similar object. You might forget to bind DofVector into MeshFunction in you method getExplicitUpdate."  );

   const IndexType maxRowLength = matrixPointer.template getData< Devices::Host >().getMaxRowLength();
   TNL_ASSERT( maxRowLength > 0, );

   {
      this->userDataPointer->setUserData(
            time,
            tau,
            &differentialOperatorPointer.template getData< DeviceType >(),
            &boundaryConditionsPointer.template getData< DeviceType >(),
            &rightHandSidePointer.template getData< DeviceType >(),
            &uPointer.template getData< DeviceType >(),
            &matrixPointer.template modifyData< DeviceType >(),
            &bPointer.template modifyData< DeviceType >() );
      Meshes::Traverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                      TraverserBoundaryEntitiesProcessor >
                                                    ( meshPointer,
                                                      userDataPointer );
      meshTraverser.template processInteriorEntities< TraverserUserData,
                                                      TraverserInteriorEntitiesProcessor >
                                                    ( meshPointer,
                                                      userDataPointer );
   }
}

} // namespace PDE
} // namespace Solvers
} // namespace TNL
