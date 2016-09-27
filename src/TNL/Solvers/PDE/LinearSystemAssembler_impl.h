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
          DofVectorPointer& bPointer ) const
{
      static_assert( std::is_same< MeshFunction,
                                Containers::Vector< typename MeshFunction::RealType,
                                           typename MeshFunction::DeviceType,
                                           typename MeshFunction::IndexType > >::value != true,
      "Error: I am getting Vector instead of MeshFunction or similar object. You might forget to bind DofVector into MeshFunction in you method getExplicitRHS."  );

   const IndexType maxRowLength = matrixPointer.template getData< Devices::Host >().getMaxRowLength();
   Assert( maxRowLength > 0, );

   {
      SharedPointer< TraverserUserData, DeviceType >
         userData( time,
                   tau,
                   &differentialOperatorPointer.template getData< DeviceType >(),
                   &boundaryConditionsPointer.template getData< DeviceType >(),
                   &rightHandSidePointer.template getData< DeviceType >(),
                   &uPointer.template getData< DeviceType >(),
                   &matrixPointer.template getData< DeviceType >(),
                   &bPointer.template getData< DeviceType >() );
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
