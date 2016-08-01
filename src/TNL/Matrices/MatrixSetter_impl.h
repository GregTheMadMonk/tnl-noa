/***************************************************************************
                          MatrixSetter_impl.h  -  description
                             -------------------
    begin                : Oct 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/mesh/tnlTraverser.h>

namespace TNL {
namespace Matrices {   

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename CompressedRowsLengthsVector >
   template< typename EntityType >
void
MatrixSetter< Mesh, DifferentialOperator, BoundaryConditions, CompressedRowsLengthsVector >::
getCompressedRowsLengths( const MeshPointer& meshPointer,
                          DifferentialOperatorPointer& differentialOperatorPointer,
                          BoundaryConditionsPointer& boundaryConditionsPointer,
                          CompressedRowsLengthsVectorPointer& rowLengthsPointer ) const
{
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
}

} // namespace Matrices
} // namespace TNL
