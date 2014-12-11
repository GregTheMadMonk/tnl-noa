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

#ifndef TNLMATRIXSETTER_IMPL_H_
#define TNLMATRIXSETTER_IMPL_H_

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RowLengthsVector >
   template< int EntityDimensions >
void
tnlMatrixSetter< Mesh, DifferentialOperator, BoundaryConditions, RowLengthsVector >::
getRowLengths( const Mesh& mesh,
               DifferentialOperator& differentialOperator,
               BoundaryConditions& boundaryConditions,
               RowLengthsVector& rowLengths ) const
{
   TraversalUserData userData( differentialOperator, boundaryConditions, rowLengths );
   tnlTraversal< MeshType, EntityDimensions > meshTraversal;
   meshTraversal.template processEntities< TraversalUserData,
                                           TraversalBoundaryEntitiesProcessor,
                                           TraversalInteriorEntitiesProcessor >
                                          ( mesh,
                                            userData );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RowLengthsVector >
   template< int EntityDimensions >
void
tnlMatrixSetter< tnlGrid< Dimensions, Real, Device, Index >, DifferentialOperator, BoundaryConditions, RowLengthsVector >::
getRowLengths( const MeshType& mesh,
               const DifferentialOperator& differentialOperator,
               const BoundaryConditions& boundaryConditions,
               RowLengthsVector& rowLengths ) const
{
   TraversalUserData userData( differentialOperator, boundaryConditions, rowLengths );
   tnlTraversal< MeshType, EntityDimensions > meshTraversal;
   meshTraversal.template processEntities< TraversalUserData,
                                           TraversalBoundaryEntitiesProcessor,
                                           TraversalInteriorEntitiesProcessor >
                                          ( mesh,
                                            userData );
}


#endif /* TNLMATRIXSETTER_IMPL_H_ */
