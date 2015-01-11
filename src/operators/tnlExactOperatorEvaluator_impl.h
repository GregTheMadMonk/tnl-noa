/***************************************************************************
                          tnlExactOperatorEvaluator_impl.h  -  description
                             -------------------
    begin                : Nov 8, 2014
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

#ifndef TNLEXACTOPERATOREVALUATOR_IMPL_H_
#define TNLEXACTOPERATOREVALUATOR_IMPL_H_

template< typename Mesh,
          typename DofVector,
          typename DifferentialOperator,
          typename Function,
          typename BoundaryConditions >
   template< int EntityDimensions >
void
tnlExactOperatorEvaluator< Mesh, DofVector, DifferentialOperator, Function, BoundaryConditions >::
evaluate( const RealType& time,
          const Mesh& mesh,
          const DifferentialOperator& differentialOperator,
          const Function& function,
          const BoundaryConditions& boundaryConditions,
          DofVector& fu ) const
{
   TraversalUserData userData( time, differentialOperator, function, boundaryConditions, fu );
   TraversalBoundaryEntitiesProcessor boundaryEntitiesProcessor;
   TraversalInteriorEntitiesProcessor interiorEntitiesProcessor;
   tnlTraverser< MeshType, EntityDimensions > meshTraversal;
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
          typename Function,
          typename BoundaryConditions >
   template< int EntityDimensions >
void
tnlExactOperatorEvaluator< tnlGrid< Dimensions, Real, Device, Index >, DofVector, DifferentialOperator, Function, BoundaryConditions >::
evaluate( const RealType& time,
          const tnlGrid< Dimensions, Real, Device, Index >& mesh,
          const DifferentialOperator& differentialOperator,
          const Function& function,
          const BoundaryConditions& boundaryConditions,
          DofVector& fu ) const
{
   TraversalUserData userData( time, differentialOperator, function, boundaryConditions, fu );
   TraversalBoundaryEntitiesProcessor boundaryEntitiesProcessor;
   TraversalInteriorEntitiesProcessor interiorEntitiesProcessor;
   tnlTraverser< MeshType, EntityDimensions > meshTraversal;
   meshTraversal.template processEntities< TraversalUserData,
                                           TraversalBoundaryEntitiesProcessor,
                                           TraversalInteriorEntitiesProcessor >
                                          ( mesh,
                                            userData,
                                            boundaryEntitiesProcessor,
                                            interiorEntitiesProcessor );
}



#endif /* TNLEXACTOPERATOREVALUATOR_IMPL_H_ */
