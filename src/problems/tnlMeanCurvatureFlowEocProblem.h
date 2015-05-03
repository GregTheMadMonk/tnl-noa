/***************************************************************************
                          tnlHeatEquationEocProblem.h  -  description
                             -------------------
    begin                : Nov 22, 2014
    copyright            : (C) 2014 by oberhuber
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

#ifndef TNLMEANCURVATUREFLOWEOCPROBLEM_H_
#define TNLMEANCURVATUREFLOWEOCPROBLEM_H_

#include <problems/tnlMeanCurvatureFlowProblem.h>
#include <operators/operator-Q/tnlOneSideDiffOperatorQForGraph.h>

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator = tnlNonlinearDiffusion< Mesh,
                                                          tnlOneSideDiffNonlinearOperator< Mesh, tnlOneSideDiffOperatorQForGraph<Mesh, typename BoundaryCondition::RealType,
                                                          typename BoundaryCondition::IndexType, 0>, typename BoundaryCondition::RealType, typename BoundaryCondition::IndexType >, 
                                                          typename BoundaryCondition::RealType, typename BoundaryCondition::IndexType > >
class tnlMeanCurvatureFlowEocProblem : public tnlMeanCurvatureFlowProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator > 
{
   public:

      static tnlString getTypeStatic();

      bool setup( const tnlParameterContainer& parameters );
};

#include <problems/tnlMeanCurvatureFlowEocProblem_impl.h>

#endif /* TNLMEANCURVATUREFLOWEOCPROBLEM_H_ */
