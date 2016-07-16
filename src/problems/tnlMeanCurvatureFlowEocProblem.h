/***************************************************************************
                          tnlHeatEquationEocProblem.h  -  description
                             -------------------
    begin                : Nov 22, 2014
    copyright            : (C) 2014 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLMEANCURVATUREFLOWEOCPROBLEM_H_
#define TNLMEANCURVATUREFLOWEOCPROBLEM_H_

#include <problems/tnlMeanCurvatureFlowProblem.h>
#include <operators/operator-Q/tnlOneSideDiffOperatorQ.h>

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator = tnlNonlinearDiffusion< Mesh,
                                                          tnlOneSideDiffNonlinearOperator< Mesh, tnlOneSideDiffOperatorQ<Mesh, typename BoundaryCondition::RealType,
                                                          typename BoundaryCondition::IndexType >, typename BoundaryCondition::RealType, typename BoundaryCondition::IndexType >,
                                                          typename BoundaryCondition::RealType, typename BoundaryCondition::IndexType > >
class tnlMeanCurvatureFlowEocProblem : public tnlMeanCurvatureFlowProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >
{
   public:

      static tnlString getTypeStatic();

      bool setup( const tnlParameterContainer& parameters );
};

#include <problems/tnlMeanCurvatureFlowEocProblem_impl.h>

#endif /* TNLMEANCURVATUREFLOWEOCPROBLEM_H_ */
