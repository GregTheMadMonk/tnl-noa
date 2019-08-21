/***************************************************************************
                          HeatEquationEocProblem.h  -  description
                             -------------------
    begin                : Nov 22, 2014
    copyright            : (C) 2014 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Szekely Ondrej, ondra.szekely@gmail.com
 */

#pragma once

#include <TNL/Problems/MeanCurvatureFlowProblem.h>
#include <TNL/Operators/operator-Q/tnlOneSideDiffOperatorQ.h>

namespace TNL {
namespace Problems {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator = NonlinearDiffusion< Mesh,
                                                          tnlOneSideDiffNonlinearOperator< Mesh, tnlOneSideDiffOperatorQ<Mesh, typename BoundaryCondition::RealType,
                                                          typename BoundaryCondition::IndexType >, typename BoundaryCondition::RealType, typename BoundaryCondition::IndexType >,
                                                          typename BoundaryCondition::RealType, typename BoundaryCondition::IndexType > >
class MeanCurvatureFlowEocProblem : public MeanCurvatureFlowProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >
{
   public:

      bool setup( const Config::ParameterContainer& parameters );
};

} // namespace Problems
} // namespace TNL

#include <TNL/Problems/MeanCurvatureFlowEocProblem_impl.h>
