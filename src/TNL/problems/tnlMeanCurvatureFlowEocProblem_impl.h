/***************************************************************************
                          tnlHeatEquationEocProblem_impl.h  -  description
                             -------------------
    begin                : Nov 22, 2014
    copyright            : (C) 2014 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
tnlString
tnlMeanCurvatureFlowEocProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getTypeStatic()
{
   return tnlString( "tnlHeatEquationEocProblem< " ) + Mesh :: getTypeStatic() + " >";
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
tnlMeanCurvatureFlowEocProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator  >::
setup( const tnlParameterContainer& parameters )
{
   if( ! this->boundaryCondition.setup( parameters ) ||
       ! this->rightHandSide.setup( parameters ) ||
       ! this->differentialOperator.nonlinearDiffusionOperator.operatorQ.setEps(parameters.getParameter< double >("eps")) )
      return false;
 
   return true;
}

} // namespace TNL