/***************************************************************************
                          HeatEquationEocProblem_impl.h  -  description
                             -------------------
    begin                : Nov 22, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Szekely Ondrej, ondra.szekely@gmail.com
 */


#pragma once

#include "HeatEquationProblem.h"

namespace TNL {
namespace Problems {   

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
String
HeatEquationEocProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getType()
{
   return String( "heatEquationEocSolver< " ) + Mesh :: getType() + " >";
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
HeatEquationEocProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator  >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   if( ! this->boundaryConditionPointer->setup( this->getMesh(), parameters, prefix ) ||
       ! this->rightHandSidePointer->setup( parameters ) )
      return false;
   return true;
}

} // namespace Problems
} // namespace TNL
