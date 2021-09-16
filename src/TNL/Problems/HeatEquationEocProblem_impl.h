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

#include "HeatEquationEocProblem.h"

namespace TNL {
namespace Problems {

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
   this->explicitUpdater.setDifferentialOperator( this->differentialOperatorPointer );
   this->explicitUpdater.setBoundaryConditions( this->boundaryConditionPointer );
   this->explicitUpdater.setRightHandSide( this->rightHandSidePointer );
   this->systemAssembler.setDifferentialOperator( this->differentialOperatorPointer );
   this->systemAssembler.setBoundaryConditions( this->boundaryConditionPointer );
   this->systemAssembler.setRightHandSide( this->rightHandSidePointer );
   return true;
}

} // namespace Problems
} // namespace TNL
