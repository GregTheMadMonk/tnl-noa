/***************************************************************************
                          heatEquationEocSolver_impl.h  -  description
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

#ifndef HEATEQUATIONEOCSOLVER_IMPL_H_
#define HEATEQUATIONEOCSOLVER_IMPL_H_

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryCondition,
          typename RightHandSide >
tnlString
heatEquationEocSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::
getTypeStatic()
{
   return tnlString( "heatEquationEocSolver< " ) + Mesh :: getTypeStatic() + " >";
}

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryCondition,
          typename RightHandSide >
bool
heatEquationEocSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >::
setup( const tnlParameterContainer& parameters )
{
   if( ! this->boundaryCondition.setup( parameters ) ||
       ! this->rightHandSide.setup( parameters ) )
      return false;
   return true;
}

#endif /* HEATEQUATIONEOCSOLVER_IMPL_H_ */
