/***************************************************************************
                          tnlHeatEquationEocProblem_impl.h  -  description
                             -------------------
    begin                : Nov 22, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
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

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Szekely Ondrej, ondra.szekely@gmail.com
 */


#ifndef TNLHEATEQUATIONEOCPROBLEM_IMPL_H_
#define TNLHEATEQUATIONEOCPROBLEM_IMPL_H_

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
tnlString
tnlHeatEquationEocProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getTypeStatic()
{
   return tnlString( "heatEquationEocSolver< " ) + Mesh :: getTypeStatic() + " >";
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
tnlHeatEquationEocProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator  >::
setup( const tnlParameterContainer& parameters )
{
   if( ! this->boundaryCondition.setup( parameters ) ||
       ! this->rightHandSide.setup( parameters ) )
      return false;
   return true;
}

#endif /* TNLHEATEQUATIONEOCPROBLEM_IMPL_H_ */
