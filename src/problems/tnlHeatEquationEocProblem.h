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

#ifndef TNLHEATEQUATIONEOCPROBLEM_H_
#define TNLHEATEQUATIONEOCPROBLEM_H_

#include <problems/tnlHeatEquationProblem.h>


template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator = tnlLinearDiffusion< Mesh,
                                                              typename BoundaryCondition::RealType > >
class tnlHeatEquationEocProblem : public tnlHeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >
{
   public:

      static tnlString getTypeStatic();

      bool setup( const tnlParameterContainer& parameters );
};

#include <problems/tnlHeatEquationEocProblem_impl.h>

#endif /* TNLHEATEQUATIONEOCPROBLEM_H_ */
