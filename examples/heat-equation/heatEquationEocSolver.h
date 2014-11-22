/***************************************************************************
                          heatEquationEocSolver.h  -  description
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

#ifndef HEATEQUATIONEOCSOLVER_H_
#define HEATEQUATIONEOCSOLVER_H_

#include "heatEquationSolver.h"


template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryCondition,
          typename RightHandSide >
class heatEquationEocSolver : public heatEquationSolver< Mesh, DifferentialOperator, BoundaryCondition, RightHandSide >
{
   public:

      static tnlString getTypeStatic();

      bool setup( const tnlParameterContainer& parameters );
};

#include "heatEquationEocSolver_impl.h"

#endif /* HEATEQUATIONEOCSOLVER_H_ */
