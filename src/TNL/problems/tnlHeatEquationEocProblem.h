/***************************************************************************
                          tnlHeatEquationEocProblem.h  -  description
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

#include <TNL/problems/tnlHeatEquationProblem.h>

namespace TNL {

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

} //namespace TNL

#include <TNL/problems/tnlHeatEquationEocProblem_impl.h>
