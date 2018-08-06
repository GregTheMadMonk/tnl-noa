/***************************************************************************
                          HeatEquationEocProblem.h  -  description
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

#include <TNL/Problems/HeatEquationProblem.h>

namespace TNL {
namespace Problems {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename Communicator,
          typename DifferentialOperator = Operators::LinearDiffusion< Mesh,
                                                              typename BoundaryCondition::RealType > >
class HeatEquationEocProblem : public HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, DifferentialOperator >
{
   public:
      
      typedef HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, Communicator, DifferentialOperator > BaseType;
      
      using typename BaseType::MeshPointer;

      static String getType();

      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix );
};

} // namespace Problems
} // namespace TNL

#include <TNL/Problems/HeatEquationEocProblem_impl.h>
