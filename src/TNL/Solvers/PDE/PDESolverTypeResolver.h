/***************************************************************************
                          PDESolverTypeResolver.h  -  description
                             -------------------
    begin                : Nov 10, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Solvers/PDE/TimeDependentPDESolver.h>
#include <TNL/Solvers/PDE/TimeIndependentPDESolver.h>

namespace TNL {
namespace Solvers {
namespace PDE {

template< typename Problem,
          typename TimeStepper,
          bool TimeDependent = Problem::isTimeDependent() >
class PDESolverTypeResolver
{
};

template< typename Problem,
          typename TimeStepper >
class PDESolverTypeResolver< Problem, TimeStepper, true >
{
   public:

      using SolverType = TimeDependentPDESolver< Problem, TimeStepper >;
};

template< typename Problem,
          typename TimeStepper >
class PDESolverTypeResolver< Problem, TimeStepper, false >
{
   public:

      using SolverType = TimeIndependentPDESolver< Problem >;
};

} // namespace PDE
} // namespace Solvers
} // namespace TNL
