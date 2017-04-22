/***************************************************************************
                          transportEquationBuildConfigTag.cpp  -  description
                             -------------------
    begin                : Feb 10, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#pragma once

#include <TNL/Solvers/BuildConfigTags.h>

namespace TNL {

class transportEquationBuildConfigTag{};

namespace Solvers {

/****
 * Turn off support for float and long double.
 */
template<> struct ConfigTagReal< transportEquationBuildConfigTag, float > { enum { enabled = false }; };
template<> struct ConfigTagReal< transportEquationBuildConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct ConfigTagIndex< transportEquationBuildConfigTag, short int >{ enum { enabled = false }; };
template<> struct ConfigTagIndex< transportEquationBuildConfigTag, long int >{ enum { enabled = false }; };

/****
 * Use of Grid is enabled for allowed dimensions and Real, Device and Index types.
 */

template< int Dimensions, typename Real, typename Device, typename Index >
   struct ConfigTagMesh< transportEquationBuildConfigTag, Meshes::Grid< Dimensions, Real, Device, Index > >
      { enum { enabled = ConfigTagDimension< transportEquationBuildConfigTag, Dimensions >::enabled  &&
                         ConfigTagReal< transportEquationBuildConfigTag, Real >::enabled &&
                         ConfigTagDevice< transportEquationBuildConfigTag, Device >::enabled &&
                         ConfigTagIndex< transportEquationBuildConfigTag, Index >::enabled }; };

/****
 * Please, chose your preferred time discretisation  here.
 */
template<> struct ConfigTagTimeDiscretisation< transportEquationBuildConfigTag, ExplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< transportEquationBuildConfigTag, SemiImplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< transportEquationBuildConfigTag, ImplicitTimeDiscretisationTag >{ enum { enabled = true }; };

/****
 * Only the Runge-Kutta-Merson solver is enabled by default.
 */
template<> struct ConfigTagExplicitSolver< transportEquationBuildConfigTag, Solvers::ExplicitEulerSolverTag >{ enum { enabled = true }; };

} // namespace Solvers
} // namespace TNL
