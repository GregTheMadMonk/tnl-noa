/***************************************************************************
                          FastBuildConfigTag.h  -  description
                             -------------------
    begin                : Jul 7, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Meshes/TypeResolver/BuildConfigTags.h>

namespace TNL {
namespace Solvers {

class FastBuildConfigTag {};

/****
 * Turn off support for float and long double.
 */
template<> struct ConfigTagReal< FastBuildConfigTag, float > { enum { enabled = false }; };
template<> struct ConfigTagReal< FastBuildConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct ConfigTagIndex< FastBuildConfigTag, short int >{ enum { enabled = false }; };
template<> struct ConfigTagIndex< FastBuildConfigTag, long int >{ enum { enabled = false }; };

/****
 * Please, chose your preferred time discretisation  here.
 */
template<> struct ConfigTagTimeDiscretisation< FastBuildConfigTag, ExplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< FastBuildConfigTag, SemiImplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< FastBuildConfigTag, ImplicitTimeDiscretisationTag >{ enum { enabled = false }; };

/****
 * Only the Runge-Kutta-Merson solver is enabled by default.
 */
//template<> struct ConfigTagExplicitSolver< FastBuildConfigTag, ExplicitEulerSolverTag >{ enum { enabled = false }; };

} // namespace Solvers

namespace Meshes {
namespace BuildConfigTags {

/****
 * Turn off support for float and long double.
 */
template<> struct GridRealTag< Solvers::FastBuildConfigTag, float > { enum { enabled = false }; };
template<> struct GridRealTag< Solvers::FastBuildConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct GridIndexTag< Solvers::FastBuildConfigTag, short int >{ enum { enabled = false }; };
template<> struct GridIndexTag< Solvers::FastBuildConfigTag, long int >{ enum { enabled = false }; };

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL
