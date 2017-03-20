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

namespace TNL {
namespace Solvers {   

class FastBuildConfig
{
   public:

      static void print() { std::cerr << "FastBuildConfig" << std::endl; }
};

/****
 * Turn off support for float and long double.
 */
template<> struct ConfigTagReal< FastBuildConfig, float > { enum { enabled = false }; };
template<> struct ConfigTagReal< FastBuildConfig, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct ConfigTagIndex< FastBuildConfig, short int >{ enum { enabled = false }; };
template<> struct ConfigTagIndex< FastBuildConfig, long int >{ enum { enabled = false }; };

/****
 * Use of Grid is enabled for allowed dimensions and Real, Device and Index types.
 */
template< int Dimensions, typename Real, typename Device, typename Index >
   struct ConfigTagMesh< FastBuildConfig, Meshes::Grid< Dimensions, Real, Device, Index > >
      { enum { enabled = ConfigTagDimensions< FastBuildConfig, Dimensions >::enabled  &&
                         ConfigTagReal< FastBuildConfig, Real >::enabled &&
                         ConfigTagDevice< FastBuildConfig, Device >::enabled &&
                         ConfigTagIndex< FastBuildConfig, Index >::enabled }; };

/****
 * Please, chose your preferred time discretisation  here.
 */
template<> struct ConfigTagTimeDiscretisation< FastBuildConfig, ExplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< FastBuildConfig, SemiImplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< FastBuildConfig, ImplicitTimeDiscretisationTag >{ enum { enabled = false }; };

/****
 * Only the Runge-Kutta-Merson solver is enabled by default.
 */
//template<> struct ConfigTagExplicitSolver< FastBuildConfig, ExplicitEulerSolverTag >{ enum { enabled = false }; };

} // namespace Solvers
} // namespace TNL
