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

class FastBuildConfigTag
{
   public:

      static void print() { std::cerr << "FastBuildConfigTag" << std::endl; }
};

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
 * Use of Grid is enabled for allowed dimensions and Real, Device and Index types.
 */
template< int Dimensions, typename Real, typename Device, typename Index >
   struct ConfigTagMesh< FastBuildConfigTag, Meshes::Grid< Dimensions, Real, Device, Index > >
      { enum { enabled = ConfigTagDimensions< FastBuildConfigTag, Dimensions >::enabled  &&
                         ConfigTagReal< FastBuildConfigTag, Real >::enabled &&
                         ConfigTagDevice< FastBuildConfigTag, Device >::enabled &&
                         ConfigTagIndex< FastBuildConfigTag, Index >::enabled }; };

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
} // namespace TNL
