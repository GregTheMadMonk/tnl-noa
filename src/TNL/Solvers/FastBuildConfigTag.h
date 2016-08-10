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

class tnlFastBuildConfig
{
   public:

      static void print() { std::cerr << "tnlFastBuildConfig" << std::endl; }
};

/****
 * Turn off support for float and long double.
 */
template<> struct ConfigTagReal< tnlFastBuildConfig, float > { enum { enabled = false }; };
template<> struct ConfigTagReal< tnlFastBuildConfig, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct ConfigTagIndex< tnlFastBuildConfig, short int >{ enum { enabled = false }; };
template<> struct ConfigTagIndex< tnlFastBuildConfig, long int >{ enum { enabled = false }; };

/****
 * Use of Grid is enabled for allowed dimensions and Real, Device and Index types.
 */
template< int Dimensions, typename Real, typename Device, typename Index >
   struct ConfigTagMesh< tnlFastBuildConfig, Meshes::Grid< Dimensions, Real, Device, Index > >
      { enum { enabled = ConfigTagDimensions< tnlFastBuildConfig, Dimensions >::enabled  &&
                         ConfigTagReal< tnlFastBuildConfig, Real >::enabled &&
                         ConfigTagDevice< tnlFastBuildConfig, Device >::enabled &&
                         ConfigTagIndex< tnlFastBuildConfig, Index >::enabled }; };

/****
 * Please, chose your preferred time discretisation  here.
 */
template<> struct ConfigTagTimeDiscretisation< tnlFastBuildConfig, tnlExplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< tnlFastBuildConfig, tnlSemiImplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< tnlFastBuildConfig, tnlImplicitTimeDiscretisationTag >{ enum { enabled = false }; };

/****
 * Only the Runge-Kutta-Merson solver is enabled by default.
 */
//template<> struct ConfigTagExplicitSolver< tnlFastBuildConfig, tnlExplicitEulerSolverTag >{ enum { enabled = false }; };

} // namespace Solvers
} // namespace TNL
