/***************************************************************************
                          tnlFastBuildConfigTag.h  -  description
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
template<> struct tnlConfigTagReal< tnlFastBuildConfig, float > { enum { enabled = false }; };
template<> struct tnlConfigTagReal< tnlFastBuildConfig, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct tnlConfigTagIndex< tnlFastBuildConfig, short int >{ enum { enabled = false }; };
template<> struct tnlConfigTagIndex< tnlFastBuildConfig, long int >{ enum { enabled = false }; };

/****
 * Use of tnlGrid is enabled for allowed dimensions and Real, Device and Index types.
 */
template< int Dimensions, typename Real, typename Device, typename Index >
   struct tnlConfigTagMesh< tnlFastBuildConfig, tnlGrid< Dimensions, Real, Device, Index > >
      { enum { enabled = tnlConfigTagDimensions< tnlFastBuildConfig, Dimensions >::enabled  &&
                         tnlConfigTagReal< tnlFastBuildConfig, Real >::enabled &&
                         tnlConfigTagDevice< tnlFastBuildConfig, Device >::enabled &&
                         tnlConfigTagIndex< tnlFastBuildConfig, Index >::enabled }; };

/****
 * Please, chose your preferred time discretisation  here.
 */
template<> struct tnlConfigTagTimeDiscretisation< tnlFastBuildConfig, tnlExplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct tnlConfigTagTimeDiscretisation< tnlFastBuildConfig, tnlSemiImplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct tnlConfigTagTimeDiscretisation< tnlFastBuildConfig, tnlImplicitTimeDiscretisationTag >{ enum { enabled = false }; };

/****
 * Only the Runge-Kutta-Merson solver is enabled by default.
 */
//template<> struct tnlConfigTagExplicitSolver< tnlFastBuildConfig, tnlExplicitEulerSolverTag >{ enum { enabled = false }; };

} // namespace Solvers
} // namespace TNL
