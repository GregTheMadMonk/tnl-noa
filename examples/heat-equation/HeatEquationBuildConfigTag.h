/***************************************************************************
                          HeatEquationBuildConfigTag.h  -  description
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
   
class HeatEquationBuildConfig
{
   public:

      static void print() { std::cerr << "HeatEquationBuildConfig" << std::endl; }
};

/****
 * Turn off support for float and long double.
 */
template<> struct ConfigTagReal< HeatEquationBuildConfig, float > { enum { enabled = false }; };
template<> struct ConfigTagReal< HeatEquationBuildConfig, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct ConfigTagIndex< HeatEquationBuildConfig, short int >{ enum { enabled = false }; };
template<> struct ConfigTagIndex< HeatEquationBuildConfig, long int >{ enum { enabled = false }; };

/****
 * Use of Grid is enabled for allowed dimensions and Real, Device and Index types.
 */
template< int Dimensions, typename Real, typename Device, typename Index >
   struct ConfigTagMesh< HeatEquationBuildConfig, Meshes::Grid< Dimensions, Real, Device, Index > >
      { enum { enabled = ConfigTagDimensions< HeatEquationBuildConfig, Dimensions >::enabled  &&
                         ConfigTagReal< HeatEquationBuildConfig, Real >::enabled &&
                         ConfigTagDevice< HeatEquationBuildConfig, Device >::enabled &&
                         ConfigTagIndex< HeatEquationBuildConfig, Index >::enabled }; };

/****
 * Please, chose your preferred time discretization  here.
 */
template<> struct ConfigTagTimeDiscretisation< HeatEquationBuildConfig, ExplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< HeatEquationBuildConfig, SemiImplicitTimeDiscretisationTag >{ enum { enabled = false }; };
template<> struct ConfigTagTimeDiscretisation< HeatEquationBuildConfig, ImplicitTimeDiscretisationTag >{ enum { enabled = false }; };

/****
 * Only the Runge-Kutta-Merson solver is enabled by default.
 */
template<> struct ConfigTagExplicitSolver< HeatEquationBuildConfig, ExplicitEulerSolverTag >{ enum { enabled = false }; };

} // namespace Solvers
} // namespace TNL
