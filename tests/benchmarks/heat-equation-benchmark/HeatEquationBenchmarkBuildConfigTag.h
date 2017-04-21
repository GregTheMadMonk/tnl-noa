#pragma once

#include <TNL/Solvers/BuildConfigTags.h>

namespace TNL {

class HeatEquationBenchmarkBuildConfigTag{};

namespace Solvers {

/****
 * Turn off support for float and long double.
 */
template<> struct ConfigTagReal< HeatEquationBenchmarkBuildConfigTag, float > { enum { enabled = false }; };
template<> struct ConfigTagReal< HeatEquationBenchmarkBuildConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct ConfigTagIndex< HeatEquationBenchmarkBuildConfigTag, short int >{ enum { enabled = false }; };
template<> struct ConfigTagIndex< HeatEquationBenchmarkBuildConfigTag, long int >{ enum { enabled = false }; };

/****
 * Use of Grid is enabled for allowed dimensions and Real, Device and Index types.
 */
template< int Dimension, typename Real, typename Device, typename Index >
   struct ConfigTagMesh< HeatEquationBenchmarkBuildConfigTag, Meshes::Grid< Dimension, Real, Device, Index > >
      { enum { enabled = ( Dimension == 2 )  &&
                         ConfigTagReal< HeatEquationBenchmarkBuildConfigTag, Real >::enabled &&
                         ConfigTagDevice< HeatEquationBenchmarkBuildConfigTag, Device >::enabled &&
                         ConfigTagIndex< HeatEquationBenchmarkBuildConfigTag, Index >::enabled }; };

/****
 * Please, chose your preferred time discretisation  here.
 */
template<> struct ConfigTagTimeDiscretisation< HeatEquationBenchmarkBuildConfigTag, ExplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< HeatEquationBenchmarkBuildConfigTag, SemiImplicitTimeDiscretisationTag >{ enum { enabled = false }; };
template<> struct ConfigTagTimeDiscretisation< HeatEquationBenchmarkBuildConfigTag, ImplicitTimeDiscretisationTag >{ enum { enabled = false }; };

/****
 * Only the Runge-Kutta-Merson solver is enabled by default.
 */
template<> struct ConfigTagExplicitSolver< HeatEquationBenchmarkBuildConfigTag, ExplicitEulerSolverTag >{ enum { enabled = true }; };
template<> struct ConfigTagExplicitSolver< HeatEquationBenchmarkBuildConfigTag, ExplicitMersonSolverTag >{ enum { enabled = false }; };

} // namespace Solvers
} // namespace TNL

