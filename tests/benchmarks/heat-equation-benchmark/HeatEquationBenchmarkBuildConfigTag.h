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
template< int Dimensions, typename Real, typename Device, typename Index >
   struct ConfigTagMesh< HeatEquationBenchmarkBuildConfigTag, Meshes::Grid< Dimensions, Real, Device, Index > >
      { enum { enabled = ( Dimensions == 2 )  &&
                         ConfigTagReal< HeatEquationBenchmarkBuildConfigTag, Real >::enabled &&
                         ConfigTagDevice< HeatEquationBenchmarkBuildConfigTag, Device >::enabled &&
                         ConfigTagIndex< HeatEquationBenchmarkBuildConfigTag, Index >::enabled }; };

/****
 * Please, chose your preferred time discretisation  here.
 */
template<> struct ConfigTagTimeDiscretisation< HeatEquationBenchmarkBuildConfigTag, tnlExplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< HeatEquationBenchmarkBuildConfigTag, tnlSemiImplicitTimeDiscretisationTag >{ enum { enabled = false }; };
template<> struct ConfigTagTimeDiscretisation< HeatEquationBenchmarkBuildConfigTag, tnlImplicitTimeDiscretisationTag >{ enum { enabled = false }; };

/****
 * Only the Runge-Kutta-Merson solver is enabled by default.
 */
template<> struct ConfigTagExplicitSolver< HeatEquationBenchmarkBuildConfigTag, tnlExplicitEulerSolverTag >{ enum { enabled = true }; };
template<> struct ConfigTagExplicitSolver< HeatEquationBenchmarkBuildConfigTag, tnlExplicitMersonSolverTag >{ enum { enabled = false }; };

} // namespace Solvers
} // namespace TNL

