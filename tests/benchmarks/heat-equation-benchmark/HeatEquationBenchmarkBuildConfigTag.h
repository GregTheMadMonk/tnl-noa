#pragma once

#include <TNL/Solvers/BuildConfigTags.h>

namespace TNL {

class HeatEquationBenchmarkBuildConfigTag{};

namespace Solvers {

/****
 * Turn off support for float and long double.
 */
template<> struct tnlConfigTagReal< HeatEquationBenchmarkBuildConfigTag, float > { enum { enabled = false }; };
template<> struct tnlConfigTagReal< HeatEquationBenchmarkBuildConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct tnlConfigTagIndex< HeatEquationBenchmarkBuildConfigTag, short int >{ enum { enabled = false }; };
template<> struct tnlConfigTagIndex< HeatEquationBenchmarkBuildConfigTag, long int >{ enum { enabled = false }; };

/****
 * Use of tnlGrid is enabled for allowed dimensions and Real, Device and Index types.
 */
template< int Dimensions, typename Real, typename Device, typename Index >
   struct tnlConfigTagMesh< HeatEquationBenchmarkBuildConfigTag, tnlGrid< Dimensions, Real, Device, Index > >
      { enum { enabled = ( Dimensions == 2 )  &&
                         tnlConfigTagReal< HeatEquationBenchmarkBuildConfigTag, Real >::enabled &&
                         tnlConfigTagDevice< HeatEquationBenchmarkBuildConfigTag, Device >::enabled &&
                         tnlConfigTagIndex< HeatEquationBenchmarkBuildConfigTag, Index >::enabled }; };

/****
 * Please, chose your preferred time discretisation  here.
 */
template<> struct tnlConfigTagTimeDiscretisation< HeatEquationBenchmarkBuildConfigTag, tnlExplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct tnlConfigTagTimeDiscretisation< HeatEquationBenchmarkBuildConfigTag, tnlSemiImplicitTimeDiscretisationTag >{ enum { enabled = false }; };
template<> struct tnlConfigTagTimeDiscretisation< HeatEquationBenchmarkBuildConfigTag, tnlImplicitTimeDiscretisationTag >{ enum { enabled = false }; };

/****
 * Only the Runge-Kutta-Merson solver is enabled by default.
 */
template<> struct tnlConfigTagExplicitSolver< HeatEquationBenchmarkBuildConfigTag, tnlExplicitEulerSolverTag >{ enum { enabled = true }; };
template<> struct tnlConfigTagExplicitSolver< HeatEquationBenchmarkBuildConfigTag, tnlExplicitMersonSolverTag >{ enum { enabled = false }; };

} // namespace Solvers
} // namespace TNL

