#ifndef advectionBUILDCONFIGTAG_H_
#define advectionBUILDCONFIGTAG_H_

#include <TNL/Solvers/BuildConfigTags.h>

namespace TNL {

class advectionBuildConfigTag{};

namespace Solvers {

/****
 * Turn off support for float and long double.
 */
template<> struct ConfigTagReal< advectionBuildConfigTag, float > { enum { enabled = false }; };
template<> struct ConfigTagReal< advectionBuildConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct ConfigTagIndex< advectionBuildConfigTag, short int >{ enum { enabled = false }; };
template<> struct ConfigTagIndex< advectionBuildConfigTag, long int >{ enum { enabled = false }; };

/****
 * Use of Grid is enabled for allowed dimensions and Real, Device and Index types.
 */

template< int Dimensions, typename Real, typename Device, typename Index >
   struct ConfigTagMesh< advectionBuildConfigTag, Meshes::Grid< Dimensions, Real, Device, Index > >
      { enum { enabled = ConfigTagDimensions< advectionBuildConfigTag, Dimensions >::enabled  &&
                         ConfigTagReal< advectionBuildConfigTag, Real >::enabled &&
                         ConfigTagDevice< advectionBuildConfigTag, Device >::enabled &&
                         ConfigTagIndex< advectionBuildConfigTag, Index >::enabled }; };

/****
 * Please, chose your preferred time discretisation  here.
 */
template<> struct ConfigTagTimeDiscretisation< advectionBuildConfigTag, tnlExplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< advectionBuildConfigTag, tnlSemiImplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< advectionBuildConfigTag, tnlImplicitTimeDiscretisationTag >{ enum { enabled = true }; };

/****
 * Only the Runge-Kutta-Merson solver is enabled by default.
 */
template<> struct ConfigTagExplicitSolver< advectionBuildConfigTag, Solvers::tnlExplicitEulerSolverTag >{ enum { enabled = true }; };

} // namespace Solvers
} // namespace TNL

#endif /* advectionBUILDCONFIGTAG_H_ */
