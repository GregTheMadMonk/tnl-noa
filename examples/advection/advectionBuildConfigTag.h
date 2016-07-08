#ifndef advectionBUILDCONFIGTAG_H_
#define advectionBUILDCONFIGTAG_H_

#include <solvers/tnlBuildConfigTags.h>

class advectionBuildConfigTag{};

/****
 * Turn off support for float and long double.
 */
template<> struct tnlConfigTagReal< advectionBuildConfigTag, float > { enum { enabled = false }; };
template<> struct tnlConfigTagReal< advectionBuildConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct tnlConfigTagIndex< advectionBuildConfigTag, short int >{ enum { enabled = false }; };
template<> struct tnlConfigTagIndex< advectionBuildConfigTag, long int >{ enum { enabled = false }; };

/****
 * Use of tnlGrid is enabled for allowed dimensions and Real, Device and Index types.
 */

template< int Dimensions, typename Real, typename Device, typename Index >
   struct tnlConfigTagMesh< advectionBuildConfigTag, tnlGrid< Dimensions, Real, Device, Index > >
      { enum { enabled = tnlConfigTagDimensions< advectionBuildConfigTag, Dimensions >::enabled  &&
                         tnlConfigTagReal< advectionBuildConfigTag, Real >::enabled &&
                         tnlConfigTagDevice< advectionBuildConfigTag, Device >::enabled &&
                         tnlConfigTagIndex< advectionBuildConfigTag, Index >::enabled }; };

/****
 * Please, chose your preferred time discretisation  here.
 */
template<> struct tnlConfigTagTimeDiscretisation< advectionBuildConfigTag, tnlExplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct tnlConfigTagTimeDiscretisation< advectionBuildConfigTag, tnlSemiImplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct tnlConfigTagTimeDiscretisation< advectionBuildConfigTag, tnlImplicitTimeDiscretisationTag >{ enum { enabled = true }; };

/****
 * Only the Runge-Kutta-Merson solver is enabled by default.
 */
template<> struct tnlConfigTagExplicitSolver< advectionBuildConfigTag, tnlExplicitEulerSolverTag >{ enum { enabled = true }; };

#endif /* advectionBUILDCONFIGTAG_H_ */
