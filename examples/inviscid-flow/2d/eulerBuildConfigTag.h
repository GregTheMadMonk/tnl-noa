#ifndef eulerBUILDCONFIGTAG_H_
#define eulerBUILDCONFIGTAG_H_

#include <solvers/tnlBuildConfigTags.h>

class eulerBuildConfigTag{};

/****
 * Turn off support for float and long double.
 */
template<> struct tnlConfigTagReal< eulerBuildConfigTag, float > { enum { enabled = false }; };
template<> struct tnlConfigTagReal< eulerBuildConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct tnlConfigTagIndex< eulerBuildConfigTag, short int >{ enum { enabled = false }; };
template<> struct tnlConfigTagIndex< eulerBuildConfigTag, long int >{ enum { enabled = false }; };

template< int Dimensions > struct tnlConfigTagDimensions< eulerBuildConfigTag, Dimensions >{ enum { enabled = ( Dimensions == 2 ) }; };

/****
 * Use of tnlGrid is enabled for allowed dimensions and Real, Device and Index types.
 */
template< int Dimensions, typename Real, typename Device, typename Index >
   struct tnlConfigTagMesh< eulerBuildConfigTag, tnlGrid< Dimensions, Real, Device, Index > >
      { enum { enabled = tnlConfigTagDimensions< eulerBuildConfigTag, Dimensions >::enabled  &&
                         tnlConfigTagReal< eulerBuildConfigTag, Real >::enabled &&
                         tnlConfigTagDevice< eulerBuildConfigTag, Device >::enabled &&
                         tnlConfigTagIndex< eulerBuildConfigTag, Index >::enabled }; };

/****
 * Please, chose your preferred time discretisation  here.
 */
template<> struct tnlConfigTagTimeDiscretisation< eulerBuildConfigTag, tnlExplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct tnlConfigTagTimeDiscretisation< eulerBuildConfigTag, tnlSemiImplicitTimeDiscretisationTag >{ enum { enabled = false }; };
template<> struct tnlConfigTagTimeDiscretisation< eulerBuildConfigTag, tnlImplicitTimeDiscretisationTag >{ enum { enabled = false }; };

/****
 * Only the Runge-Kutta-Merson solver is enabled by default.
 */
template<> struct tnlConfigTagExplicitSolver< eulerBuildConfigTag, tnlExplicitEulerSolverTag >{ enum { enabled = false }; };

#endif /* eulerBUILDCONFIGTAG_H_ */
