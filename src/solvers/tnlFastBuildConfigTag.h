/***************************************************************************
                          tnlFastBuildConfigTag.h  -  description
                             -------------------
    begin                : Jul 7, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLFASTBUILDCONFIGTAG_H_
#define TNLFASTBUILDCONFIGTAG_H_

#include <solvers/tnlBuildConfigTags.h>

class tnlFastBuildConfig
{
   public:

      static void print() { cerr << "tnlFastBuildConfig" << endl; }
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
template<> struct tnlConfigTagExplicitSolver< tnlFastBuildConfig, tnlExplicitEulerSolverTag >{ enum { enabled = false }; };

#endif /* TNLFASTBUILDCONFIGTAG_H_ */
