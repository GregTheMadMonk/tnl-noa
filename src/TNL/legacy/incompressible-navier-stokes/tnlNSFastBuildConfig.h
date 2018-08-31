/***************************************************************************
                          tnlNSFastBuildConfig.h  -  description
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

#ifndef TNLNSFASTBUILDCONFIG_H_
#define TNLNSFASTBUILDCONFIG_H_

class tnlNSFastBuildConfig
{
   public:

      static void print() { cerr << "tnlNSFastBuildConfig" << endl; }
};

/****
 * Turn off support for float and long double.
 */
template<> struct tnlConfigTagReal< tnlNSFastBuildConfig, float > { enum { enabled = false }; };
template<> struct tnlConfigTagReal< tnlNSFastBuildConfig, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct tnlConfigTagIndex< tnlNSFastBuildConfig, short int >{ enum { enabled = false }; };
template<> struct tnlConfigTagIndex< tnlNSFastBuildConfig, long int >{ enum { enabled = false }; };

/****
 * 1, 2, and 3 dimensions are enabled by default
 */
template<> struct tnlConfigTagDimensions< tnlNSFastBuildConfig, 1 >{ enum { enabled = false }; };
template<> struct tnlConfigTagDimensions< tnlNSFastBuildConfig, 2 >{ enum { enabled = true }; };
template<> struct tnlConfigTagDimensions< tnlNSFastBuildConfig, 3 >{ enum { enabled = false }; };

/****
 * Use of tnlGrid is enabled for allowed dimensions and Real, Device and Index types.
 */
template< int Dimensions, typename Real, typename Device, typename Index >
   struct tnlConfigTagMesh< tnlNSFastBuildConfig, tnlGrid< Dimensions, Real, Device, Index > >
      { enum { enabled = tnlConfigTagDimensions< tnlNSFastBuildConfig, Dimensions >::enabled  &&
                         tnlConfigTagReal< tnlNSFastBuildConfig, Real >::enabled &&
                         tnlConfigTagDevice< tnlNSFastBuildConfig, Device >::enabled &&
                         tnlConfigTagIndex< tnlNSFastBuildConfig, Index >::enabled }; };

/****
 * Please, chose your preferred time discretisation  here.
 */
template<> struct tnlConfigTagTimeDiscretisation< tnlNSFastBuildConfig, tnlExplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct tnlConfigTagTimeDiscretisation< tnlNSFastBuildConfig, tnlSemiImplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct tnlConfigTagTimeDiscretisation< tnlNSFastBuildConfig, tnlImplicitTimeDiscretisationTag >{ enum { enabled = false }; };

/****
 * Only the Runge-Kutta-Merson solver is enabled by default.
 */
template<> struct tnlConfigTagExplicitSolver< tnlNSFastBuildConfig, tnlExplicitEulerSolverTag >{ enum { enabled = false }; };

#endif /* TNLNSFASTBUILDCONFIG_H_ */
