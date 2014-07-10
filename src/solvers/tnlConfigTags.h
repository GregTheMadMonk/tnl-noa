/***************************************************************************
                          tnlConfigTags.h  -  description
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

#ifndef TNLCONFIGTAGS_H_
#define TNLCONFIGTAGS_H_

#include <mesh/tnlGrid.h>

class tnlDefaultConfigTag{};

/****
 * All devices are enabled by default. Those which are not available
 * are disabled.
 */
template< typename ConfigTag, typename Device > struct tnlConfigTagDevice{ enum { enabled = true }; };
#ifndef HAVE_CUDA
template< typename ConfigTag > struct tnlConfigTagDevice< ConfigTag, tnlCuda >{ enum { enabled = false }; };
#endif

/****
 * All real types are enabled by default.
 */
template< typename ConfigTag, typename Real > struct tnlConfigTagReal{ enum { enabled = true }; };

/****
 * All index types are enabled ba default.
 */
template< typename ConfigTag, typename Index > struct tnlConfigTagIndex{ enum { enabled = true }; };

/****
 * The mesh type will be resolved by the tnlSolver by default.
 */
template< typename ConfigTag > struct tnlConfigTagMeshResolve{ enum { enabled = true }; };

/****
 * 1, 2, and 3 dimensions are enabled by default
 */
template< typename ConfigTag, int Dimensions > struct tnlConfigTagDimensions{ enum { enabled = false }; };
   template< typename ConfigTag > struct tnlConfigTagDimensions< ConfigTag, 1 >{ enum { enabled = true }; };
   template< typename ConfigTag > struct tnlConfigTagDimensions< ConfigTag, 2 >{ enum { enabled = true }; };
   template< typename ConfigTag > struct tnlConfigTagDimensions< ConfigTag, 3 >{ enum { enabled = true }; };

/****
 * Up to the exceptions enlisted below, all mesh types are disabled by default.
 */
template< typename ConfigTag, typename MeshType > struct tnlConfigTagMesh{ enum { enabled = false }; };

/****
 * Use of tnlGrid is enabled for allowed dimensions by default.
 */
template< typename ConfigTag, int Dimensions, typename Real, typename Device, typename Index >
   struct tnlConfigTagMesh< ConfigTag, tnlGrid< Dimensions, Real, Device, Index > >
      { enum { enabled = tnlConfigTagDimensions< ConfigTag, Dimensions >::enabled }; };

/****
 * All time discretisations (explicit, semi-impicit and implicit ) are
 * enabled by default.
 */
class tnlExplicitTimeDiscretisationTag{};
class tnlSemiImplicitTimeDiscretisationTag{};
class tnlImplicitTimeDiscretisationTag{};

template< typename ConfigTag, typename TimeDiscretisation > struct tnlConfigTagTimeDiscretisation{ enum { enabled = true }; };

/****
 * All explicit solvers are enabled by default
 */
class tnlExplicitEulerSolverTag{};
class tnlExplicitMersonSolverTag{};

template< typename ConfigTag, typename ExplicitSolver > struct tnlConfigTagExplicitSolver{ enum { enabled = true }; };


#endif /* TNLCONFIGTAGS_H_ */
