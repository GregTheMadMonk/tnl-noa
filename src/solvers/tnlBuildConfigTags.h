/***************************************************************************
                          tnlMeshConfigs.h  -  description
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

#ifndef TNLMeshConfigS_H_
#define TNLMeshConfigS_H_

#include <mesh/tnlGrid.h>

class tnlDefaultBuildMeshConfig{};

/****
 * All devices are enabled by default. Those which are not available
 * are disabled.
 */
template< typename MeshConfig, typename Device > struct tnlMeshConfigDevice{ enum { enabled = true }; };
#ifndef HAVE_CUDA
template< typename MeshConfig > struct tnlMeshConfigDevice< MeshConfig, tnlCuda >{ enum { enabled = false }; };
#endif

/****
 * All real types are enabled by default.
 */
template< typename MeshConfig, typename Real > struct tnlMeshConfigReal{ enum { enabled = true }; };

/****
 * All index types are enabled ba default.
 */
template< typename MeshConfig, typename Index > struct tnlMeshConfigIndex{ enum { enabled = true }; };

/****
 * The mesh type will be resolved by the tnlSolver by default.
 */
template< typename MeshConfig > struct tnlMeshConfigMeshResolve{ enum { enabled = true }; };

/****
 * 1, 2, and 3 dimensions are enabled by default
 */
template< typename MeshConfig, int Dimensions > struct tnlMeshConfigDimensions{ enum { enabled = false }; };
   template< typename MeshConfig > struct tnlMeshConfigDimensions< MeshConfig, 1 >{ enum { enabled = true }; };
   template< typename MeshConfig > struct tnlMeshConfigDimensions< MeshConfig, 2 >{ enum { enabled = true }; };
   template< typename MeshConfig > struct tnlMeshConfigDimensions< MeshConfig, 3 >{ enum { enabled = true }; };

/****
 * Up to the exceptions enlisted below, all mesh types are disabled by default.
 */
template< typename MeshConfig, typename MeshType > struct tnlMeshConfigMesh{ enum { enabled = false }; };

/****
 * Use of tnlGrid is enabled for allowed dimensions by default.
 */
template< typename MeshConfig, int Dimensions, typename Real, typename Device, typename Index >
   struct tnlMeshConfigMesh< MeshConfig, tnlGrid< Dimensions, Real, Device, Index > >
      { enum { enabled = tnlMeshConfigDimensions< MeshConfig, Dimensions >::enabled }; };

/****
 * All time discretisations (explicit, semi-impicit and implicit ) are
 * enabled by default.
 */
class tnlExplicitTimeDiscretisationTag{};
class tnlSemiImplicitTimeDiscretisationTag{};
class tnlImplicitTimeDiscretisationTag{};

template< typename MeshConfig, typename TimeDiscretisation > struct tnlConfigTagTimeDiscretisation{ enum { enabled = true }; };

/****
 * All explicit solvers are enabled by default
 */
class tnlExplicitEulerSolverTag{};
class tnlExplicitMersonSolverTag{};

template< typename MeshConfig, typename ExplicitSolver > struct tnlMeshConfigExplicitSolver{ enum { enabled = true }; };

/****
 * All semi-implicit solvers are enabled by default
 */

class  tnlSemiImplicitSORSolverTag{};
class  tnlSemiImplicitCGSolverTag{};
class  tnlSemiImplicitBICGStabSolverTag{};
class  tnlSemiImplicitGMRESSolverTag{};

template< typename MeshConfig, typename SemiImplicitSolver > struct tnlMeshConfigSemiImplicitSolver{ enum { enabled = true }; };

#endif /* TNLMeshConfigS_H_ */
