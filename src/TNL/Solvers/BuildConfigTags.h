/***************************************************************************
                          BuildConfigTags.h  -  description
                             -------------------
    begin                : Jul 7, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Solvers/ODE/Merson.h>
#include <TNL/Solvers/ODE/Euler.h>

namespace TNL {
namespace Solvers {

class DefaultBuildConfigTag {};

/****
 * All devices are enabled by default. Those which are not available
 * are disabled.
 */
template< typename ConfigTag, typename Device > struct ConfigTagDevice{ enum { enabled = true }; };
#ifndef HAVE_CUDA
template< typename ConfigTag > struct ConfigTagDevice< ConfigTag, Devices::Cuda >{ enum { enabled = false }; };
#endif

/****
 * All real types are enabled by default.
 */
template< typename ConfigTag, typename Real > struct ConfigTagReal{ enum { enabled = true }; };

/****
 * All index types are enabled by default.
 */
template< typename ConfigTag, typename Index > struct ConfigTagIndex{ enum { enabled = true }; };

/****
 * The mesh type will be resolved by the Solver by default.
 * (The detailed mesh configuration is in TNL/Meshes/TypeResolver/BuildConfigTags.h)
 */
template< typename ConfigTag > struct ConfigTagMeshResolve{ enum { enabled = true }; };

/****
 * All time discretisations (explicit, semi-impicit and implicit ) are
 * enabled by default.
 */
class ExplicitTimeDiscretisationTag{};
class SemiImplicitTimeDiscretisationTag{};
class ImplicitTimeDiscretisationTag{};

template< typename ConfigTag, typename TimeDiscretisation > struct ConfigTagTimeDiscretisation{ enum { enabled = true }; };

/****
 * All explicit solvers are enabled by default
 */
class ExplicitEulerSolverTag
{
public:
    template< typename Problem, typename SolverMonitor >
    using Template = ODE::Euler< Problem, SolverMonitor >;
};

class ExplicitMersonSolverTag
{
public:
    template< typename Problem, typename SolverMonitor >
    using Template = ODE::Merson< Problem, SolverMonitor >;
};

template< typename ConfigTag, typename ExplicitSolver > struct ConfigTagExplicitSolver{ enum { enabled = true }; };

} // namespace Solvers
} // namespace TNL
