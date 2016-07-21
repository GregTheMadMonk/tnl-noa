/***************************************************************************
                          tnlBuildConfigTags.h  -  description
                             -------------------
    begin                : Jul 7, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/mesh/tnlGrid.h>
#include <TNL/solvers/ode/tnlMersonSolver.h>
#include <TNL/solvers/ode/tnlEulerSolver.h>
#include <TNL/solvers/linear/stationary/tnlSORSolver.h>
#include <TNL/solvers/linear/krylov/tnlCGSolver.h>
#include <TNL/solvers/linear/krylov/tnlBICGStabSolver.h>
#include <TNL/solvers/linear/krylov/tnlGMRESSolver.h>
#include <TNL/solvers/linear/krylov/tnlTFQMRSolver.h>
#include <TNL/solvers/linear/tnlUmfpackWrapper.h>
#include <TNL/solvers/preconditioners/tnlDummyPreconditioner.h>

namespace TNL {

class tnlDefaultBuildConfigTag{};

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
 * All index types are enabled by default.
 */
template< typename ConfigTag, typename Index > struct tnlConfigTagIndex{ enum { enabled = true }; };

/****
 * The mesh type will be resolved by the tnlSolver by default.
 */
template< typename ConfigTag > struct tnlConfigTagMeshResolve{ enum { enabled = true }; };

/****
 * 1, 2, and 3 dimensions are enabled by default
 */
template< typename ConfigTag, int Dimensions > struct tnlConfigTagDimensions{ enum { enabled = ( Dimensions > 0 && Dimensions <= 3 ) }; };

/****
 * Up to the exceptions enlisted below, all mesh types are disabled by default.
 */
template< typename ConfigTag, typename MeshType > struct tnlConfigTagMesh{ enum { enabled = false }; };

/****
 * Use of tnlGrid is enabled for allowed dimensions and Real, Device and Index types.
 */
template< typename ConfigTag, int Dimensions, typename Real, typename Device, typename Index >
   struct tnlConfigTagMesh< ConfigTag, tnlGrid< Dimensions, Real, Device, Index > >
      { enum { enabled = tnlConfigTagDimensions< ConfigTag, Dimensions >::enabled  &&
                         tnlConfigTagReal< ConfigTag, Real >::enabled &&
                         tnlConfigTagDevice< ConfigTag, Device >::enabled &&
                         tnlConfigTagIndex< ConfigTag, Index >::enabled }; };

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
class tnlExplicitEulerSolverTag
{
public:
    template< typename Problem >
    using Template = tnlEulerSolver< Problem >;
};

class tnlExplicitMersonSolverTag
{
public:
    template< typename Problem >
    using Template = tnlMersonSolver< Problem >;
};

template< typename ConfigTag, typename ExplicitSolver > struct tnlConfigTagExplicitSolver{ enum { enabled = true }; };

/****
 * All semi-implicit solvers are enabled by default
 */
class  tnlSemiImplicitSORSolverTag
{
public:
    template< typename Matrix, typename Preconditioner = tnlDummyPreconditioner< typename Matrix::RealType,
                                                                                 typename Matrix::DeviceType,
                                                                                 typename Matrix::IndexType > >
    using Template = tnlSORSolver< Matrix, Preconditioner >;
};

class  tnlSemiImplicitCGSolverTag
{
public:
    template< typename Matrix, typename Preconditioner = tnlDummyPreconditioner< typename Matrix::RealType,
                                                                                 typename Matrix::DeviceType,
                                                                                 typename Matrix::IndexType > >
    using Template = tnlCGSolver< Matrix, Preconditioner >;
};

class  tnlSemiImplicitBICGStabSolverTag
{
public:
    template< typename Matrix, typename Preconditioner = tnlDummyPreconditioner< typename Matrix::RealType,
                                                                                 typename Matrix::DeviceType,
                                                                                 typename Matrix::IndexType > >
    using Template = tnlBICGStabSolver< Matrix, Preconditioner >;
};

class  tnlSemiImplicitGMRESSolverTag
{
public:
    template< typename Matrix, typename Preconditioner = tnlDummyPreconditioner< typename Matrix::RealType,
                                                                                 typename Matrix::DeviceType,
                                                                                 typename Matrix::IndexType > >
    using Template = tnlGMRESSolver< Matrix, Preconditioner >;
};

class  tnlSemiImplicitTFQMRSolverTag
{
public:
    template< typename Matrix, typename Preconditioner = tnlDummyPreconditioner< typename Matrix::RealType,
                                                                                 typename Matrix::DeviceType,
                                                                                 typename Matrix::IndexType > >
    using Template = tnlTFQMRSolver< Matrix, Preconditioner >;
};

#ifdef HAVE_UMFPACK
class  tnlSemiImplicitUmfpackSolverTag
{
public:
    template< typename Matrix, typename Preconditioner = tnlDummyPreconditioner< typename Matrix::RealType,
                                                                                 typename Matrix::DeviceType,
                                                                                 typename Matrix::IndexType > >
    using Template = tnlUmfpackWrapper< Matrix, Preconditioner >;
};
#endif

template< typename ConfigTag, typename SemiImplicitSolver > struct tnlConfigTagSemiImplicitSolver{ enum { enabled = true }; };

} // namespace TNL
