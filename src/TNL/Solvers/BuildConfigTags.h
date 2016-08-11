/***************************************************************************
                          BuildConfigTags.h  -  description
                             -------------------
    begin                : Jul 7, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Solvers/ODE/Merson.h>
#include <TNL/Solvers/ODE/Euler.h>
#include <TNL/Solvers/Linear/SOR.h>
#include <TNL/Solvers/Linear/CG.h>
#include <TNL/Solvers/Linear/BICGStab.h>
#include <TNL/Solvers/Linear/GMRES.h>
#include <TNL/Solvers/Linear/TFQMR.h>
#include <TNL/Solvers/Linear/UmfpackWrapper.h>
#include <TNL/Solvers/Linear/Preconditioners/Dummy.h>

namespace TNL {
namespace Solvers {   

class tnlDefaultBuildConfigTag{};

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
 */
template< typename ConfigTag > struct ConfigTagMeshResolve{ enum { enabled = true }; };

/****
 * 1, 2, and 3 dimensions are enabled by default
 */
template< typename ConfigTag, int Dimensions > struct ConfigTagDimensions{ enum { enabled = ( Dimensions > 0 && Dimensions <= 3 ) }; };

/****
 * Up to the exceptions enlisted below, all mesh types are disabled by default.
 */
template< typename ConfigTag, typename MeshType > struct ConfigTagMesh{ enum { enabled = false }; };

/****
 * Use of Grid is enabled for allowed dimensions and Real, Device and Index types.
 */
template< typename ConfigTag, int Dimensions, typename Real, typename Device, typename Index >
   struct ConfigTagMesh< ConfigTag, Meshes::Grid< Dimensions, Real, Device, Index > >
      { enum { enabled = ConfigTagDimensions< ConfigTag, Dimensions >::enabled  &&
                         ConfigTagReal< ConfigTag, Real >::enabled &&
                         ConfigTagDevice< ConfigTag, Device >::enabled &&
                         ConfigTagIndex< ConfigTag, Index >::enabled }; };

/****
 * All time discretisations (explicit, semi-impicit and implicit ) are
 * enabled by default.
 */
class tnlExplicitTimeDiscretisationTag{};
class tnlSemiImplicitTimeDiscretisationTag{};
class tnlImplicitTimeDiscretisationTag{};

template< typename ConfigTag, typename TimeDiscretisation > struct ConfigTagTimeDiscretisation{ enum { enabled = true }; };

/****
 * All explicit solvers are enabled by default
 */
class tnlExplicitEulerSolverTag
{
public:
    template< typename Problem >
    using Template = ODE::Euler< Problem >;
};

class tnlExplicitMersonSolverTag
{
public:
    template< typename Problem >
    using Template = ODE::Merson< Problem >;
};

template< typename ConfigTag, typename ExplicitSolver > struct ConfigTagExplicitSolver{ enum { enabled = true }; };

/****
 * All semi-implicit solvers are enabled by default
 */
class  tnlSemiImplicitSORSolverTag
{
public:
    template< typename Matrix, typename Preconditioner = Linear::Preconditioners::Dummy< typename Matrix::RealType,
                                                                                 typename Matrix::DeviceType,
                                                                                 typename Matrix::IndexType > >
    using Template = Linear::SOR< Matrix, Preconditioner >;
};

class  tnlSemiImplicitCGSolverTag
{
public:
    template< typename Matrix, typename Preconditioner = Linear::Preconditioners::Dummy< typename Matrix::RealType,
                                                                                 typename Matrix::DeviceType,
                                                                                 typename Matrix::IndexType > >
    using Template = Linear::CG< Matrix, Preconditioner >;
};

class  tnlSemiImplicitBICGStabSolverTag
{
public:
    template< typename Matrix, typename Preconditioner = Linear::Preconditioners::Dummy< typename Matrix::RealType,
                                                                                 typename Matrix::DeviceType,
                                                                                 typename Matrix::IndexType > >
    using Template = Linear::BICGStab< Matrix, Preconditioner >;
};

class  tnlSemiImplicitGMRESSolverTag
{
public:
    template< typename Matrix, typename Preconditioner = Linear::Preconditioners::Dummy< typename Matrix::RealType,
                                                                                 typename Matrix::DeviceType,
                                                                                 typename Matrix::IndexType > >
    using Template = Linear::GMRES< Matrix, Preconditioner >;
};

class  tnlSemiImplicitTFQMRSolverTag
{
public:
    template< typename Matrix, typename Preconditioner = Linear::Preconditioners::Dummy< typename Matrix::RealType,
                                                                                 typename Matrix::DeviceType,
                                                                                 typename Matrix::IndexType > >
    using Template = Linear::TFQMR< Matrix, Preconditioner >;
};

#ifdef HAVE_UMFPACK
class  tnlSemiImplicitUmfpackSolverTag
{
public:
    template< typename Matrix, typename Preconditioner = Linear::Preconditioners::Dummy< typename Matrix::RealType,
                                                                                 typename Matrix::DeviceType,
                                                                                 typename Matrix::IndexType > >
    using Template = Linear::UmfpackWrapper< Matrix, Preconditioner >;
};
#endif

template< typename ConfigTag, typename SemiImplicitSolver > struct ConfigTagSemiImplicitSolver{ enum { enabled = true }; };

} // namespace Solvers
} // namespace TNL
