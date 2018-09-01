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
#include <TNL/Solvers/Linear/SOR.h>
#include <TNL/Solvers/Linear/CG.h>
#include <TNL/Solvers/Linear/BICGStab.h>
#include <TNL/Solvers/Linear/BICGStabL.h>
#include <TNL/Solvers/Linear/CWYGMRES.h>
#include <TNL/Solvers/Linear/GMRES.h>
#include <TNL/Solvers/Linear/TFQMR.h>
#include <TNL/Solvers/Linear/UmfpackWrapper.h>
#include <TNL/Solvers/Linear/Preconditioners/Dummy.h>

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

#ifndef HAVE_MIC
template< typename ConfigTag > struct ConfigTagDevice< ConfigTag, Devices::MIC >{ enum { enabled = false }; };
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
 * (The detailed mesh configuration is in TNL/Meshes/BuildConfigTags.h)
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
    template< typename Problem >
    using Template = ODE::Euler< Problem >;
};

class ExplicitMersonSolverTag
{
public:
    template< typename Problem >
    using Template = ODE::Merson< Problem >;
};

template< typename ConfigTag, typename ExplicitSolver > struct ConfigTagExplicitSolver{ enum { enabled = true }; };

/****
 * All semi-implicit solvers are enabled by default
 */
class  SemiImplicitSORSolverTag
{
public:
    template< typename Matrix,
              typename Preconditioner = Linear::Preconditioners::Dummy< typename Matrix::RealType,
                                                                        typename Matrix::DeviceType,
                                                                        typename Matrix::IndexType > >
    using Template = Linear::SOR< Matrix, Preconditioner >;
};

class  SemiImplicitCGSolverTag
{
public:
    template< typename Matrix,
              typename Preconditioner = Linear::Preconditioners::Dummy< typename Matrix::RealType,
                                                                        typename Matrix::DeviceType,
                                                                        typename Matrix::IndexType > >
    using Template = Linear::CG< Matrix, Preconditioner >;
};

class  SemiImplicitBICGStabSolverTag
{
public:
    template< typename Matrix,
              typename Preconditioner = Linear::Preconditioners::Dummy< typename Matrix::RealType,
                                                                        typename Matrix::DeviceType,
                                                                        typename Matrix::IndexType > >
    using Template = Linear::BICGStab< Matrix, Preconditioner >;
};

class  SemiImplicitBICGStabLSolverTag
{
public:
    template< typename Matrix,
              typename Preconditioner = Linear::Preconditioners::Dummy< typename Matrix::RealType,
                                                                        typename Matrix::DeviceType,
                                                                        typename Matrix::IndexType > >
    using Template = Linear::BICGStabL< Matrix, Preconditioner >;
};

class  SemiImplicitCWYGMRESSolverTag
{
public:
    template< typename Matrix,
              typename Preconditioner = Linear::Preconditioners::Dummy< typename Matrix::RealType,
                                                                        typename Matrix::DeviceType,
                                                                        typename Matrix::IndexType > >
    using Template = Linear::CWYGMRES< Matrix, Preconditioner >;
};

class  SemiImplicitGMRESSolverTag
{
public:
    template< typename Matrix,
              typename Preconditioner = Linear::Preconditioners::Dummy< typename Matrix::RealType,
                                                                        typename Matrix::DeviceType,
                                                                        typename Matrix::IndexType > >
    using Template = Linear::GMRES< Matrix, Preconditioner >;
};

class  SemiImplicitTFQMRSolverTag
{
public:
    template< typename Matrix,
              typename Preconditioner = Linear::Preconditioners::Dummy< typename Matrix::RealType,
                                                                        typename Matrix::DeviceType,
                                                                        typename Matrix::IndexType > >
    using Template = Linear::TFQMR< Matrix, Preconditioner >;
};

#ifdef HAVE_UMFPACK
class  SemiImplicitUmfpackSolverTag
{
public:
    template< typename Matrix,
              typename Preconditioner = Linear::Preconditioners::Dummy< typename Matrix::RealType,
                                                                        typename Matrix::DeviceType,
                                                                        typename Matrix::IndexType > >
    using Template = Linear::UmfpackWrapper< Matrix, Preconditioner >;
};
#endif

template< typename ConfigTag, typename SemiImplicitSolver > struct ConfigTagSemiImplicitSolver{ enum { enabled = true }; };

} // namespace Solvers
} // namespace TNL
