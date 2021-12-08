/***************************************************************************
                          BICGStabL.h  -  description
                             -------------------
    begin                : Jul 4, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/*
 * BICGStabL implements an iterative solver for non-symmetric linear systems,
 * using the BiCGstab(l) algorithm described in [1] and [2]. It is a
 * generalization of the stabilized biconjugate-gradient (BiCGstab) algorithm
 * proposed by van der Vorst [3]. BiCGstab(1) is equivalent to BiCGstab, and
 * BiCGstab(2) is a slightly more efficient version of the BiCGstab2 algorithm
 * by Gutknecht [4], while BiCGstab(l>2) is a further generalization.
 *
 * This code was implemented by: Jakub Klinkovsky <klinkjak@fjfi.cvut.cz>
 *
 * [1] Gerard L. G. Sleijpen and Diederik R. Fokkema, "BiCGstab(l) for linear
 *     equations involving unsymmetric matrices with complex spectrum",
 *     Electronic Trans. on Numerical Analysis 1, 11-32 (1993).
 * [2] Gerard L. G. Sleijpen, Henk A. van der Vorst, and Diederik R. Fokkema,
 *     "BiCGstab(l) and other Hybrid Bi-CG Methods", Numerical Algorithms 7,
 *     75-109 (1994).
 * [3] Henk A. van der Vorst, "Bi-CGSTAB: A fast and smoothly converging variant
 *     of Bi-CG for the solution of nonsymmetric linear systems, SIAM Journal on
 *     scientific and Statistical Computing 13.2, 631-644 (1992).
 * [4] Martin H. Gutknecht, "Variants of BiCGStab for matrices with complex
 *     spectrum", IPS Research Report No. 91-14 (1991).
 *
 * TODO: further variations to explore:
 *
 * [5] Gerard L. G. Sleijpen and Henk A. van der Vorst, "Reliable updated
 *     residuals in hybrid Bi-CG methods", Computing 56 (2), 141-163 (1996).
 * [6] Gerard L. G. Sleijpen and Henk A. van der Vorst, "Maintaining convergence
 *     properties of BiCGstab methods in finite precision arithmetic", Numerical
 *     Algorithms 10, 203-223 (1995).
 */

#pragma once

#include "LinearSolver.h"

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
class BICGStabL
: public LinearSolver< Matrix >
{
   using Base = LinearSolver< Matrix >;

   // compatibility shortcut
   using Traits = Linear::Traits< Matrix >;

public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   // distributed vectors/views
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;
   using VectorType = typename Traits::VectorType;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" ) override;

   bool solve( ConstVectorViewType b, VectorViewType x ) override;

protected:
   using DeviceVector = Containers::Vector< RealType, DeviceType, IndexType >;
   using HostVector = Containers::Vector< RealType, Devices::Host, IndexType >;

   void compute_residue( VectorViewType r, ConstVectorViewType x, ConstVectorViewType b );

   void preconditioned_matvec( ConstVectorViewType src, VectorViewType dst );

   void setSize( const VectorViewType& x );

   int ell = 1;

   bool exact_residue = false;

   // matrices (in column-major format)
   DeviceVector R, U;
   // single vectors (distributed)
   VectorType r_ast, M_tmp, res_tmp;
   // host-only storage
   HostVector T, sigma, g_0, g_1, g_2;

   IndexType size = 0;
   IndexType ldSize = 0;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include "BICGStabL.hpp"
