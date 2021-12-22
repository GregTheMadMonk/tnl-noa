/***************************************************************************
                          _NamespaceDoxy.h  -  description
                             -------------------
    begin                : Dec 21, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
   namespace Solvers {

      /**
       * \brief Namespace for linear system solvers.
       *
       * This namespace contains the following algorithms and methods for solution of linear systems.
       *
       * # Direct methods
       *
       * # Iterative methods
       *
       * ## Stationary methods
       *    1. Jacobi method - \ref TNL::Solvers::Linear::Jacobi
       *    2. Successive-overrelaxation (SOR) method - \ref TNL::Solvers::Linear::SOR
       *
       * ## Krylov subspace methods
       *    1. Conjugate gradient (CG) method - \ref TNL::Solvers::Linear::CG
       *    2. Biconjugate gradient stabilised (BICGStab) method  - \ref TNL::Solvers::Linear::BICGStab
       *    3. Generalized minimal residual (GMRES) method - \ref TNL::Solvers::Linear::GMRES
       *    4. Transpose-free quasi-minimal residual (TFQMR) method - \ref TNL::Solvers::Linear::TFQMR
       */
      namespace Linear {
      } // namespace Linear
   } // namespace Solvers
} // namespace TNL
