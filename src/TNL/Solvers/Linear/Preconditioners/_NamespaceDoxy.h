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
      namespace Linear {
         /**
          * \brief Namespace for preconditioners of linear system solvers.
          *
          * This namespace contains the following preconditioners for iterative solvers linear systems.
          *
          * 1. Diagonal - is diagonal or Jacobi preconditioner - see[Netlib](http://netlib.org/linalg/html_templates/node55.html)
          * 2. ILU0 - is Incomplete LU preconditioner with the same sparsity pattern as the original matrix - see [Wikipedia](https://en.wikipedia.org/wiki/Incomplete_LU_factorization)
          * 3. ILUT - is Incomplete LU preconiditoner with thresholding - see [paper by Y. Saad](https://www-users.cse.umn.edu/~saad/PDF/umsi-92-38.pdf)
          */
         namespace Preconditioners {
         } // namespace Preconditioners
      } // namespace Linear
   } // namespace Solvers
} // namespace TNL
