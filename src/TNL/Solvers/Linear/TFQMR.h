/***************************************************************************
                          TFQMR.h  -  description
                             -------------------
    begin                : Dec 8, 2012
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Solvers/Linear/LinearSolver.h>

namespace TNL {
   namespace Solvers {
      namespace Linear {

/**
 * \brief Iterative solver of linear systems based on the Transpose-free quasi-minimal residual (TFQMR) method.
 *
 * See (Wikipedia)[https://second.wiki/wiki/algoritmo_tfqmr] for more details.
 *
 * \tparam Matrix is type of matrix describing the linear system.
 */
template< typename Matrix >
class TFQMR
: public LinearSolver< Matrix >
{
   using Base = LinearSolver< Matrix >;

   public:

      /**
       * \brief Floating point type used for computations.
       */
      using RealType = typename Base::RealType;

      /**
       * \brief Device where the solver will run on and auxillary data will alloacted on.
       */
      using DeviceType = typename Base::DeviceType;

      /**
       * \brief Type for indexing.
       */
      using IndexType = typename Base::IndexType;

      /**
       * \brief Type for vector view.
       */
      using VectorViewType = typename Base::VectorViewType;

      /**
       * \brief Type for constant vector view.
       */
      using ConstVectorViewType = typename Base::ConstVectorViewType;

      /**
       * \brief Method for solving of a linear system.
       *
       * See \ref LinearSolver::solve for more details.
       *
       * \param b vector with the right-hand side of the linear system.
       * \param x vector for the solution of the linear system.
       * \return true if the solver converged.
       * \return false if the solver did not converge.
       */
      bool solve( ConstVectorViewType b, VectorViewType x ) override;

   protected:
      void setSize( const VectorViewType& x );

      typename Traits< Matrix >::VectorType d, r, w, u, v, r_ast, Au, M_tmp;
};

      } // namespace Linear
   } // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/TFQMR.hpp>
