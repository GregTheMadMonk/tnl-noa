/***************************************************************************
                          Jacobi.h  -  description
                             -------------------
    begin                : Jul 30, 2007
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Solvers/Linear/LinearSolver.h>
#include <TNL/Solvers/Linear/Utils/LinearResidueGetter.h>

namespace TNL {
   namespace Solvers {
      namespace Linear {

/**
 * \brief Iterative solver of linear systems based on the Jacobi method.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Jacobi_method) for more details.
 *
 * \tparam Matrix is type of matrix describing the linear system.
 */
template< typename Matrix >
class Jacobi
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
       * \brief This is method defines configuration entries for setup of the linear iterative solver.
       *
       * In addition to config entries defined by \ref IterativeSolver::configSetup, this method
       * defines the following:
       *
       * \e jacobi-omega - relaxation parameter of the weighted/damped Jacobi method.
       *
       * \param config contains description of configuration parameters.
       * \param prefix is a prefix of particular configuration entries.
       */
      static void configSetup( Config::ConfigDescription& config, const String& prefix = "" );

      /**
       * \brief Method for setup of the linear iterative solver based on configuration parameters.
       *
       * \param parameters contains values of the define configuration entries.
       * \param prefix is a prefix of particular configuration entries.
       */
      bool setup( const Config::ParameterContainer& parameters, const String& prefix = "" ) override;

      /**
       * \brief Setter of the relaxation parameter.
       *
       * \param omega the relaxation parameter. It is 1 by default.
       */
      void setOmega( RealType omega );

      /**
       * \brief Getter of the relaxation parameter.
       *
       * \return value of the relaxation parameter.
       */
      RealType getOmega() const;

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
      RealType omega = 0.0;
};

      } // namespace Linear
   } // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/Jacobi.hpp>