/***************************************************************************
                          SOR.h  -  description
                             -------------------
    begin                : 2007/07/30
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
 * \brief Iterative solver of linear systems based on the Successive-overrelaxation (SOR) or Gauss-Seidel method.
 *
 * See (Wikipedia)[https://en.wikipedia.org/wiki/Successive_over-relaxation] for more details.
 *
 * \tparam Matrix is type of matrix describing the linear system.
 */
template< typename Matrix >
class SOR
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
       * \e sor-omega - relaxation parameter of the weighted/damped Jacobi method.
       *
       * \param config contains description of configuration parameters.
       * \param prefix is a prefix of particular configuration entries.
       */
      static void configSetup( Config::ConfigDescription& config,
                              const String& prefix = "" );

      /**
       * \brief Method for setup of the linear iterative solver based on configuration parameters.
       *
       * \param parameters contains values of the define configuration entries.
       * \param prefix is a prefix of particular configuration entries.
       */
      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" ) override;

      /**
       * \brief Setter of the relaxation parameter.
       *
       * \param omega the relaxation parameter. It is 1 by default.
       */
      void setOmega( const RealType& omega );

      /**
       * \brief Getter of the relaxation parameter.
       *
       * \return value of the relaxation parameter.
       */
      const RealType& getOmega() const;

      /**
       * \brief Set the period for a recomputation of the residue.
       *
       * \param period number of iterations between subsequent recomputations of the residue.
       */
      void setResiduePeriod( IndexType period );

      /**
       * \brief Get the period for a recomputation of the residue.
       *
       * \return number of iterations between subsequent recomputations of the residue.
       */
      IndexType getResiduePerid() const;

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
      using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

      RealType omega = 1.0;

      IndexType residuePeriod = 4;

      VectorType diagonal;

   public: // because nvcc does not accept lambda functions within private or protected methods
      void performIteration( const ConstVectorViewType& b,
                             const ConstVectorViewType& diagonalView,
                             VectorViewType& x ) const;

};

      } // namespace Linear
   } // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/SOR.hpp>
