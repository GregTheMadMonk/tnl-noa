/***************************************************************************
                          BICGStab_impl.h  -  description
                             -------------------
    begin                : 2007/07/31
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <cmath>

#include "BICGStab.h"

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
String BICGStab< Matrix > :: getType() const
{
   return String( "BICGStab< " ) +
          this->matrix -> getType() + ", " +
          this->preconditioner -> getType() + " >";
}

template< typename Matrix >
void
BICGStab< Matrix >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   config.addEntry< bool >( prefix + "bicgstab-exact-residue", "Whether the BiCGstab should compute the exact residue in each step (true) or to use a cheap approximation (false).", false );
}

template< typename Matrix >
bool
BICGStab< Matrix >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   exact_residue = parameters.getParameter< bool >( "bicgstab-exact-residue" );
   return LinearSolver< Matrix >::setup( parameters, prefix );
}

template< typename Matrix >
bool BICGStab< Matrix >::solve( ConstVectorViewType b, VectorViewType x )
{
   this->setSize( x );

   RealType alpha, beta, omega, aux, rho, rho_old, b_norm;

   if( this->preconditioner ) {
      this->preconditioner->solve( b, M_tmp );
      b_norm = lpNorm( M_tmp, 2.0 );

      this->matrix->vectorProduct( x, M_tmp );
      M_tmp = b - M_tmp;
      this->preconditioner->solve( M_tmp, r );
   }
   else {
      b_norm = lpNorm( b, 2.0 );
      this->matrix->vectorProduct( x, r );
      r = b - r;
   }

   p = r_ast = r;
   s.setValue( 0.0 );
   rho = (r, r_ast);

   if( b_norm == 0.0 )
       b_norm = 1.0;

   this->resetIterations();
   this->setResidue( std::sqrt( rho ) / b_norm );

   while( this->nextIteration() )
   {
      /****
       * alpha_j = ( r_j, r^ast_0 ) / ( A * p_j, r^ast_0 )
       */
      if( this->preconditioner ) {
         this->matrix->vectorProduct( p, M_tmp );
         this->preconditioner->solve( M_tmp, Ap );
      }
      else {
         this->matrix->vectorProduct( p, Ap );
      }
      aux = (Ap, r_ast);
      alpha = rho / aux;

      /****
       * s_j = r_j - alpha_j * A p_j
       */
      s = r - alpha * Ap;

      /****
       * omega_j = ( A s_j, s_j ) / ( A s_j, A s_j )
       */
      if( this->preconditioner ) {
         this->matrix->vectorProduct( s, M_tmp );
         this->preconditioner->solve( M_tmp, As );
      }
      else {
         this->matrix->vectorProduct( s, As );
      }
      aux = lpNorm( As, 2.0 );
      omega = (As, s) / (aux * aux);

      /****
       * x_{j+1} = x_j + alpha_j * p_j + omega_j * s_j
       */
      x += alpha * p + omega * s;

      /****
       * r_{j+1} = s_j - omega_j * A s_j
       */
      r = s - omega * As;

      /****
       * beta = alpha_j / omega_j * ( r_{j+1}, r^ast_0 ) / ( r_j, r^ast_0 )
       */
      rho_old = rho;
      rho = (r, r_ast);
      beta = (rho / rho_old) * (alpha / omega);

      /****
       * p_{j+1} = r_{j+1} + beta_j * ( p_j - omega_j * A p_j )
       */
      p = r + beta * p - (beta * omega) * Ap;

      if( exact_residue ) {
         /****
          * Compute the exact preconditioned residue into the 's' vector.
          */
         if( this->preconditioner ) {
            this->matrix->vectorProduct( x, M_tmp );
            M_tmp = b - M_tmp;
            this->preconditioner->solve( M_tmp, s );
         }
         else {
            this->matrix->vectorProduct( x, s );
            s = b - s;
         }
         const RealType residue = lpNorm( s, 2.0 );
         this->setResidue( residue / b_norm );
      }
      else {
         /****
          * Use the "orthogonal residue vector" for stopping.
          */
         const RealType residue = lpNorm( r, 2.0 );
         this->setResidue( residue / b_norm );
      }
   }

   this->refreshSolverMonitor( true );
   return this->checkConvergence();
}

template< typename Matrix >
void BICGStab< Matrix > :: setSize( const VectorViewType& x )
{
   r.setLike( x );
   r_ast.setLike( x );
   p.setLike( x );
   s.setLike( x );
   Ap.setLike( x );
   As.setLike( x );
   M_tmp.setLike( x );
}

} // namespace Linear
} // namespace Solvers
} // namespace TNL
