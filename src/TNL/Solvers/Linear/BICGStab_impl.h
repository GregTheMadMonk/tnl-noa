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

template< typename Matrix,
          typename Preconditioner >
BICGStab< Matrix, Preconditioner > :: BICGStab()
: exact_residue( false )
{
   /****
    * Clearing the shared pointer means that there is no
    * preconditioner set.
    */
   this->preconditioner.clear();   
}

template< typename Matrix,
          typename Preconditioner >
String BICGStab< Matrix, Preconditioner > :: getType() const
{
   return String( "BICGStab< " ) +
          this->matrix -> getType() + ", " +
          this->preconditioner -> getType() + " >";
}

template< typename Matrix,
          typename Preconditioner >
void
BICGStab< Matrix, Preconditioner >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   //IterativeSolver< RealType, IndexType >::configSetup( config, prefix );
   config.addEntry< bool >( prefix + "bicgstab-exact-residue", "Whether the BiCGstab should compute the exact residue in each step (true) or to use a cheap approximation (false).", false );
}

template< typename Matrix,
          typename Preconditioner >
bool
BICGStab< Matrix, Preconditioner >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   exact_residue = parameters.getParameter< bool >( "bicgstab-exact-residue" );
   return IterativeSolver< RealType, IndexType >::setup( parameters, prefix );
}

template< typename Matrix,
          typename Preconditioner >
bool BICGStab< Matrix, Preconditioner >::solve( const ConstVectorViewType& b, VectorViewType& x )
{
   this->setSize( this->matrix->getRows() );

   RealType alpha, beta, omega, aux, rho, rho_old, b_norm;

   if( this->preconditioner ) {
      this->preconditioner->solve( b, M_tmp );
      b_norm = M_tmp.lpNorm( ( RealType ) 2.0 );

      this->matrix->vectorProduct( x, M_tmp );
      M_tmp.addVector( b, 1.0, -1.0 );
      this->preconditioner->solve( M_tmp, r );
   }
   else {
      b_norm = b.lpNorm( 2.0 );
      this->matrix->vectorProduct( x, r );
      r.addVector( b, 1.0, -1.0 );
   }

   p = r_ast = r;
   s.setValue( 0.0 );
   rho = r.scalarProduct( r_ast );

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
      aux = Ap.scalarProduct( r_ast );
      alpha = rho / aux;

      /****
       * s_j = r_j - alpha_j * A p_j
       */
      s.addVectors( r, 1.0, Ap, -alpha, 0.0 );

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
      aux = As.lpNorm( 2.0 );
      omega = As.scalarProduct( s ) / ( aux * aux );

      /****
       * x_{j+1} = x_j + alpha_j * p_j + omega_j * s_j
       */
      x.addVectors( p, alpha, s, omega );

      /****
       * r_{j+1} = s_j - omega_j * A s_j
       */
      r.addVectors( s, 1.0, As, -omega, 0.0 );

      /****
       * beta = alpha_j / omega_j * ( r_{j+1}, r^ast_0 ) / ( r_j, r^ast_0 )
       */
      rho_old = rho;
      rho = r.scalarProduct( r_ast );
      beta = ( rho / rho_old ) * ( alpha / omega );

      /****
       * p_{j+1} = r_{j+1} + beta_j * ( p_j - omega_j * A p_j )
       */
      p.addVectors( r, 1.0, Ap, -beta * omega, beta );

      if( exact_residue ) {
         /****
          * Compute the exact preconditioned residue into the 's' vector.
          */
         if( this->preconditioner ) {
            this->matrix->vectorProduct( x, M_tmp );
            M_tmp.addVector( b, 1.0, -1.0 );
            this->preconditioner->solve( M_tmp, s );
         }
         else {
            this->matrix->vectorProduct( x, s );
            s.addVector( b, 1.0, -1.0 );
         }
         const RealType residue = s.lpNorm( 2.0 );
         this->setResidue( residue / b_norm );
      }
      else {
         /****
          * Use the "orthogonal residue vector" for stopping.
          */
         const RealType residue = r.lpNorm( 2.0 );
         this->setResidue( residue / b_norm );
      }
   }

   this->refreshSolverMonitor( true );
   return this->checkConvergence();
}

template< typename Matrix,
          typename Preconditioner >
void BICGStab< Matrix, Preconditioner > :: setSize( IndexType size )
{
   r.setSize( size );
   r_ast.setSize( size );
   p.setSize( size );
   s.setSize( size );
   Ap.setSize( size );
   As.setSize( size );
   M_tmp.setSize( size );
}

} // namespace Linear
} // namespace Solvers
} // namespace TNL
