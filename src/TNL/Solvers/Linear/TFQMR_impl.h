/***************************************************************************
                          TFQMR_impl.h  -  description
                             -------------------
    begin                : Dec 8, 2012
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <cmath>

#include "TFQMR.h"

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
String TFQMR< Matrix > :: getType() const
{
   return String( "TFQMR< " ) +
          this->matrix -> getType() + ", " +
          this->preconditioner -> getType() + " >";
}

template< typename Matrix >
bool TFQMR< Matrix >::solve( ConstVectorViewType b, VectorViewType x )
{
   this->setSize( x );

   RealType tau, theta, eta, rho, alpha, b_norm, w_norm;

   if( this->preconditioner ) {
      this->preconditioner->solve( b, M_tmp );
      b_norm = lpNorm( M_tmp, 2.0 );

      this->matrix->vectorProduct( x, M_tmp );
      M_tmp.addVector( b, 1.0, -1.0 );
      this->preconditioner->solve( M_tmp, r );
   }
   else {
      b_norm = lpNorm( b, 2.0 );
      this->matrix->vectorProduct( x, r );
      r.addVector( b, 1.0, -1.0 );
   }
   w = u = r;
   if( this->preconditioner ) {
      this->matrix->vectorProduct( u, M_tmp );
      this->preconditioner->solve( M_tmp, Au );
   }
   else {
      this->matrix->vectorProduct( u, Au );
   }
   v = Au;
   d.setValue( 0.0 );
   tau = lpNorm( r, 2.0 );
   theta = eta = 0.0;
   r_ast = r;
   rho = r_ast.scalarProduct( r );
   // only to avoid compiler warning; alpha is initialized inside the loop
   alpha = 0.0;

   if( b_norm == 0.0 )
       b_norm = 1.0;

   this->resetIterations();
   this->setResidue( tau / b_norm );

   while( this->nextIteration() )
   {
      const IndexType iter = this->getIterations();

      if( iter % 2 == 1 ) {
         alpha = rho / v. scalarProduct( this->r_ast );
      }
      else {
         // not necessary in odd iter since the previous iteration
         // already computed v_{m+1} = A*u_{m+1}
         if( this->preconditioner ) {
            this->matrix->vectorProduct( u, M_tmp );
            this->preconditioner->solve( M_tmp, Au );
         }
         else {
            this->matrix->vectorProduct( u, Au );
         }
      }
      w.addVector( Au, -alpha );
      d.addVector( u, 1.0, theta * theta * eta / alpha );
      w_norm = lpNorm( w, 2.0 );
      theta = w_norm / tau;
      const RealType c = 1.0 / std::sqrt( 1.0 + theta * theta );
      tau = tau * theta * c;
      eta = c * c  * alpha;
      x.addVector( d, eta );

      this->setResidue( tau * std::sqrt(iter+1) / b_norm );
      if( iter > this->getMinIterations() && this->getResidue() < this->getConvergenceResidue() ) {
          break;
      }

      if( iter % 2 == 0 ) {
         const RealType rho_new  = w.scalarProduct( this->r_ast );
         const RealType beta = rho_new / rho;
         rho = rho_new;

         u.addVector( w, 1.0, beta );
         v.addVector( Au, beta, beta * beta );
         if( this->preconditioner ) {
            this->matrix->vectorProduct( u, M_tmp );
            this->preconditioner->solve( M_tmp, Au );
         }
         else {
            this->matrix->vectorProduct( u, Au );
         }
         v.addVector( Au, 1.0 );
      }
      else {
         u.addVector( v, -alpha );
      }

      this->refreshSolverMonitor();
   }

   this->refreshSolverMonitor( true );
   return this->checkConvergence();
}

template< typename Matrix >
void TFQMR< Matrix > :: setSize( const VectorViewType& x )
{
   d.setLike( x );
   r.setLike( x );
   w.setLike( x );
   u.setLike( x );
   v.setLike( x );
   r_ast.setLike( x );
   Au.setLike( x );
   M_tmp.setLike( x );
}

} // namespace Linear
} // namespace Solvers
} // namespace TNL
