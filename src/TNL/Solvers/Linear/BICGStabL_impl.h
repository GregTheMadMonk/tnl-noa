/***************************************************************************
                          BICGStabL.h  -  description
                             -------------------
    begin                : Jul 4, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include "BICGStabL.h"

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix,
          typename Preconditioner >
BICGStabL< Matrix, Preconditioner >::BICGStabL()
{
   /****
    * Clearing the shared pointer means that there is no
    * preconditioner set.
    */
   this->preconditioner.clear();
}

template< typename Matrix,
          typename Preconditioner >
String
BICGStabL< Matrix, Preconditioner >::getType() const
{
   return String( "BICGStabL< " ) +
          this->matrix -> getType() + ", " +
          this->preconditioner -> getType() + " >";
}

template< typename Matrix,
          typename Preconditioner >
void
BICGStabL< Matrix, Preconditioner >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   //IterativeSolver< RealType, IndexType >::configSetup( config, prefix );
   config.addEntry< int >( prefix + "bicgstab-ell", "Number of Bi-CG iterations before the MR part starts.", 1 );
   config.addEntry< bool >( prefix + "bicgstab-exact-residue", "Whether the BiCGstab should compute the exact residue in each step (true) or to use a cheap approximation (false).", false );
}

template< typename Matrix,
          typename Preconditioner >
bool
BICGStabL< Matrix, Preconditioner >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   ell = parameters.getParameter< int >( "bicgstab-ell" );
   exact_residue = parameters.getParameter< bool >( "bicgstab-exact-residue" );
   return IterativeSolver< RealType, IndexType >::setup( parameters, prefix );
}

template< typename Matrix,
          typename Preconditioner >
void
BICGStabL< Matrix, Preconditioner >::setMatrix( const MatrixPointer& matrix )
{
   this->matrix = matrix;
}

template< typename Matrix,
          typename Preconditioner >
void
BICGStabL< Matrix, Preconditioner >::setPreconditioner( const PreconditionerPointer& preconditioner )
{
   this->preconditioner = preconditioner;
}

template< typename Matrix,
          typename Preconditioner >
   template< typename Vector, typename ResidueGetter >
bool
BICGStabL< Matrix, Preconditioner >::solve( const Vector& b, Vector& x )
{
   this->setSize( matrix->getRows() );

   RealType alpha, beta, gamma, rho_0, rho_1, omega, b_norm;
   DeviceVector r_0, r_j, r_i, u_0, Au, u;
   r_0.bind( R.getData(), size );
   u_0.bind( U.getData(), size );

   auto matvec = [this]( const DeviceVector& src, DeviceVector& dst )
   {
      if( preconditioner ) {
         matrix->vectorProduct( src, M_tmp );
         preconditioner->solve( M_tmp, dst );
      }
      else {
         matrix->vectorProduct( src, dst );
      }
   };

   if( preconditioner ) {
      preconditioner->solve( b, M_tmp );
      b_norm = M_tmp.lpNorm( ( RealType ) 2.0 );

      matrix->vectorProduct( x, M_tmp );
      M_tmp.addVector( b, 1.0, -1.0 );
      preconditioner->solve( M_tmp, r_0 );
   }
   else {
      b_norm = b.lpNorm( 2.0 );
      matrix->vectorProduct( x, r_0 );
      r_0.addVector( b, 1.0, -1.0 );
   }

   sigma[ 0 ] = r_0.lpNorm( 2.0 );
   if( std::isnan( sigma[ 0 ] ) )
      throw std::runtime_error( "BiCGstab(ell): initial residue is NAN" );

   r_ast = r_0;
   r_ast /= sigma[ 0 ];
   rho_0 = 1.0;
   alpha = 0.0;
   omega = 1.0;
   u_0.setValue( 0.0 );

   if( b_norm == 0.0 )
       b_norm = 1.0;

   this->resetIterations();
   this->setResidue( sigma[ 0 ] / b_norm );

   while( this->checkNextIteration() )
   {
      rho_0 = - omega * rho_0;

      /****
       * Bi-CG part
       */
      for( int j = 0; j < ell; j++ ) {
         this->nextIteration();
         r_j.bind( &R.getData()[ j * ldSize ], size );

         rho_1 = r_ast.scalarProduct( r_j );
         beta = alpha * rho_1 / rho_0;
         rho_0 = rho_1;

         for( int i = 0; i <= j; i++ ) {
            u.bind( &U.getData()[ i * ldSize ], size );
            r_i.bind( &R.getData()[ i * ldSize ], size );
            /****
             * u_i := r_i - beta * u_i
             */
            u.addVector( r_i, 1.0, -beta );
         }

         /****
          * u_{j+1} = A u_j
          */
         u.bind( &U.getData()[ j * ldSize ], size );
         Au.bind( &U.getData()[ (j + 1) * ldSize ], size );
         matvec( u, Au );

         gamma = r_ast.scalarProduct( Au );
         alpha = rho_0 / gamma;

         for( int i = 0; i <= j; i++ ) {
            r_i.bind( &R.getData()[ i * ldSize ], size );
            u.bind( &U.getData()[ (i + 1) * ldSize ], size );
            /****
             * r_i := r_i - alpha * u_{i+1}
             */
            r_i.addVector( u, -alpha );
         }

         /****
          * r_{j+1} = A r_j
          */
         r_j.bind( &R.getData()[ j * ldSize ], size );
         r_i.bind( &R.getData()[ (j + 1) * ldSize ], size );
         matvec( r_j, r_i );

         /****
          * x_0 := x_0 + alpha * u_0
          */
         x.addVector( u_0, alpha );
      }

      /****
       * MGS part
       */
      for( int j = 1; j <= ell; j++ ) {
         r_j.bind( &R.getData()[ j * ldSize ], size );
         for( int i = 1; i < j; i++ ) {
            r_i.bind( &R.getData()[ i * ldSize ], size );
            /****
             * T_{i,j} = (r_i, r_j) / sigma_i
             * r_j := r_j - T_{i,j} * r_i
             */
            const int ij = (i-1) + (j-1) * ell;
            T[ ij ] = r_i.scalarProduct( r_j ) / sigma[ i ];
            r_j.addVector( r_i, -T[ ij ] );
         }
         sigma[ j ] = r_j.scalarProduct( r_j );
         g_1[ j ] = r_0.scalarProduct( r_j ) / sigma[ j ];
      }

      omega = g_1[ ell ];

      /****
       * g_0 = T^{-1} g_1
       */
      for( int j = ell; j >= 1; j-- ) {
         g_0[ j ] = g_1[ j ];
         for( int i = j + 1; i <= ell; i++ )
            g_0[ j ] -= T[ (j-1) + (i-1) * ell ] * g_0[ i ];
      }

      /****
       * g_2 = T * S * g_0,
       * where S e_1 = 0, S e_j = e_{j-1} for j = 2, ... ell
       */
      for( int j = 1; j < ell; j++ ) {
         g_2[ j ] = g_0[ j + 1 ];
         for( int i = j + 1; i < ell; i++ )
            g_2[ j ] += T[ (j-1) + (i-1) * ell ] * g_0[ i + 1 ];
      }

      /****
       * Final update
       */
      x.addVector( r_0, g_0[ 1 ] );
      u.bind( &U.getData()[ ell * ldSize ], size );
      r_i.bind( &R.getData()[ ell * ldSize ], size );
      u_0.addVector( u, -g_0[ ell ] );
      r_0.addVector( r_i, -g_1[ ell ] );
      // TODO: pro u_0 a r_0 lze rozšířit cyklus až do ell
      for( int j = 1; j < ell; j++ ) {
         u.bind( &U.getData()[ j * ldSize ], size );
         r_j.bind( &R.getData()[ j * ldSize ], size );
         u_0.addVector( u, -g_0[ j ] );
         r_0.addVector( r_j, -g_1[ j ] );
         x.addVector( r_j, g_2[ j ] );
      }

      if( exact_residue ) {
         /****
          * Compute the exact preconditioned residue into the 's' vector.
          */
         if( preconditioner ) {
            matrix->vectorProduct( x, M_tmp );
            M_tmp.addVector( b, 1.0, -1.0 );
            preconditioner->solve( M_tmp, res_tmp );
         }
         else {
            matrix->vectorProduct( x, res_tmp );
            res_tmp.addVector( b, 1.0, -1.0 );
         }
         sigma[ 0 ] = res_tmp.lpNorm( 2.0 );
         this->setResidue( sigma[ 0 ] / b_norm );
      }
      else {
         /****
          * Use the "orthogonal residue vector" for stopping.
          */
         sigma[ 0 ] = r_0.lpNorm( 2.0 );
         this->setResidue( sigma[ 0 ] / b_norm );
      }
   }

   this->refreshSolverMonitor( true );
   return this->checkConvergence();
}

template< typename Matrix,
          typename Preconditioner >
void
BICGStabL< Matrix, Preconditioner >::setSize( IndexType size )
{
   this->size = ldSize = size;
   R.setSize( (ell + 1) * ldSize );
   U.setSize( (ell + 1) * ldSize );
   r_ast.setSize( size );
   M_tmp.setSize( size );
   if( exact_residue )
      res_tmp.setSize( size );
   T.setSize( ell * ell );
   sigma.setSize( ell + 1 );
   g_0.setSize( ell + 1 );
   g_1.setSize( ell + 1 );
   g_2.setSize( ell + 1 );
}

} // namespace Linear
} // namespace Solvers
} // namespace TNL
