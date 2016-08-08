/***************************************************************************
                          BICGStab_impl.h  -  description
                             -------------------
    begin                : 2007/07/31
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename RealType,
          typename Vector >
RealType computeBICGStabNewP( Vector& p,
                              const Vector&r,
                              const RealType& beta,
                              const RealType& omega,
                              const Vector& Ap );

template< typename Matrix,
          typename Preconditioner >
BICGStab< Matrix, Preconditioner > :: BICGStab()
: preconditioner( 0 )
{
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
   //tnlIterativeSolver< RealType, IndexType >::configSetup( config, prefix );
}

template< typename Matrix,
          typename Preconditioner >
bool
BICGStab< Matrix, Preconditioner >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   return tnlIterativeSolver< RealType, IndexType >::setup( parameters, prefix );
}

template< typename Matrix,
          typename Preconditioner >
void BICGStab< Matrix, Preconditioner >::setMatrix( MatrixPointer& matrix )
{
   this->matrix = matrix;
}

template< typename Matrix,
           typename Preconditioner >
void BICGStab< Matrix, Preconditioner > :: setPreconditioner( const PreconditionerType& preconditioner )
{
   this->preconditioner = &preconditioner;
}

template< typename Matrix,
          typename Preconditioner >
   template< typename VectorPointer, typename ResidueGetter >
bool BICGStab< Matrix, Preconditioner >::solve( const VectorPointer& b, VectorPointer& x )
{
   if( ! this->setSize( matrix->getRows() ) ) return false;

   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   RealType alpha, beta, omega, s1, s2, rho( 0.0 ), bNorm( 0.0 );
   // r_0 = b - A x_0, p_0 = r_0
   // r^ast_0 = r_0

   this->matrix -> vectorProduct( *x, r );

   //if( bNorm == 0.0 ) bNorm = 1.0;

   /*if( M )
   {
      M -> Solve( b, M_tmp );
      for( i = 0; i < size; i ++ )
         b_norm += M_tmp[ i ] * M_tmp[ i ];

      for( i = 0; i < size; i ++ )
         M_tmp[ i ] =  b[ i ] - r[ i ];
      M -> Solve( M_tmp, r );
      for( i = 0; i < size; i ++ )
      {
         r_ast[ i ] = p[ i ] = r[ i ];
         rho += r[ i ] * r_ast[ i ];
      }
   }
   else*/
   {
      r.addVector( *b, 1.0, -1.0 );
      p = r_ast = r;
      rho = r.scalarProduct( r_ast );
      bNorm = b->lpNorm( 2.0 );
   }

   while( this->nextIteration() )
   {
      /****
       * alpha_j = ( r_j, r^ast_0 ) / ( A * p_j, r^ast_0 )
       */
      /*if( M ) // preconditioner
      {
         A. vectorProduct( p, M_tmp );
         M -> Solve( M_tmp, Ap );
      }
      else*/
          this->matrix -> vectorProduct( p, Ap );

      //dbgCout( "Computing alpha" );
      s2 = Ap.scalarProduct( r_ast );
      if( s2 == 0.0 ) alpha = 0.0;
      else alpha = rho / s2;

      /****
       * s_j = r_j - alpha_j * A p_j
       */
      s.addVectors( r, 1.0, Ap, -alpha );

      /****
       * omega_j = ( A s_j, s_j ) / ( A s_j, A s_j )
       */
      /*if( M ) // preconditioner
      {
         A. vectorProduct( s, M_tmp );
         DrawVector( "As", M_tmp, ( m_int ) ::sqrt( ( m_real ) size ) );
         M -> Solve( M_tmp, As );
      }
      else*/
          this->matrix -> vectorProduct( s, As );

      s1 = As. scalarProduct( s );
      s2 = As. scalarProduct( As );
      if( s2 == 0.0 ) omega = 0.0;
      else omega = s1 / s2;

      /****
       * x_{j+1} = x_j + alpha_j * p_j + omega_j * s_j
       */
      x->addVectors( p, alpha, s, omega );
      
      /****
       * r_{j+1} = s_j - omega_j * A * s_j
       */
      r.addVectors( s, 1.0, As, -omega );

      /****
       * beta = alpha_j / omega_j * ( r_{j+1}, r^ast_0 ) / ( r_j, r^ast_0 )
       */
      s1 = 0.0;
      s1 = r. scalarProduct( r_ast );
      if( rho == 0.0 ) beta = 0.0;
      else beta = ( s1 / rho ) * ( alpha / omega );
      rho = s1;

      /****
       * p_{j+1} = r_{j+1} + beta_j * ( p_j - omega_j * A p_j )
       */
      p.addVectors( r, 1.0, Ap, -beta * omega, beta );
      RealType residue = r.lpNorm( 2.0 );
      //RealType residue = computeBICGStabNewP( p, r, beta, omega, Ap );

      residue /= bNorm;
      this->setResidue( residue );
      if( this->getIterations() % 10 == 0 )
         this->setResidue( ResidueGetter::getResidue( matrix, b, x, bNorm ) );
   }
   //this->setResidue( ResidueGetter :: getResidue( *matrix, b, x, bNorm ) );
   this->refreshSolverMonitor( true );
   return this->checkConvergence();
};

template< typename Matrix,
          typename Preconditioner >
BICGStab< Matrix, Preconditioner > :: ~BICGStab()
{
};

template< typename Matrix,
          typename Preconditioner >
bool BICGStab< Matrix, Preconditioner > :: setSize( IndexType size )
{
   if( ! r. setSize( size ) ||
       ! r_ast. setSize( size ) ||
       ! p. setSize( size ) ||
       ! s. setSize( size ) ||
       ! Ap. setSize( size ) ||
       ! As. setSize( size ) ||
       ! M_tmp. setSize( size ) )
   {
      std::cerr << "I am not able to allocate all supporting arrays for the BICGStab solver." << std::endl;
      return false;
   }
   return true;

};

} // namespace Linear
} // namespace Solvers
} // namespace TNL
