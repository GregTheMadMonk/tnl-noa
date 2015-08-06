/***************************************************************************
                          tnlBICGStabSolver_impl.h  -  description
                             -------------------
    begin                : 2007/07/31
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef tnlBICGStabSolver_implH
#define tnlBICGStabSolver_implH

#include <debug/tnlDebug.h>

template< typename RealType,
          typename Vector >
RealType computeBICGStabNewP( Vector& p,
                              const Vector&r,
                              const RealType& beta,
                              const RealType& omega,
                              const Vector& Ap );

template< typename Matrix,
          typename Preconditioner >
tnlBICGStabSolver< Matrix, Preconditioner > :: tnlBICGStabSolver()
{
}

template< typename Matrix,
          typename Preconditioner >
tnlString tnlBICGStabSolver< Matrix, Preconditioner > :: getType() const
{
   return tnlString( "tnlBICGStabSolver< " ) +
          this -> matrix -> getType() + ", " +
          this -> preconditioner -> getType() + " >";
}

template< typename Matrix,
          typename Preconditioner >
void
tnlBICGStabSolver< Matrix, Preconditioner >::
configSetup( tnlConfigDescription& config,
             const tnlString& prefix )
{
   //tnlIterativeSolver< RealType, IndexType >::configSetup( config, prefix );
}

template< typename Matrix,
          typename Preconditioner >
bool
tnlBICGStabSolver< Matrix, Preconditioner >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   return tnlIterativeSolver< RealType, IndexType >::setup( parameters, prefix );
}

template< typename Matrix,
          typename Preconditioner >
void tnlBICGStabSolver< Matrix, Preconditioner > :: setMatrix( const MatrixType& matrix )
{
   this -> matrix = &matrix;
}

template< typename Matrix,
           typename Preconditioner >
void tnlBICGStabSolver< Matrix, Preconditioner > :: setPreconditioner( const Preconditioner& preconditioner )
{
   this -> preconditioner = &preconditioner;
}

template< typename Matrix,
          typename Preconditioner >
   template< typename Vector, typename ResidueGetter >
bool tnlBICGStabSolver< Matrix, Preconditioner > :: solve( const Vector& b, Vector& x )
{
   dbgFunctionName( "tnlBICGStabSolver", "Solve" );
   if( ! this->setSize( matrix->getRows() ) ) return false;

   this -> resetIterations();
   this -> setResidue( this -> getConvergenceResidue() + 1.0 );

   RealType alpha, beta, omega, s1, s2, rho( 0.0 ), bNorm( 0.0 );
   // r_0 = b - A x_0, p_0 = r_0
   // r^ast_0 = r_0

   dbgCout( "Computing Ax" );
   this -> matrix -> vectorProduct( x, r );

   //if( bNorm == 0.0 ) bNorm = 1.0;

   dbgCout( "Computing r_0, r_ast_0, p_0 and b_norm ..." );
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
      r. addVector( b, 1.0, -1.0 );
      p = r_ast = r;
      rho = r. scalarProduct( r_ast );
      bNorm = b. lpNorm( 2.0 );
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
          this -> matrix -> vectorProduct( p, Ap );

      //dbgCout( "Computing alpha" );
      s2 = Ap. scalarProduct( r_ast );
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
         DrawVector( "As", M_tmp, ( m_int ) sqrt( ( m_real ) size ) );
         M -> Solve( M_tmp, As );
      }
      else*/
          this -> matrix -> vectorProduct( s, As );

      s1 = As. scalarProduct( s );
      s2 = As. scalarProduct( As );
      if( s2 == 0.0 ) omega = 0.0;
      else omega = s1 / s2;

      /****
       * x_{j+1} = x_j + alpha_j * p_j + omega_j * s_j
       */
      x.addVectors( p, alpha, s, omega );
      
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
         this->setResidue( ResidueGetter :: getResidue( *matrix, b, x, bNorm ) );
   }
   //this->setResidue( ResidueGetter :: getResidue( *matrix, b, x, bNorm ) );
   this->refreshSolverMonitor();
   return this->checkConvergence();
};

template< typename Matrix,
          typename Preconditioner >
tnlBICGStabSolver< Matrix, Preconditioner > :: ~tnlBICGStabSolver()
{
};

template< typename Matrix,
          typename Preconditioner >
bool tnlBICGStabSolver< Matrix, Preconditioner > :: setSize( IndexType size )
{
   if( ! r. setSize( size ) ||
       ! r_ast. setSize( size ) ||
       ! p. setSize( size ) ||
       ! s. setSize( size ) ||
       ! Ap. setSize( size ) ||
       ! As. setSize( size ) ||
       ! M_tmp. setSize( size ) )
   {
      cerr << "I am not able to allocate all supporting arrays for the BICGStab solver." << endl;
      return false;
   }
   return true;

};


#endif
