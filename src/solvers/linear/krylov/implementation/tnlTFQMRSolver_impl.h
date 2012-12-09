/***************************************************************************
                          tnlTFQMRSolver_impl.h  -  description
                             -------------------
    begin                : Dec 8, 2012
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

#ifndef tnlTFQMRSolver_implH
#define tnlTFQMRSolver_implH

template< typename RealType,
          typename Vector >
RealType computeTFQMRNewP( Vector& p,
                           const Vector&r,
                           const RealType& beta,
                           const RealType& omega,
                           const Vector& Ap );

template< typename Matrix,
          typename Preconditioner >
tnlTFQMRSolver< Matrix, Preconditioner > :: tnlTFQMRSolver()
{
}

template< typename Matrix,
          typename Preconditioner >
tnlString tnlTFQMRSolver< Matrix, Preconditioner > :: getType() const
{
   return tnlString( "tnlTFQMRSolver< " ) +
          tnlString( GetParameterType( ( RealType ) 0.0 ) ) + ", " +
          Device :: getDeviceType() + ", " +
          tnlString( GetParameterType( ( IndexType ) 0 ) ) + " >";
}

template< typename Matrix,
          typename Preconditioner >
void tnlTFQMRSolver< Matrix, Preconditioner > :: setMatrix( const MatrixType& matrix )
{
   this -> matrix = &matrix;
}

template< typename Matrix,
           typename Preconditioner >
void tnlTFQMRSolver< Matrix, Preconditioner > :: setPreconditioner( const Preconditioner& preconditioner )
{
   this -> preconditioner = &preconditioner;
}

template< typename Matrix,
          typename Preconditioner >
   template< typename Vector, typename ResidueGetter >
bool tnlTFQMRSolver< Matrix, Preconditioner > :: solve( const Vector& b, Vector& x )
{
   dbgFunctionName( "tnlTFQMRSolver", "Solve" );
   if( ! this -> setSize( matrix -> getSize() ) ) return false;

   this -> resetIterations();
   this -> setResidue( this -> getMaxResidue() + 1.0 );

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
      r. saxmy( 1.0, b );
      w = u = r;
      matrix -> vectorProduct( u, v );
      d. setValue( 0.0 );
      tau = r. lpNorm( 2.0 );
      theta = 0.0;
      eta = 0.0;
      r_ast = r;
      rho = r_ast. sdot( r_ast );
   }

   while( this -> getIterations() < this -> getMaxIterations() &&
          this -> getResidue() > this -> getMaxResidue() )
   {
      //dbgCout( "Starting TFQMR iteration " << iter + 1 );

      if( this -> getIterations() % 2 == 0 )
      {
         alpha_new = alpha = rho / v. sdot( r_ast );
         u_new. saxpy( -alpha, v );
      }
      matrix -> vectorProduct( u, Au );
      w. saxpy( -alpha, Au );
      d. saxpsby( 1.0, u, theta * theta / alpha * eta );
      theta = w. lpNorm( 2.0 ) / tau;
      c = sqrt( 1 + theta * theta );
      tau = tau * theta * c;
      eta = c * c  * alpha;
      x. saxpy( eta, d );
      if( this -> getIterations() % 2 == 1 )
      {
         rho_new  = r ;
      }


      RealType sigma = r. sdot( v );
      alpha = rho / sigma;
      y[ 1 ]. saxpsbz( 1.0, y[ 0 ], -alpha, v );
      matrix -> vectorProduct( y[ 1 ], u[ 1 ] );

      for( IndexType j = 0; j < 2; j ++ )
      {
         const IndexType m = 2 * this -> getIterations() + j;
         w. saxpy( -alpha, u[ j ] );
         d. saxpsby( 1.0, y[ j ], theta * theta * eta / alpha , d );
         theta = w. lpNorm( 2.0 ) / tau;
         eta = w. lpNorm( 2.0 ) / tau;
         c = 1.0 / sqrt( 1.0 + theta * theta );
         tau = tau * theta * c;
         eta = c * c * alpha;
         x. saxpy( eta, d );
         if( tau * sqrt( M -1 ) ) *
      }

      // alpha_j = ( r_j, r^ast_0 ) / ( A * p_j, r^ast_0 )
      /*if( M ) // preconditioner
      {
         A. vectorProduct( p, M_tmp );
         M -> Solve( M_tmp, Ap );
      }
      else*/
          this -> matrix -> vectorProduct( p, Ap );

      //dbgCout( "Computing alpha" );
      s2 = Ap. sdot( r_ast );
      if( s2 == 0.0 ) alpha = 0.0;
      else alpha = rho / s2;

      // s_j = r_j - alpha_j * A p_j
      s. saxpsbz( 1.0, r, -alpha, Ap );

      // omega_j = ( A s_j, s_j ) / ( A s_j, A s_j )
      //dbgCout( "Computing As" );
      /*if( M ) // preconditioner
      {
         A. vectorProduct( s, M_tmp );
         DrawVector( "As", M_tmp, ( m_int ) sqrt( ( m_real ) size ) );
         M -> Solve( M_tmp, As );
      }
      else*/
          this -> matrix -> vectorProduct( s, As );
      s1 = s2 = 0.0;
      s1 = As. sdot( s );
      s2 = As. sdot( As );
      if( s2 == 0.0 ) omega = 0.0;
      else omega = s1 / s2;
      
      // x_{j+1} = x_j + alpha_j * p_j + omega_j * s_j
      // r_{j+1} = s_j - omega_j * A * s_j
      //dbgCout( "Computing new x and new r." );
      x. saxpsbzpy( alpha, p, omega, s );
      r. saxpsbz( 1.0, s, -omega, As );
      
      // beta = alpha_j / omega_j * ( r_{j+1}, r^ast_0 ) / ( r_j, r^ast_0 )
      s1 = 0.0;
      s1 = r. sdot( r_ast );
      if( rho == 0.0 ) beta = 0.0;
      else beta = ( s1 / rho ) * ( alpha / omega );
      rho = s1;

      // p_{j+1} = r_{j+1} + beta_j * ( p_j - omega_j * A p_j )
      RealType residue = computeTFQMRNewP( p, r, beta, omega, Ap );

      residue /= bNorm;
      this -> setResidue( residue );
      if( this -> getIterations() % 10 == 0 )
         this -> setResidue( ResidueGetter :: getResidue( *matrix, b, x, bNorm ) );
      if( ! this -> nextIteration() )
         return false;
      this -> refreshSolverMonitor();
   }
   this -> setResidue( ResidueGetter :: getResidue( *matrix, b, x, bNorm ) );
   this -> refreshSolverMonitor();
      if( this -> getResidue() > this -> getMaxResidue() ) return false;
   return true;
};

template< typename Matrix,
          typename Preconditioner >
tnlTFQMRSolver< Matrix, Preconditioner > :: ~tnlTFQMRSolver()
{
};

template< typename Matrix,
          typename Preconditioner >
bool tnlTFQMRSolver< Matrix, Preconditioner > :: setSize( IndexType size )
{
   if( ! r. setSize( size ) ||
       ! r_ast. setSize( size ) ||
       ! p. setSize( size ) ||
       ! s. setSize( size ) ||
       ! Ap. setSize( size ) ||
       ! As. setSize( size ) ||
       ! M_tmp. setSize( size ) )
   {
      cerr << "I am not able to allocated all supporting arrays for the TFQMR solver." << endl;
      return false;
   }
   return true;

};

template< typename RealType,
          typename Vector >
RealType computeTFQMRNewPHost( Vector& p,
                                  const Vector&r,
                                  const RealType& beta,
                                  const RealType& omega,
                                  const Vector& Ap )
{
   typedef typename Vector :: IndexType IndexType;
   const IndexType& size = p. getSize();
   RealType residue( 0.0 );
   for( IndexType i = 0; i < size; i ++ )
   {
      p[ i ] = r[ i ] + beta * ( p[ i ] - omega * Ap[ i ] );
      residue += r[ i ] * r[ i ];
   }
   return residue;
}

template< typename RealType,
          typename Vector >
RealType computeTFQMRNewPCuda( Vector& p,
                                  const Vector&r,
                                  const RealType& beta,
                                  const RealType& omega,
                                  const Vector& Ap )
{

}


template< typename RealType,
          typename Vector >
RealType computeTFQMRNewP( Vector& p,
                              const Vector&r,
                              const RealType& beta,
                              const RealType& omega,
                              const Vector& Ap )
{
   typedef typename Vector :: DeviceType DeviceType;
   switch( DeviceType :: getDevice() )
   {
      case tnlHostDevice:
         return computeTFQMRNewPHost( p, r, beta, omega, Ap );
         break;
      case tnlCudaDevice:
         return computeTFQMRNewPCuda( p, r, beta, omega, Ap );
         break;
   }
}


#endif
