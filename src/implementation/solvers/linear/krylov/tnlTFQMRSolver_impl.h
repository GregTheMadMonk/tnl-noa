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
   if( ! this -> setSize( matrix -> getRows() ) ) return false;

   this -> resetIterations();
   this -> setResidue( this -> getConvergenceResidue() + 1.0 );

   RealType tau, theta, eta, rho, alpha;
   const RealType bNorm = b. lpNorm( 2.0 );
   this -> setResidue( ResidueGetter :: getResidue( *matrix, b, x, bNorm ) );

   dbgCout( "Computing Ax" );
   this -> matrix -> vectorProduct( x, r );

   /*if( M )
   {
   }
   else*/
   {
      r. alphaXPlusBetaY( -1.0, b, -1.0 );
      w = u = r;
      matrix -> vectorProduct( u, v );
      d. setValue( 0.0 );
      tau = r. lpNorm( 2.0 );
      theta = 0.0;
      eta = 0.0;
      r_ast = r;
      //cerr << "r_ast = " << r_ast << endl;
      rho = this -> r_ast. scalarProduct( this -> r_ast );
   }

   while( this -> getIterations() < this -> getMaxIterations() &&
          this -> getResidue() > this -> getConvergenceResidue() )
   {
      //dbgCout( "Starting TFQMR iteration " << iter + 1 );

      if( this -> getIterations() % 2 == 0 )
      {
         //cerr << "rho = " << rho << endl;
         alpha = rho / v. scalarProduct( this -> r_ast );
         //cerr << "new alpha = " << alpha << endl;
         u_new.addVector( v, -alpha );
      }
      matrix -> vectorProduct( u, Au );
      w.addVector( Au, -alpha );
      //cerr << "alpha = " << alpha << endl;
      //cerr << "theta * theta / alpha * eta = " << theta * theta / alpha * eta << endl;
      d. alphaXPlusBetaY( 1.0, u, theta * theta / alpha * eta );
      theta = w. lpNorm( 2.0 ) / tau;
      const RealType c = sqrt( 1.0 + theta * theta );
      tau = tau * theta * c;
      eta = c * c  * alpha;
      //cerr << "eta = " << eta << endl;
      x.addVector( d, eta );
      if( this -> getIterations() % 2 == 1 )
      {
         const RealType rho_new  = w. scalarProduct( this -> r_ast );
         const RealType beta = rho_new / rho;
         rho = rho_new;
         matrix -> vectorProduct( u, Au );
         Au.addVector( v, beta );
         u.addVector( w, 1.0, beta );
         matrix -> vectorProduct( u, Au_new );
         v.alphaXPlusBetaZ( 1.0, Au_new, beta, Au );
      }
      
      //this -> setResidue( residue );
      //if( this -> getIterations() % 10 == 0 )
         this -> setResidue( ResidueGetter :: getResidue( *matrix, b, x, bNorm ) );
      if( ! this -> nextIteration() )
         return false;
      this -> refreshSolverMonitor();
   }
   this -> setResidue( ResidueGetter :: getResidue( *matrix, b, x, bNorm ) );
   this -> refreshSolverMonitor();
      if( this -> getResidue() > this -> getConvergenceResidue() ) return false;
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
   if( ! d. setSize( size ) ||
       ! r. setSize( size ) ||
       ! w. setSize( size ) ||
       ! u. setSize( size ) ||
       ! v. setSize( size ) ||
       ! r_ast. setSize( size ) ||
       ! u_new. setSize( size ) ||
       ! Au. setSize( size ) ||
       ! Au_new. setSize( size ) )
   {
      cerr << "I am not able to allocate all supporting vectors for the TFQMR solver." << endl;
      return false;
   }
   return true;

};

#endif
