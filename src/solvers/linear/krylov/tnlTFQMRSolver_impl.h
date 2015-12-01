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
   /*return tnlString( "tnlTFQMRSolver< " ) +
          tnlString( getType< RealType >() + ", " +
          Device :: getDeviceType() + ", " +
          tnlString( getType< RealType >() + " >";*/
}

template< typename Matrix,
          typename Preconditioner >
void
tnlTFQMRSolver< Matrix, Preconditioner >::
configSetup( tnlConfigDescription& config,
             const tnlString& prefix )
{
   //tnlIterativeSolver< RealType, IndexType >::configSetup( config, prefix );
}

template< typename Matrix,
          typename Preconditioner >
bool
tnlTFQMRSolver< Matrix, Preconditioner >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   return tnlIterativeSolver< RealType, IndexType >::setup( parameters, prefix );
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

   RealType tau, theta, eta, rho, alpha, w_norm;
   RealType b_norm = b. lpNorm( 2.0 );
   if( b_norm == 0.0 )
       b_norm = 1.0;

   this -> matrix -> vectorProduct( x, r );
   r. addVector( b, 1.0, -1.0 );
   w = u = r;
   matrix -> vectorProduct( u, Au );
   v = Au;
   d. setValue( 0.0 );
   tau = r. lpNorm( 2.0 );
   theta = eta = 0.0;
   r_ast = r;
   rho = r_ast. scalarProduct( r );
   alpha = 0.0; // TODO

   this->resetIterations();
   this -> setResidue( tau / b_norm );

   while( this->nextIteration() )
   {
      // start counting from 0
      const IndexType iter = this->getIterations() - 1;

//      cerr << "Starting TFQMR iteration " << iter << endl;

      if( iter % 2 == 0 ) {
         alpha = rho / v. scalarProduct( this -> r_ast );
      }
      else {
         // not necessary in even iteration since the previous odd iteration
         // already computed v_{m+1} = A*u_{m+1}
         matrix -> vectorProduct( u, Au );
      }
      w.addVector( Au, -alpha );
//      cerr << "alpha = " << alpha << endl;
      //cerr << "theta * theta / alpha * eta = " << theta * theta / alpha * eta << endl;
      d. addVector( u, 1.0, theta * theta * eta / alpha );
      w_norm = w. lpNorm( 2.0 );
//      cerr << "w_norm / b_norm = residue = " << w_norm / b_norm << endl;
      theta = w_norm / tau;
      const RealType c = 1.0 / sqrt( 1.0 + theta * theta );
      tau = tau * theta * c;
//      cerr << "tau * sqrt(m+1) = " << tau * sqrt(iter+1) << endl;
      eta = c * c  * alpha;
      //cerr << "eta = " << eta << endl;
      x.addVector( d, eta );

      this->setResidue( tau * sqrt(iter+1) / b_norm );
      if( iter > this->getMinIterations() && this->getResidue() < this->getConvergenceResidue() ) {
          break;
      }

      if( iter % 2 == 1 )
      {
         const RealType rho_new  = w. scalarProduct( this -> r_ast );
         const RealType beta = rho_new / rho;
         rho = rho_new;

         u.addVector( w, 1.0, beta );
         v.addVector( Au, beta, beta * beta );
         matrix -> vectorProduct( u, Au );
         v.addVector( Au, 1.0 );
      }
      else {
         u.addVector( v, -alpha );
      }
      
      this -> refreshSolverMonitor();
   }

   this->refreshSolverMonitor( true );
   return this->checkConvergence();
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
