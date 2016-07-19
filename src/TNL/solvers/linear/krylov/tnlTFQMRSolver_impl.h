/***************************************************************************
                          tnlTFQMRSolver_impl.h  -  description
                             -------------------
    begin                : Dec 8, 2012
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

template< typename Matrix,
          typename Preconditioner >
tnlTFQMRSolver< Matrix, Preconditioner > :: tnlTFQMRSolver()
: size( 0 ),
  matrix( 0 ),
  preconditioner( 0 )
{
}

template< typename Matrix,
          typename Preconditioner >
tnlString tnlTFQMRSolver< Matrix, Preconditioner > :: getType() const
{
   return tnlString( "tnlTFQMRSolver< " ) +
          this->matrix -> getType() + ", " +
          this->preconditioner -> getType() + " >";
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
   this->matrix = &matrix;
}

template< typename Matrix,
           typename Preconditioner >
void tnlTFQMRSolver< Matrix, Preconditioner > :: setPreconditioner( const Preconditioner& preconditioner )
{
   this->preconditioner = &preconditioner;
}

template< typename Matrix,
          typename Preconditioner >
   template< typename Vector, typename ResidueGetter >
bool tnlTFQMRSolver< Matrix, Preconditioner > :: solve( const Vector& b, Vector& x )
{
   dbgFunctionName( "tnlTFQMRSolver", "Solve" );
   if( ! this->setSize( matrix -> getRows() ) ) return false;

   RealType tau, theta, eta, rho, alpha, b_norm, w_norm;

   if( preconditioner ) {
      preconditioner -> solve( b, M_tmp );
      b_norm = M_tmp. lpNorm( ( RealType ) 2.0 );

      matrix -> vectorProduct( x, M_tmp );
      M_tmp.addVector( b, 1.0, -1.0 );
      preconditioner -> solve( M_tmp, r );
   }
   else {
      b_norm = b. lpNorm( 2.0 );
      matrix -> vectorProduct( x, r );
      r.addVector( b, 1.0, -1.0 );
   }
   w = u = r;
   if( preconditioner ) {
      matrix -> vectorProduct( u, M_tmp );
      preconditioner -> solve( M_tmp, Au );
   }
   else {
      matrix -> vectorProduct( u, Au );
   }
   v = Au;
   d. setValue( 0.0 );
   tau = r. lpNorm( 2.0 );
   theta = eta = 0.0;
   r_ast = r;
   rho = r_ast. scalarProduct( r );
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
         if( preconditioner ) {
            matrix -> vectorProduct( u, M_tmp );
            preconditioner -> solve( M_tmp, Au );
         }
         else {
            matrix -> vectorProduct( u, Au );
         }
      }
      w.addVector( Au, -alpha );
      d.addVector( u, 1.0, theta * theta * eta / alpha );
      w_norm = w. lpNorm( 2.0 );
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
         const RealType rho_new  = w. scalarProduct( this->r_ast );
         const RealType beta = rho_new / rho;
         rho = rho_new;

         u.addVector( w, 1.0, beta );
         v.addVector( Au, beta, beta * beta );
         if( preconditioner ) {
            matrix -> vectorProduct( u, M_tmp );
            preconditioner -> solve( M_tmp, Au );
         }
         else {
            matrix -> vectorProduct( u, Au );
         }
         v.addVector( Au, 1.0 );
      }
      else {
         u.addVector( v, -alpha );
      }
 
      this->refreshSolverMonitor();
   }

//   this->matrix->vectorProduct( x, r );
//   r.addVector( b, 1.0, -1.0 );
//   this->setResidue( r.lpNorm( 2.0 ) / b_norm );

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
   if( this->size == size )
      return true;
   this->size = size;
   if( ! d. setSize( size ) ||
       ! r. setSize( size ) ||
       ! w. setSize( size ) ||
       ! u. setSize( size ) ||
       ! v. setSize( size ) ||
       ! r_ast. setSize( size ) ||
       ! Au. setSize( size ) ||
       ! M_tmp. setSize( size ) )
   {
      std::cerr << "I am not able to allocate all supporting vectors for the TFQMR solver." << std::endl;
      return false;
   }
   return true;

};

} // namespace TNL

