/***************************************************************************
                          tnlCGSolver_impl.h  -  description
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

#ifndef tnlCGSolver_implH
#define tnlCGSolver_implH

template< typename Matrix,
          typename Preconditioner >
tnlCGSolver< Matrix, Preconditioner > :: tnlCGSolver()
{
}

template< typename Matrix,
           typename Preconditioner >
tnlString tnlCGSolver< Matrix, Preconditioner > :: getType() const
{
   return tnlString( "tnlCGSolver< " ) +
          this -> matrix -> getType() + ", " +
          this -> preconditioner -> getType() + " >";
}

template< typename Matrix,
          typename Preconditioner >
void
tnlCGSolver< Matrix, Preconditioner >::
configSetup( tnlConfigDescription& config,
             const tnlString& prefix )
{
   //tnlIterativeSolver< RealType, IndexType >::configSetup( config, prefix );
}

template< typename Matrix,
          typename Preconditioner >
bool
tnlCGSolver< Matrix, Preconditioner >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   return tnlIterativeSolver< RealType, IndexType >::setup( parameters, prefix );
}

template< typename Matrix,
          typename Preconditioner >
void tnlCGSolver< Matrix, Preconditioner > :: setMatrix( const MatrixType& matrix )
{
   this -> matrix = &matrix;
}

template< typename Matrix,
           typename Preconditioner >
void tnlCGSolver< Matrix, Preconditioner > :: setPreconditioner( const Preconditioner& preconditioner )
{
   this -> preconditioner = &preconditioner;
}

template< typename Matrix,
          typename Preconditioner >
   template< typename Vector, typename ResidueGetter >
bool
tnlCGSolver< Matrix, Preconditioner >::
solve( const Vector& b, Vector& x )
{
   if( ! this->setSize( matrix->getRows() ) ) return false;

   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   RealType alpha, beta, s1, s2;
   RealType bNorm = b. lpNorm( ( RealType ) 2.0 );

   /****
    * r_0 = b - A x_0, p_0 = r_0
    */
   this->matrix->vectorProduct( x, r );
   r. addVector( b, 1.0, -1.0 );
   p = r;

   while( this->nextIteration() )
   {
      /****
       * 1. alpha_j = ( r_j, r_j ) / ( A * p_j, p_j )
       */
      this->matrix->vectorProduct( p, Ap );

      s1 = r.scalarProduct( r );
      s2 = Ap.scalarProduct( p );

      /****
       * if s2 = 0 => p = 0 => r = 0 => we have the solution (provided A != 0)
       */
      if( s2 == 0.0 ) alpha = 0.0;
      else alpha = s1 / s2;
      
      /****
       * 2. x_{j+1} = x_j + \alpha_j p_j
       */
      x.addVector( p, alpha );
      
      /****
       * 3. r_{j+1} = r_j - \alpha_j A * p_j
       */
      new_r = r;
      new_r.addVector( Ap, -alpha );

      /****
       * 4. beta_j = ( r_{j+1}, r_{j+1} ) / ( r_j, r_j )
       */
      s1 = new_r. scalarProduct( new_r );
      s2 = r. scalarProduct( r );

      /****
       * if s2 = 0 => r = 0 => we have the solution
       */
      if( s2 == 0.0 ) beta = 0.0;
      else beta = s1 / s2;

      /****
       * 5. p_{j+1} = r_{j+1} + beta_j * p_j
       */
      p. addVector( new_r, 1.0, beta );

      /****
       * 6. r_{j+1} = new_r
       */
      new_r.swap( r );
      
      if( this -> getIterations() % 10 == 0 )
         this -> setResidue( ResidueGetter :: getResidue( *matrix, b, x, bNorm ) );
   }
   this -> setResidue( ResidueGetter :: getResidue( *matrix, b, x, bNorm ) );
   this -> refreshSolverMonitor();
   return this->checkConvergence();
};

template< typename Matrix,
          typename Preconditioner >
tnlCGSolver< Matrix, Preconditioner > :: ~tnlCGSolver()
{
};

template< typename Matrix,
          typename Preconditioner >
bool tnlCGSolver< Matrix, Preconditioner > :: setSize( IndexType size )
{
   if( ! r. setSize( size ) ||
       ! new_r. setSize( size ) ||
       ! p. setSize( size ) ||
       ! Ap. setSize( size ) )
   {
      cerr << "I am not able to allocated all supporting arrays for the CG solver." << endl;
      return false;
   }
   return true;
};

#endif
