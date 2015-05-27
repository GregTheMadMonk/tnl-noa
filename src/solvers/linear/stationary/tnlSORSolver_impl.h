/***************************************************************************
                          tnlSORSolver_impl.h  -  description
                             -------------------
    begin                : Nov 25, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
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

#ifndef TNLSORSOLVER_IMPL_H_
#define TNLSORSOLVER_IMPL_H_

template< typename Matrix, typename Preconditioner >
tnlSORSolver< Matrix, Preconditioner > :: tnlSORSolver()
: omega( 1.0 )
{
}

template< typename Matrix, typename Preconditioner >
tnlString tnlSORSolver< Matrix, Preconditioner > :: getType() const
{
   return tnlString( "tnlSORSolver< " ) +
          this -> matrix -> getType() + ", " +
          this -> preconditioner -> getType() + " >";
}

template< typename Matrix,
          typename Preconditioner >
void
tnlSORSolver< Matrix, Preconditioner >::
configSetup( tnlConfigDescription& config,
             const tnlString& prefix )
{
   //tnlIterativeSolver< RealType, IndexType >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "sor-omega", "Relaxation parameter of the SOR method.", 1.0 );
}

template< typename Matrix,
          typename Preconditioner >
bool
tnlSORSolver< Matrix, Preconditioner >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   tnlIterativeSolver< RealType, IndexType >::setup( parameters, prefix );
   this->setOmega( parameters.getParameter< double >( prefix + "sor-omega" ) );
   if( this->omega <= 0.0 || this->omega > 2.0 )
   {
      cerr << "Warning: The SOR method parameter omega is out of interval (0,2). The value is " << this->omega << " the method will not converge." << endl;
   }
   return true;   
}


template< typename Matrix, typename Preconditioner >
void tnlSORSolver< Matrix, Preconditioner > :: setOmega( const RealType& omega )
{
   this -> omega = omega;
}

template< typename Matrix, typename Preconditioner >
const typename tnlSORSolver< Matrix, Preconditioner > :: RealType& tnlSORSolver< Matrix, Preconditioner > :: getOmega( ) const
{
   return this -> omega;
}

template< typename Matrix,
          typename Preconditioner >
void tnlSORSolver< Matrix, Preconditioner > :: setMatrix( const MatrixType& matrix )
{
   this -> matrix = &matrix;
}

template< typename Matrix,
           typename Preconditioner >
void tnlSORSolver< Matrix, Preconditioner > :: setPreconditioner( const Preconditioner& preconditioner )
{
   this -> preconditioner = &preconditioner;
}


template< typename Matrix, typename Preconditioner >
   template< typename Vector, typename ResidueGetter >
bool tnlSORSolver< Matrix, Preconditioner > :: solve( const Vector& b,
                                                      Vector& x )
{
   const IndexType size = matrix -> getRows();

   this -> resetIterations();
   this -> setResidue( this -> getConvergenceResidue() + 1.0 );

   RealType bNorm = b. lpNorm( ( RealType ) 2.0 );

   while( this->nextIteration() )
   {
      for( IndexType row = 0; row < size; row ++ )
         matrix->performSORIteration( b,
                                      row,
                                      x,
                                      this->getOmega() );
      this -> setResidue( ResidueGetter :: getResidue( *matrix, x, b, bNorm ) );
   }
   this -> setResidue( ResidueGetter :: getResidue( *matrix, x, b, bNorm ) );
   this -> refreshSolverMonitor();
   return this->checkConvergence();
};

template< typename Matrix, typename Preconditioner >
tnlSORSolver< Matrix, Preconditioner > :: ~tnlSORSolver()
{
}

#include <matrices/tnlCSRMatrix.h>
#include <matrices/tnlEllpackMatrix.h>
#include <matrices/tnlMultidiagonalMatrix.h>

extern template class tnlSORSolver< tnlCSRMatrix< float,  tnlHost, int > >;
extern template class tnlSORSolver< tnlCSRMatrix< double, tnlHost, int > >;
extern template class tnlSORSolver< tnlCSRMatrix< float,  tnlHost, long int > >;
extern template class tnlSORSolver< tnlCSRMatrix< double, tnlHost, long int > >;

// TODO: fix this

/*extern template class tnlSORSolver< tnlEllpackMatrix< float,  tnlHost, int > >;
extern template class tnlSORSolver< tnlEllpackMatrix< double, tnlHost, int > >;
extern template class tnlSORSolver< tnlEllpackMatrix< float,  tnlHost, long int > >;
extern template class tnlSORSolver< tnlEllpackMatrix< double, tnlHost, long int > >;

extern template class tnlSORSolver< tnlMultiDiagonalMatrix< float,  tnlHost, int > >;
extern template class tnlSORSolver< tnlMultiDiagonalMatrix< double, tnlHost, int > >;
extern template class tnlSORSolver< tnlMultiDiagonalMatrix< float,  tnlHost, long int > >;
extern template class tnlSORSolver< tnlMultiDiagonalMatrix< double, tnlHost, long int > >;*/


#ifdef HAVE_CUDA
// TODO: fix this - it does not work with CUDA
/*extern template class tnlSORSolver< tnlCSRMatrix< float,  tnlCuda, int > >;
extern template class tnlSORSolver< tnlCSRMatrix< double, tnlCuda, int > >;
extern template class tnlSORSolver< tnlCSRMatrix< float,  tnlCuda, long int > >;
extern template class tnlSORSolver< tnlCSRMatrix< double, tnlCuda, long int > >;*/


/*
extern template class tnlSORSolver< tnlEllpackMatrix< float,  tnlCuda, int > >;
extern template class tnlSORSolver< tnlEllpackMatrix< double, tnlCuda, int > >;
extern template class tnlSORSolver< tnlEllpackMatrix< float,  tnlCuda, long int > >;
extern template class tnlSORSolver< tnlEllpackMatrix< double, tnlCuda, long int > >;
*/

/*
extern template class tnlSORSolver< tnlMutliDiagonalMatrix< float,  tnlCuda, int > >;
extern template class tnlSORSolver< tnlMutliDiagonalMatrix< double, tnlCuda, int > >;
extern template class tnlSORSolver< tnlMutliDiagonalMatrix< float,  tnlCuda, long int > >;
extern template class tnlSORSolver< tnlMutliDiagonalMatrix< double, tnlCuda, long int > >;
*/
#endif

#endif /* TNLSORSOLVER_IMPL_H_ */
