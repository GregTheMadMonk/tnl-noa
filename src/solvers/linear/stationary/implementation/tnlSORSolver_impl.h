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
           tnlString( GetParameterType( ( RealType ) 0.0 ) ) +
           tnlString( ", " ) +
           Device :: getDeviceType() +
           tnlString( ", " ) +
           tnlString( GetParameterType( ( IndexType ) 0 ) ) +
           tnlString( " >" );
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
   const IndexType size = matrix -> getSize();

   this -> resetIterations();
   this -> setResidue( this -> getMaxResidue() + 1.0 );

   RealType bNorm = b. lpNorm( ( RealType ) 2.0 );

   while( this -> getIterations() < this -> getMaxIterations() &&
          this -> getResidue() > this -> getMaxResidue() )
   {
      matrix -> performSORIteration( this -> getOmega(),
                                     b,
                                     x,
                                     0,
                                     size );
      if( this -> getIterations() % 10 == 0 )
         this -> setResidue( ResidueGetter :: getResidue( *matrix, b, x, bNorm ) );
      if( ! this -> nextIteration() )
         return false;
      this -> refreshSolverMonitor();
   }
   this -> setResidue( ResidueGetter :: getResidue( *matrix, b, x, bNorm ) );
   this -> refreshSolverMonitor();
      if( this -> getIterations() == this -> getMaxIterations() ) return false;
   return true;
};

template< typename Matrix, typename Preconditioner >
tnlSORSolver< Matrix, Preconditioner > :: ~tnlSORSolver()
{
}


#endif /* TNLSORSOLVER_IMPL_H_ */
