/***************************************************************************
                          tnlSORSolver.h  -  description
                             -------------------
    begin                : 2007/07/30
    copyright            : (C) 2007 by Tomá¹ Oberhuber
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

#ifndef tnlSORSolverH
#define tnlSORSolverH

#include <math.h>
#include <matrix/tnlMatrixSolver.h>

template< typename T > class tnlSORSolver : public tnlMatrixSolver< T >
{
   public:
   
   tnlSORSolver()
   : sor_omega( 1.0 )
   {};

   void SetSOROmega( const T& omega )
   {
      sor_omega = omega;
   };

   bool Solve( const tnlBaseMatrix< T >& A,
               const T* b,
               T* x, 
               const double& max_residue,
               const int max_iterations,
               tnlPreconditioner< T >* precond = 0 )
   {
      const int size = A. GetSize();
      int i;
      tnlMatrixSolver< T > :: iteration = 0;
      tnlMatrixSolver< T > :: residue = max_residue + 1.0;;

      T b_norm( 0.0 );
      for( i = 0; i < size; i ++ )
         b_norm += b[ i ] * b[ i ];
      b_norm = sqrt( b_norm );

      while( tnlMatrixSolver< T > :: iteration < max_iterations && 
             max_residue < tnlMatrixSolver< T > :: residue )
      {
         for( i = 0; i < size; i ++ )
         {
            T x_i = x[ i ];
            x[ i ] = 0.0; // diagonal entry is excluded from the vector and matrix-row product
            T sigma = A. RowProduct( i, x );
            T diag_entry = A. GetElement( i, i );
            if( diag_entry == 0.0 )
            {
               cerr << "I found a zero at diagonal at line " << i << " when runnnig SOR solver. I cannot continue." << endl;
               return false;
            }
            x[ i ] = ( 1.0 - sor_omega ) * x_i + sor_omega * ( b[ i ] - sigma ) / diag_entry;
         }
         if( tnlMatrixSolver< T > :: iteration % 10 == 0 && tnlMatrixSolver< T > :: verbosity > 1 )
         {
            tnlMatrixSolver< T > :: residue = GetResidue( A, b, x, b_norm );
            tnlMatrixSolver< T > :: PrintOut();
         }
         tnlMatrixSolver< T > :: iteration ++;
      }
      if( tnlMatrixSolver< T > :: verbosity > 0 )
      {
         tnlMatrixSolver< T > :: residue = GetResidue( A, b, x, b_norm );
         tnlMatrixSolver< T > :: PrintOut();
      }
      if( tnlMatrixSolver< T > :: iteration <= max_iterations ) return true;
      return false;
   };

   protected:
   
   T GetResidue( const tnlBaseMatrix< T >& A,
                 const T* b,
                 const T* x,
                 const T& b_norm )
   {
      const int size = A. GetSize();
      int i;
      T res( ( T ) 0.0 );
      for( i = 0; i < size; i ++ )
      {
         T err = fabs( A. RowProduct( i, x ) - b[ i ] );
         res += err * err;
      }
      return sqrt( res ) / b_norm;

   };

   T sor_omega;
};

#endif
