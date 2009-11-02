/***************************************************************************
                          mSORSolver.h  -  description
                             -------------------
    begin                : 2007/07/30
    copyright            : (C) 2007 by Tomá¹ Oberhuber
    email                : oberhuber@seznam.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef mSORSolverH
#define mSORSolverH

#include <math.h>
#include "mMatrixSolver.h"

template< typename T > class mSORSolver : public mMatrixSolver< T >
{
   public:
   
   mSORSolver()
   : sor_omega( 1.0 )
   {};

   void SetSOROmega( const T& omega )
   {
      sor_omega = omega;
   };

   bool Solve( const mBaseMatrix< T >& A,
               const T* b,
               T* x, 
               const double& max_residue,
               const long int max_iterations,
               mPreconditioner< T >* precond = 0 )
   {
      const long int size = A. GetSize();
      long int i;
      mMatrixSolver< T > :: iteration = 0;
      mMatrixSolver< T > :: residue = max_residue + 1.0;;

      T b_norm( 0.0 );
      for( i = 0; i < size; i ++ )
         b_norm += b[ i ] * b[ i ];
      b_norm = sqrt( b_norm );

      while( mMatrixSolver< T > :: iteration < max_iterations && 
             max_residue < mMatrixSolver< T > :: residue )
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
         if( mMatrixSolver< T > :: iteration % 10 == 0 && mMatrixSolver< T > :: verbosity > 1 )
         {
            mMatrixSolver< T > :: residue = GetResidue( A, b, x, b_norm );
            mMatrixSolver< T > :: PrintOut();
         }
         mMatrixSolver< T > :: iteration ++;
      }
      if( mMatrixSolver< T > :: verbosity > 0 )
      {
         mMatrixSolver< T > :: residue = GetResidue( A, b, x, b_norm );
         mMatrixSolver< T > :: PrintOut();
      }
      if( mMatrixSolver< T > :: iteration <= max_iterations ) return true;
      return false;
   };

   protected:
   
   T GetResidue( const mBaseMatrix< T >& A,
                 const T* b,
                 const T* x,
                 const T& b_norm )
   {
      const long int size = A. GetSize();
      long int i;
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
