/***************************************************************************
                          tnlCGSolver.h  -  description
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

#ifndef tnlCGSolverH
#define tnlCGSolverH

#include <math.h>
#include <matrix/tnlMatrixSolver.h>

template< typename T > class tnlCGSolver : public tnlMatrixSolver< T >
{
   public:

   tnlCGSolver()
   : r( 0 ), new_r( 0 ), p( 0 ), Ap( 0 ), size( 0 )
   {};
   
   bool Solve( const tnlBaseMatrix< T >& A,
               const T* b,
               T* x, 
               const double& max_residue,
               const int max_iterations,
               tnlPreconditioner< T >* precond = 0 )
   {
      if( ! SetSize( A. GetSize() ) ) return false;
      
      T alpha, beta, s1, s2;
      int i;
      tnlMatrixSolver< T > :: residue = max_residue + 1.0;
      tnlMatrixSolver< T > :: iteration = 0;
      
      T b_norm( 0.0 );
      for( i = 0; i < size; i ++ )
         b_norm += b[ i ] * b[ i ];
      b_norm = sqrt( b_norm );

      // r_0 = b - A x_0, p_0 = r_0
      A. VectorProduct( x, r );
      for( i = 0; i < size; i ++ )
         p[ i ] = r[ i ] = b[ i ] - r[ i ];
      

      while( tnlMatrixSolver< T > :: iteration < max_iterations && 
             tnlMatrixSolver< T > :: residue > max_residue )
      {
         // 1. alpha_j = ( r_j, r_j ) / ( A * p_j, p_j )
         A. VectorProduct( p, Ap );
         
         s1 = s2 = 0.0;
         for( i = 0; i < size; i ++ )
         {
            s1 += r[ i ] * r[ i ];
            s2 += Ap[ i ] * p[ i ];
         }
         // if s2 = 0 => p = 0 => r = 0 => we have the solution (provided A != 0)
         if( s2 == 0.0 ) alpha = 0.0;
         else alpha = s1 / s2;
         
         // 2. x_{j+1} = x_j + \alpha_j p_j
         for( i = 0; i < size; i ++ )
            x[ i ] += alpha * p[ i ];
         
         // 3. r_{j+1} = r_j - \alpha_j A * p_j
         for( i = 0; i < size; i ++ )
            new_r[ i ] = r[ i ] - alpha * Ap[ i ];

         //4. beta_j = ( r_{j+1}, r_{j+1} ) / ( r_j, r_j )
         s1 = s2 = 0.0;
         for( i = 0; i < size; i ++ )
         {
            s1 += new_r[ i ] * new_r[ i ];
            s2 += r[ i ] * r[ i ];
         }
         // if s2 = 0 => r = 0 => we have the solution
         if( s2 == 0.0 ) beta = 0.0;
         else beta = s1 / s2;

         // 5. p_{j+1} = r_{j+1} + beta_j * p_j
         for( i = 0; i < size; i ++ )
            p[ i ] = new_r[ i ] + beta * p[ i ];
     
         // 6. r_{j+1} = new_r
         T* tmp = r;
         r = new_r;
         new_r = tmp;
         
         if( tnlMatrixSolver< T > :: iteration % 10 == 0 )
         {
            tnlMatrixSolver< T > :: residue = GetResidue( A, b, x, b_norm, tmp );
            if( tnlMatrixSolver< T > :: verbosity > 1 ) 
               tnlMatrixSolver< T > :: PrintOut();
         }
         tnlMatrixSolver< T > :: iteration ++;
      }
      tnlMatrixSolver< T > :: residue = GetResidue( A, b, x, b_norm, r );
      if( tnlMatrixSolver< T > :: verbosity > 0 ) 
         tnlMatrixSolver< T > :: PrintOut();
   };

   ~tnlCGSolver()
   {
      FreeSupportingArrays();
   };

   protected:

   double GetResidue( const tnlBaseMatrix< T >& A,
                      const T* b,
                      const T* x,
                      const T& b_norm,
                      T* tmp ) 
   {
      A. VectorProduct( x, tmp );
      T res = 0.0;
      const int size = A. GetSize();
      int i;
      for( i = 0; i < size; i ++ )
      {
         T v = tmp[ i ] - b[ i ];
         res += v * v;
      }
      return sqrt( res ) / b_norm;
   };

   bool AllocateSupportingArrays( int size )
   {
      r = new T[ size ];
      new_r = new T[ size ];
      p = new T[ size ];
      Ap = new T[ size ];
      if( ! r || ! new_r || ! p || ! Ap )
      {
         cerr << "I could not allocated all supporting arrays for the CG solver." << endl;
         return false;
      }
      return true;
   };

   bool SetSize( int _size )
   {
      if( size == _size ) return true;
      size = _size;
      FreeSupportingArrays();
      return AllocateSupportingArrays( size );
   };

   void FreeSupportingArrays()
   {
      if( r ) delete[] r;
      if( new_r ) delete[] new_r;
      if( p ) delete[] p;
      if( Ap ) delete[] Ap;
   };

   T *r, *new_r, *p, *Ap;

   int size;
};

#endif
