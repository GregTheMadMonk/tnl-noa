/***************************************************************************
                          tnlBICGSolver.h  -  description
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

#ifndef tnlBICGSolverH
#define tnlBICGSolverH

#include <math.h>
#include <matrix/tnlMatrixSolver.h>

template< typename T > class tnlBICGSolver : public tnlMatrixSolver< T >
{
   public:

   tnlBICGSolver()
   : r( 0 ), r_ast( 0 ), r_new( 0 ), r_ast_new( 0 ), p( 0 ), p_ast( 0 ), tmp( 0 ),
     A_T( 0 ), size( 0 )
   {
   };

   void SetTransposeMatrix( tnlMatrix< T >* _A_T )
   {
      A_T = _A_T;
   };
   
   bool Solve( const tnlMatrix< T >& A,
               const T* b,
               T* x, 
               const double& max_residue,
               const int max_iterations,
               tnlPreconditioner< T >* precond = 0 )
   {
      if( ! SetSize( A. GetSize() ) ) return false;
      if( ! A_T )
      {
         cerr << "I need to know the transpose matrix for BICG solver. Use the SetTransposeMatrix method, please." << endl;
         return false;
      }

      tnlMatrixSolver< T > :: residue =  max_residue + 1.0;
      tnlMatrixSolver< T > :: iteration = 0;
      T alpha, beta, s1, s2;
      T b_norm( 0.0 );
      int i;
      for( i = 0; i < size; i ++ )
         b_norm += b[ i ] * b[ i ];

      // r_0 = b - A x_0, p_0 = r_0
      // r^ast_0 = r_0, p^ast_0 = r^ast_0
      
      //dbgCout( "Computing Ax" );
      A. VectorProduct( x, r );
      
      //dbgCout( "Computing r_0, r_ast_0, p_0 and p_ast_0..." );
      for( i = 0; i < size; i ++ )
         r[ i ] = r_ast[ i ] = 
         p[ i ] = p_ast[ i ] = b[ i ] - r[ i ];
      

      while( tnlMatrixSolver< T > :: iteration < max_iterations && 
             tnlMatrixSolver< T > :: residue > max_residue )
      {
         //dbgCout( "Starting BiCG iteration " << iter + 1 );

         // alpha_j = ( r_j, r^ast_j ) / ( A * p_j, p^ast_j )
         //dbgCout( "Computing Ap" );
         A. VectorProduct( p, tmp );

         //dbgCout( "Computing alpha" );
         s1 = s2 = 0.0;
         for( i = 0; i < size; i ++ )
         {
            s1 += r[ i ] * r_ast[ i ];
            s2 += tmp[ i ] * p_ast[ i ];
         }
         if( s2 == 0.0 ) alpha = 0.0;
         else alpha = s1 / s2;
         
         // x_{j+1} = x_j + alpha_j * p_j
         // r_{j+1} = r_j - alpha_j * A * p_j
         //dbgCout( "Computing new x and new r." );
         for( i = 0; i < size; i ++ )
         {
            x[ i ] += alpha * p[ i ];
            r_new[ i ] = r[ i ] - alpha * tmp[ i ];
         }
         
         //dbgCout( "Computing (A^T)p." );
         A_T -> VectorProduct( p_ast, tmp );

         s1 = s2 = 0.0;
         // r^ast_{j+1} = r^ats_j - alpha_j * A^T * p^ast_j
         // beta_j = ( r_{j+1}, r^ast_{j+1} ) / ( r_j, r^ast_j )
         //dbgCout( "Computing beta." );
         for( i = 0; i < size; i ++ )
         {
            r_ast_new[ i ] = r_ast[ i ] - alpha * tmp[ i ];
            s1 += r_new[ i ] * r_ast_new[ i ];
            s2 += r[ i ] * r_ast[ i ];
         }
         if( s2 == 0.0 ) beta = 0.0;
         else beta = s1 / s2;

         // p_{j+1} = r_{j+1} + beta_j * p_j
         // p^ast_{j+1} = r^ast_{j+1} + beta_j * p^ast_j
         for( i = 0; i < size; i ++ )
         {
            p[ i ] = r_new[ i ] + beta * p[ i ];
            p_ast[ i ] = r_ast_new[ i ] + beta * p_ast[ i ];
         }
         
         T* q;
         q = r_new;
         r_new = r;
         r = q;
         q = r_ast_new;
         r_ast_new = r_ast;
         r_ast = q;
         
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

   ~tnlBICGSolver()
   {
      FreeSupportingArrays();
   };

   protected:

   double GetResidue( const tnlMatrix< T >& A,
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
      r_ast = new T[ size ];
      r_new = new T[ size ];
      r_ast_new = new T[ size ];
      p = new T[ size ];
      p_ast = new T[ size ];
      tmp = new T[ size ];
      if( ! r || ! r_ast || ! r_ast_new || ! p || ! p_ast || ! tmp )
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
      if( r_ast ) delete[] r_ast;
      if( r_ast_new ) delete[] r_ast_new;
      if( p ) delete[] p;
      if( p_ast ) delete[] p_ast;
      if( tmp ) delete[] tmp;
   };

   T *r, *r_ast, *r_new, *r_ast_new, *p, *p_ast, *tmp;

   tnlMatrix< T >* A_T;

   int size;
};

#endif
