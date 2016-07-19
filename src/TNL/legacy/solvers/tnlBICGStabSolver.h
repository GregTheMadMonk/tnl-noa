/***************************************************************************
                          tnlBICGStabSolver.h  -  description
                             -------------------
    begin                : 2007/07/31
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef tnlBICGStabSolverOldH
#define tnlBICGStabSolverOldH


#include <math.h>
#include <TNL/legacy/solvers/tnlMatrixSolver.h>

template< typename T > class tnlBICGStabSolverOld : public tnlMatrixSolver< T >
{
   public:

   tnlBICGStabSolverOld()
   : r( 0 ), r_ast( 0 ), r_new( 0 ), p( 0 ), s( 0 ), Ap( 0 ), As( 0 ), M_tmp( 0 ),
     size( 0 )
   {
   };

   bool Solve( const tnlMatrix< T >& A,
               const T* b,
               T* x,
               const double& max_residue,
               const int max_iterations,
               tnlPreconditioner< T >* precond = 0 )
   {
      dbgFunctionName( "tnlBICGStabSolver", "Solve" );
      if( ! SetSize( A. GetSize() ) ) return false;

      tnlMatrixSolver< T > :: residue =  max_residue + 1.0;
      tnlMatrixSolver< T > :: iteration = 0;
 
      T alpha, beta, omega, s1, s2, rho( 0.0 ), b_norm( 0.0 );
      int i;
      // r_0 = b - A x_0, p_0 = r_0
      // r^ast_0 = r_0
 
      dbgCout( "Computing Ax" );
      A. VectorProduct( x, r );
 
      dbgCout( "Computing r_0, r_ast_0, p_0 and b_norm ..." );
      /*if( M )
      {
         M -> Solve( b, M_tmp );
         for( i = 0; i < size; i ++ )
            b_norm += M_tmp[ i ] * M_tmp[ i ];

         for( i = 0; i < size; i ++ )
            M_tmp[ i ] =  b[ i ] - r[ i ];
         M -> Solve( M_tmp, r );
         for( i = 0; i < size; i ++ )
         {
            r_ast[ i ] = p[ i ] = r[ i ];
            rho += r[ i ] * r_ast[ i ];
         }
      }
      else*/
         for( i = 0; i < size; i ++ )
         {
            r[ i ] = r_ast[ i ] = p[ i ] = b[ i ] - r[ i ];
            rho += r[ i ] * r_ast[ i ];
            b_norm += b[ i ] * b[ i ];
         }
      if( b_norm == 0.0 ) b_norm = 1.0;
      //dbgExpr( b_norm );
 

      while( tnlMatrixSolver< T > :: iteration < max_iterations &&
             tnlMatrixSolver< T > :: residue > max_residue )
      {
         //dbgCout( "Starting BiCGStab iteration " << iter + 1 );

         // alpha_j = ( r_j, r^ast_0 ) / ( A * p_j, r^ast_0 )
         //dbgCout( "Computing Ap" );
         /*if( M ) // preconditioner
         {
            A. vectorProduct( p, M_tmp );
            DrawVector( "MAp", M_tmp, ( m_int ) ::sqrt( ( m_real ) size ) );
            M -> Solve( M_tmp, Ap );
            DrawVector( "Ap", Ap, ( m_int ) ::sqrt( ( m_real ) size ) );
         }
         else*/
             A. VectorProduct( p, Ap );
 
         //dbgCout( "Computing alpha" );
         s2 = 0.0;
         for( i = 0; i < size; i ++ )
         {
            s2 += Ap[ i ] * r_ast[ i ];
         }
         if( s2 == 0.0 ) alpha = 0.0;
         else alpha = rho / s2;
         //dbgExpr( alpha );

         // s_j = r_j - alpha_j * A p_j
         for( i = 0; i < size; i ++ )
         {
            //dbgExpr( r[ i ] );
            //dbgExpr( alpha * Ap[ i ] );
            s[ i ] = r[ i ] - alpha * Ap[ i ];
         }
         //DrawVector( "s", s, ( m_int ) ::sqrt( ( m_real ) size ) );

         // omega_j = ( A s_j, s_j ) / ( A s_j, A s_j )
         //dbgCout( "Computing As" );
         /*if( M ) // preconditioner
         {
            A. vectorProduct( s, M_tmp );
            DrawVector( "As", M_tmp, ( m_int ) ::sqrt( ( m_real ) size ) );
            M -> Solve( M_tmp, As );
         }
         else*/
             A. VectorProduct( s, As );
         s1 = s2 = 0.0;
         for( i = 0; i < size; i ++ )
         {
            s1 += As[ i ] * s[ i ];
            s2 += As[ i ] * As[ i ];
         }
         if( s2 == 0.0 ) omega = 0.0;
         else omega = s1 / s2;
         //dbgExpr( omega );
 
         //DrawVector( "p", p, ( m_int ) ::sqrt( ( m_real ) size ) );
         //DrawVector( "s", s, ( m_int ) ::sqrt( ( m_real ) size ) );
         // x_{j+1} = x_j + alpha_j * p_j + omega_j * s_j
         // r_{j+1} = s_j - omega_j * A * s_j
         //dbgCout( "Computing new x and new r." );
         for( i = 0; i < size; i ++ )
         {
            x[ i ] += alpha * p[ i ] + omega * s[ i ];
            r[ i ] = s[ i ] - omega * As[ i ];
         }
         //DrawVector( "x", x, ( m_int ) ::sqrt( ( m_real ) size ) );
         //DrawVector( "r", r, ( m_int ) ::sqrt( ( m_real ) size ) );
 
         // beta = alpha_j / omega_j * ( r_{j+1}, r^ast_0 ) / ( r_j, r^ast_0 )
         s1 = 0.0;
         for( i = 0; i < size; i ++ )
            s1 += r[ i ] * r_ast[ i ];
         if( rho == 0.0 ) beta = 0.0;
         else beta = ( s1 / rho ) * ( alpha / omega );
         rho = s1;

         // p_{j+1} = r_{j+1} + beta_j * ( p_j - omega_j * A p_j )
         tnlMatrixSolver< T > :: residue = 0.0;
         for( i = 0; i < size; i ++ )
         {
            p[ i ] = r[ i ] + beta * ( p[ i ] - omega * Ap[ i ] );
            tnlMatrixSolver< T > :: residue += r[ i ] * r[ i ];
            //dbgExpr( r[ i ] );
            //dbgExpr( res );
         }
         tnlMatrixSolver< T > :: residue = ::sqrt( tnlMatrixSolver< T > :: residue / b_norm );
 
         if( tnlMatrixSolver< T > :: iteration % 10 == 0 &&
             tnlMatrixSolver< T > :: verbosity > 1 )
                  tnlMatrixSolver< T > :: PrintOut();
         tnlMatrixSolver< T > :: iteration ++;
      }
      tnlMatrixSolver< T > :: residue = GetResidue( A, b, x, b_norm, r );
      if( tnlMatrixSolver< T > :: verbosity > 0 )
         tnlMatrixSolver< T > :: PrintOut();
   };

   ~tnlBICGStabSolverOld()
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
      return ::sqrt( res ) / b_norm;
   };

   bool AllocateSupportingArrays( int size )
   {
      r = new T[ size ];
      r_ast = new T[ size ];
      r_new = new T[ size ];
      p = new T[ size ];
      s = new T[ size ];
      Ap = new T[ size ];
      As = new T[ size ];
      M_tmp = new T[ size ];
      if( ! r || ! r_ast || ! p || ! s || ! Ap || ! Ap || ! M_tmp )
      {
         std::cerr << "I could not allocated all supporting arrays for the CG solver." << std::endl;
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
      if( r_new ) delete[] r_new;
      if( p ) delete[] p;
      if( s ) delete[] s;
      if( Ap ) delete[] Ap;
      if( As ) delete[] As;
      if( M_tmp ) delete[] M_tmp;
   };

   T *r, *r_ast, *r_new, *p, *s, *Ap, *As, *M_tmp;

   int size;
};

#endif
