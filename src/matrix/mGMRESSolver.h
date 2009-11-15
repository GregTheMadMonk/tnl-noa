/***************************************************************************
                          mGMRESSolver.h  -  description
                             -------------------
    begin                : 2007/07/31
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

#ifndef mGMRESSolverH
#define mGMRESSolverH


#include <math.h>
#include <matrix/mMatrixSolver.h>

template< typename T > class mGMRESSolver : public mMatrixSolver< T >
{
   public:

   mGMRESSolver()
   : _r( 0 ), _w( 0 ), _s( 0 ), _cs( 0 ), _sn( 0 ), _v( 0 ), _H( 0 ), _M_tmp( 0 ),
     size( 0 ), restarting( 0 )
   {};

   void SetRestarting( long int rest )
   {
      if( size != 0 )
         AllocateSupportingArrays( size, rest );
      restarting = rest;
   };
   
   bool Solve( const mBaseMatrix< T >& A,
               const T* b,
               T* x, 
               const double& max_residue,
               const long int max_iterations,
               mPreconditioner< T >* precond = 0 ) 
   {
      if( restarting <= 0 )
      {
         cerr << "I have wrong value for the restarting of the GMRES solver. It is set to " << restarting 
              << ". Please set some positive value using the SetRestarting method." << endl;
         return false;
      }
      if( ! SetSize( A. GetSize(), restarting ) ) return false;
      
      
      long int i, j = 1, k, l;
      
      long int _size = size;
   
      T *r( _r ), *w( _w ), *p( _p ), *s( _s ), *cs( _cs ), *sn( _sn ), *v( _v ), *H( _H ), *M_tmp( _M_tmp );
    
      double normb( 0.0 ), beta( 0.0 ); 
      //T normb( 0.0 ), beta( 0.0 ); does not work with openmp yet
      // 1. Solve r from M r = b - A x_0
      if( precond )
      {
         precond -> Solve( b, M_tmp );
#ifdef HAVE_OPENMP
#pragma omp parallel for private( i ) reduction(+: normb ) firstprivate( M_tmp, _size )
#endif
         for( i = 0; i < _size; i ++ )
            normb += M_tmp[ i ] * M_tmp[ i ];

         A. VectorProduct( x, M_tmp );
#ifdef HAVE_OPENMP
#pragma omp parallel for private( i ) firstprivate( M_tmp, b )
#endif
         for( i = 0; i < size; i ++ )
            M_tmp[ i ] = b[ i ] - M_tmp[ i ];

         precond -> Solve( M_tmp, r );
#ifdef HAVE_OPENMP
#pragma omp parallel for private( i ) reduction(+: beta ) firstprivate( r )
#endif
         for( i = 0; i < size; i ++ )
            beta += r[ i ] * r[ i ];
      }
      else
      {
         A. VectorProduct( x, r );
#ifdef HAVE_OPENMP
#pragma omp parallel for private( i ) firstprivate( _size, r, b ) reduction(+: beta, normb )
#endif
         for( i = 0; i < _size; i ++ )
         {
            normb += b[ i ] * b[ i ];
            r[ i ] = b[ i ] - r[ i ];
            beta += r[ i ] * r[ i ];
         }
      }
      normb =sqrt( normb );
      beta = sqrt( beta );
      
      //dbgCout_ARRAY( r, size );
      //dbgCout_ARRAY( x, size );
     
      if( normb == 0.0 ) normb = 1.0;
    
      mMatrixSolver< T > :: iteration = 0; 
      mMatrixSolver< T > :: residue = beta / normb;
      //if( mMatrixSolver< T > :: residue <= max_residue )
      //{
      //   if( mMatrixSolver< T > :: verbosity > 0 )
      //      mMatrixSolver< T > :: PrintOut();
      //   return true;
      //}
      

      while( mMatrixSolver< T > :: iteration < max_iterations && 
             mMatrixSolver< T > :: residue > max_residue )
      {
         const long int m = restarting;
#ifdef HAVE_OPENMP
#pragma omp parallel for private( i ) firstprivate( m, H, s, cs, sn )
#endif
         for( i = 0; i < m + 1; i ++ )
            H[ i ] = s[ i ] = cs[ i ] = sn[ i ] = 0.0;
#ifdef HAVE_OPENMP
#pragma omp parallel for private( i ) firstprivate( m, _size, v )
#endif
         for( i = 0; i < _size * ( m + 1 ); i ++ )
            v[ i ] = 0;
         
         //dbgExpr( beta );
         //dbgCout_ARRAY( r, size );
         // v_0 = r / |r|
#ifdef HAVE_OPENMP
#pragma omp parallel for private( i ) firstprivate( beta, _size, v, r )
#endif
         for( i = 0; i < _size; i ++ )
            v[ i ] = r[ i ] / beta;    // ??? r / beta
         // s = |r| e_1
#ifdef HAVE_OPENMP
#pragma omp parallel for private( i ) firstprivate( m, s )
#endif
         for( i = 1; i <= m; i ++ )
            s[ i ] = 0.0;
         s[ 0 ] = beta;

         
         //dbgCout( " ----------- Starting m-loop -----------------" );
         for( i = 0; i < m && mMatrixSolver< T > :: iteration <= max_iterations; i++ )
         {

            // Solve w from M w = A v_i
            if( precond )
            {
               A. VectorProduct( &v[ i * size ], M_tmp );
               precond -> Solve( M_tmp, w );
            }
            else
                A. VectorProduct( &v[ i * size ], w );
            //dbgCout_ARRAY( s, m );
            //dbgCout_ARRAY( w, size );
            for( k = 0; k <= i; k++ )
            {
               // H_{k,i} = ( w, v_k )
               long int l;
               double H_k_i( 0.0 );
               //T H_k_i( 0.0 ); does not work with openmp ?yet?
#ifdef HAVE_OPENMP
#pragma omp parallel for private( l ) firstprivate( _size, w, v, k ) reduction( +: H_k_i )
#endif
               for( l = 0; l < _size; l ++ )
                  H_k_i += w[ l ] * v[ k * _size + l ];
               H[ k + i * ( m + 1 ) ] = H_k_i;
               
               // w = w - H_{k,i} v_k
#ifdef HAVE_OPENMP
#pragma omp parallel for private( l ) firstprivate( k, _size, w, H_k_i, v )
#endif
               for( l = 0; l < _size; l ++ )
                  w[ l ] -= H_k_i * v[ k * _size + l ];
            }
            // H_{i+1,i} = |w|
            T normw( 0.0 );
            for( l = 0; l < size; l ++ )
               normw += w[ l ] * w[ l ];
            normw = sqrt( normw );
            H[ i + 1 + i * ( m + 1 ) ] = normw;

            //dbgCout_ARRAY( w, size );
            //dbgExpr( normw );
            //dbgCout_MATRIX_CW( H, m + 1, m, 12  );
            
            // v_{i+1} = w / |w|
#ifdef HAVE_OPENMP
#pragma omp parallel for private( l ) firstprivate( _size, normw, i, v, w )
#endif
            for( l = 0; l < _size; l ++ )
               v[ ( i + 1 ) * _size + l ] = w[ l ] / normw;
            //dbgCout_MATRIX_CW( v, size, m + 1, 12 );


            //dbgCout( "Applying rotations" );
            for( k = 0; k < i; k++ )
               ApplyPlaneRotation( H[ k + i * ( m + 1 )],
                                   H[ k + 1 + i * ( m + 1 ) ],
                                   cs[ k ],
                                   sn[ k ] );
            
            GeneratePlaneRotation( H[ i + i * ( m + 1 ) ],
                                   H[ i + 1 + i * ( m + 1 ) ],
                                   cs[ i ],
                                   sn[ i ]);
            ApplyPlaneRotation( H[ i + i * ( m + 1 ) ],
                                H[ i + 1 + i * ( m + 1 ) ],
                                cs[ i ],
                                sn[ i ]);
            ApplyPlaneRotation( s[ i ],
                                s[ i + 1 ],
                                cs[ i ],
                                sn[ i ] );
            
            //dbgCout_MATRIX_CW( H, m + 1, m, 12 );
            //dbgCout_ARRAY( s, i + 2 );
            //dbgCout_ARRAY( cs, m + 1 );
            //dbgCout_ARRAY( sn, m + 1 );
            
            mMatrixSolver< T > :: residue = fabs( s[ i + 1 ] ) / normb;

            //dbgExpr( resid );
            //dbgExpr( normb );
            //dbgExpr( resid / normb );
            //dbgExpr( tol );

            if( mMatrixSolver< T > :: iteration % 10 == 0 &&
                mMatrixSolver< T > :: verbosity > 1 ) 
               mMatrixSolver< T > :: PrintOut();
            if( mMatrixSolver< T > :: residue < max_residue )
            {
               Update(x, i, m, H, s, v);
               if( mMatrixSolver< T > :: verbosity > 0 )
                  mMatrixSolver< T > :: PrintOut();
               return true;
            }
            //DBG_WAIT;
            mMatrixSolver< T > :: iteration ++;
         }
         Update( x, m - 1, m, H, s, v);
         //dbgCout_ARRAY( x, size );
         
         // r = M.solve(b - A * x);
         beta = 0.0;
         if( precond )
         {
            A. VectorProduct( x, M_tmp );
#ifdef HAVE_OPENMP
#pragma omp parallel for private( i ) firstprivate( _size, M_tmp, b )
#endif
            for( i = 0; i < _size; i ++ )
               M_tmp[ i ] = b[ i ] - M_tmp[ i ];
            precond -> Solve( M_tmp, r );
#ifdef HAVE_OPENMP
#pragma omp parallel for private( i ) firstprivate( _size, r ) reduction( +: beta )
#endif
            for( i = 0; i < _size; i ++ )
               beta += r[ i ] * r[ i ];
         } 
         else
         {
            A. VectorProduct( x, r );
#ifdef HAVE_OPENMP
#pragma omp parallel for private( i ) firstprivate( _size, r, b ) reduction(+:beta)
#endif
            for( i = 0; i < _size; i ++ )
            {
               r[ i ] = b[ i ] - r[ i ];
               beta += r[ i ] * r[ i ];
            }
         }
         beta = sqrt( beta );
         //dbgCout_ARRAY( r, size );
         //dbgExpr( beta );
         //dbgExpr( beta / normb );
         mMatrixSolver< T > :: residue = beta / normb;
         mMatrixSolver< T > :: iteration ++;
      }
      if( mMatrixSolver< T > :: verbosity > 0 ) 
         mMatrixSolver< T > :: PrintOut();
      if( mMatrixSolver< T > :: iteration == max_iterations ) return false;
      return true;
   };

   ~mGMRESSolver()
   {
      FreeSupportingArrays();
   };

   protected:

   void Update( T* x,
                long int k,
                long int m,
                const T* H,
                const T* s,
                const T* v )
   {
      //dbgFunctionName( "mGMRESSolver", "Update" );
      T* y = new T[ m + 1 ];
      long int i, j;
#ifdef HAVE_OPENMP
#pragma omp parallel for private( i ) firstprivate( m, y, s )
#endif
      for( i = 0; i <= m ; i ++ )
         y[ i ] = s[ i ];

      //dbgCout_ARRAY( y, m + 1 );
      // Backsolve:  
      for( i = k; i >= 0; i--)
      {
         y[ i ] /= H[ i + i * ( m + 1 ) ];
#ifdef HAVE_OPENMP
#pragma omp parallel for private( j ) firstprivate( i, y, H, m )
#endif
         for( j = i - 1; j >= 0; j--)
            y[ j ] -= H[ j + i * ( m + 1 ) ] * y[ i ];
      }
      //dbgCout_ARRAY( y, m + 1 );


      const long int _size = size;
      for( i = 0; i <= k; i++)
#ifdef HAVE_OPENMP
#pragma omp parallel for private( j ) firstprivate( i , _size )
#endif
         for( j = 0; j < _size; j ++ )
            x[ j ] += v[ i * _size + j ] * y[ i ];
      
      //dbgCout_ARRAY( x, size );

      delete[] y;
   };

   void GeneratePlaneRotation( T &dx,
                               T &dy,
                               T &cs,
                               T &sn )
   {
      if( dy == 0.0 )
      {
         cs = 1.0;
         sn = 0.0;
      }
      else
         if( fabs( dy ) > fabs( dx ) )
         {
            T temp = dx / dy;
            sn = 1.0 / sqrt( 1.0 + temp * temp );
            cs = temp * sn;
         } 
         else
         {
            T temp = dy / dx;
            cs = 1.0 / sqrt( 1.0 + temp * temp );
            sn = temp * cs;
         }
   };

   void ApplyPlaneRotation( T &dx,
                            T &dy,
                            T &cs,
                            T &sn )
   {
      T temp  =  cs * dx + sn * dy;
      dy = -sn * dx + cs * dy;
      dx = temp;
   };


   bool AllocateSupportingArrays( long int size, long int restart )
   {
      _r = new T[ size ];
      _w = new T[ size ];
      _s = new T[ restart + 1 ];
      _cs = new T[ restart + 1 ];
      _sn = new T[ restart + 1 ];
      _v = new T[ size * ( restart + 1 ) ];
      _H = new T[ ( restart + 1 ) * restart ];
      _M_tmp = new T[ size ];
      if( ! _r || ! _w || ! _s || ! _cs || ! _sn || ! _v || ! _H || ! _M_tmp )
      {
         cerr << "I could not allocated all supporting arrays for the CG solver." << endl;
         return false;
      }
      return true;
   };

   bool SetSize( long int _size, long int m )
   {
      if( size == _size && restarting == m ) return true;
      size = _size;
      restarting = m;
      FreeSupportingArrays();
      return AllocateSupportingArrays( size, restarting );
   };

   void FreeSupportingArrays()
   {
      if( _r ) delete[] _r;
      if( _w ) delete[] _w;
      if( _s ) delete[] _s;
      if( _cs ) delete[] _cs;
      if( _sn ) delete[] _sn;
      if( _v ) delete[] _v;
      if( _H ) delete[] _H;
      if( _M_tmp ) delete[] _M_tmp;
   };

   T *_r, *_w, *_p, *_s, *_cs, *_sn, *_v, *_H, *_M_tmp;

   long int size, restarting;
};

#endif
