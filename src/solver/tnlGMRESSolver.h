/***************************************************************************
                          tnlGMRESSolver.h  -  description
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

#ifndef tnlGMRESSolverH
#define tnlGMRESSolverH


#include <math.h>
#include <solver/tnlMatrixSolver.h>

template< typename Real, tnlDevice Device = tnlHost, typename Index = int >
class tnlGMRESSolver : public tnlMatrixSolver< Real, Device, Index >
{
   public:

   tnlGMRESSolver( const tnlString& name );

   tnlString getType() const;

   void setRestarting( Index rest );

   bool solve( const tnlMatrix< Real, Device, Index >& A,
               const tnlLongVector< Real, Device, Index >& b,
               tnlLongVector< Real, Device, Index >& x,
               const Real& max_residue,
               const Index max_iterations,
               tnlPreconditioner< Real >* precond = 0 );

   ~tnlGMRESSolver();

   protected:

   /*!**
    * Here the parameter v is not constant because setSharedData is used 
    * on it inside of this method. It is not changed however.
    */
   void update( Index k,
                Index m,
                const tnlLongVector< Real, tnlHost, Index >& H,
                const tnlLongVector< Real, tnlHost, Index >& s,
                tnlLongVector< Real, Device, Index >& v,
                tnlLongVector< Real, Device, Index >& x );

   void generatePlaneRotation( Real &dx,
                               Real &dy,
                               Real &cs,
                               Real &sn );

   void applyPlaneRotation( Real &dx,
                            Real &dy,
                            Real &cs,
                            Real &sn );


   bool setSize( Index _size, Index m );

   tnlLongVector< Real, Device, Index > _r, _w, _v, _M_tmp;
   tnlLongVector< Real, tnlHost, Index > _s, _cs, _sn, _H;

   Index size, restarting;
};

template< typename Real, tnlDevice Device, typename Index >
tnlGMRESSolver< Real, Device, Index > :: tnlGMRESSolver( const tnlString& name )
: tnlMatrixSolver< Real, Device, Index >( name ),
  _r( "tnlGMRESSolver::_r" ),
  _w( "tnlGMRESSolver::_w" ),
  _v( "tnlGMRESSolver::_v" ),
  _M_tmp( "tnlGMRESSolver::_M_tmp" ),
  _s( "tnlGMRESSolver::_s" ),
  _cs( "tnlGMRESSolver::_cs" ),
  _sn( "tnlGMRESSolver::_sn" ),
  _H( "tnlGMRESSolver:_H" ),
  size( 0 ),
  restarting( 0 )
{
};
   
template< typename Real, tnlDevice Device, typename Index >
tnlString tnlGMRESSolver< Real, Device, Index > :: getType() const
{
   return tnlString( "tnlGMRESSolver< " ) +
          tnlString( GetParameterType( ( Real ) 0.0 ) ) +
          tnlString( ", " ) +
          getDeviceType( Device ) +
          tnlString( ", " ) +
          tnlString( GetParameterType( ( Index ) 0 ) ) +
          tnlString( " >" );
}

template< typename Real, tnlDevice Device, typename Index >
void tnlGMRESSolver< Real, Device, Index > :: setRestarting( Index rest )
{
   if( size != 0 )
      setSize( size, rest );
   restarting = rest;
};

template< typename Real, tnlDevice Device, typename Index >
bool tnlGMRESSolver< Real, Device, Index > :: solve( const tnlMatrix< Real, Device, Index >& A,
                                                     const tnlLongVector< Real, Device, Index >& b,
                                                     tnlLongVector< Real, Device, Index >& x,
                                                     const Real& max_residue,
                                                     const Index max_iterations,
                                                     tnlPreconditioner< Real >* precond )
{
   if( restarting <= 0 )
   {
      cerr << "I have wrong value for the restarting of the GMRES solver. It is set to " << restarting
           << ". Please set some positive value using the SetRestarting method." << endl;
      return false;
   }
   if( ! setSize( A. getSize(), restarting ) ) return false;


   Index i, j = 1, k, l;
   
   Index _size = size;

   Real *r = _r. getVector();
   Real *w = _w. getVector();
   Real *s = _s. getVector();
   Real *cs = _cs. getVector();
   Real *sn = _sn. getVector();
   Real *v = _v. getVector();
   Real *H = _H. getVector();
   Real *M_tmp = _M_tmp. getVector();

   Real normb( 0.0 ), beta( 0.0 );
   //T normb( 0.0 ), beta( 0.0 ); does not work with openmp yet
   /****
    * 1. Solve r from M r = b - A x_0
    */
   if( precond )
   {
      //precond -> Solve( b, M_tmp );
      for( i = 0; i < _size; i ++ )
         normb += M_tmp[ i ] * M_tmp[ i ];

      A. vectorProduct( x, _M_tmp );
      for( i = 0; i < size; i ++ )
         M_tmp[ i ] = b[ i ] - M_tmp[ i ];

      //precond -> Solve( M_tmp, r );
      for( i = 0; i < size; i ++ )
         beta += r[ i ] * r[ i ];
   }
   else
   {
      A. vectorProduct( x, _r );
      normb = tnlLpNorm( b, ( Real ) 2.0 );
      tnlSAXMY( ( Real ) 1.0, b, _r );
      beta = tnlLpNorm( _r, ( Real ) 2.0 );
   }

   if( normb == 0.0 ) normb = 1.0;

   this -> iteration = 0;
   this -> residue = beta / normb;

   tnlLongVector< Real, Device, Index > vi( "tnlGMRESSolver::vi" );
   tnlLongVector< Real, Device, Index > vk( "tnlGMRESSolver::vk" );
   while( this -> iteration < max_iterations &&
          this -> residue > max_residue )
   {
      const Index m = restarting;
      for( i = 0; i < m + 1; i ++ )
         H[ i ] = s[ i ] = cs[ i ] = sn[ i ] = 0.0;

      /****
       * v = 0
       */   
      _v. setValue( ( Real ) 0.0 );

      /***
       * v_0 = r / | r | =  1.0 / beta * r
       */
      vi. setSharedData( _v. getVector(), size );
      tnlSAXPY( ( Real ) 1.0 / beta, _r, vi );
                
      _s. setValue( ( Real ) 0.0 );
      _s[ 0 ] = beta;
      


      //dbgCout( " ----------- Starting m-loop -----------------" );
      for( i = 0; i < m && this -> iteration <= max_iterations; i++ )
      {
         vi. setSharedData( &( _v. getVector()[ i * size ] ), size );
         /****
          * Solve w from M w = A v_i
          */
         if( precond )
         {
            A. vectorProduct( vi, _M_tmp );            
            precond -> Solve( M_tmp, w );
         }
         else
             A. vectorProduct( vi, _w );
         
         for( k = 0; k <= i; k++ )
         {
            vk. setSharedData( &( _v. getVector()[ k * _size ] ), _size );
            /***
             * H_{k,i} = ( w, v_k )
             */
            Real H_k_i = tnlSDOT( _w, vk );
            H[ k + i * ( m + 1 ) ] = H_k_i;
            
            /****
             * w = w - H_{k,i} v_k
             */
            tnlSAXPY( -H_k_i, vk, _w );
         }
         /***
          * H_{i+1,i} = |w|
          */
         Real normw = tnlLpNorm( _w, ( Real ) 2.0 );
         H[ i + 1 + i * ( m + 1 ) ] = normw;

         /***
          * v_{i+1} = w / |w|
          */
         vi. setSharedData( &( _v. getVector()[ ( i + 1 ) * size ] ), size );
         tnlSAXPY( ( Real ) 1.0 / normw, _w, vi );


         //dbgCout( "Applying rotations" );
         for( k = 0; k < i; k++ )
            applyPlaneRotation( H[ k + i * ( m + 1 )],
                                H[ k + 1 + i * ( m + 1 ) ],
                                cs[ k ],
                                sn[ k ] );

         generatePlaneRotation( H[ i + i * ( m + 1 ) ],
                                H[ i + 1 + i * ( m + 1 ) ],
                                cs[ i ],
                                sn[ i ]);
         applyPlaneRotation( H[ i + i * ( m + 1 ) ],
                             H[ i + 1 + i * ( m + 1 ) ],
                             cs[ i ],
                             sn[ i ]);
         applyPlaneRotation( s[ i ],
                             s[ i + 1 ],
                             cs[ i ],
                             sn[ i ] );

         this -> residue = fabs( s[ i + 1 ] ) / normb;

         if( this -> iteration % 10 == 0 &&
             this -> verbosity > 1 )
            this -> printOut();
         if( this -> residue < max_residue )
         {
            update( i, m, _H, _s, _v, x );
            if( this -> verbosity > 0 )
               this -> printOut();
            return true;
         }
         //DBG_WAIT;
         this -> iteration ++;
      }
      update( m - 1, m, _H, _s, _v, x );
      //dbgCout_ARRAY( x, size );

      // r = M.solve(b - A * x);
      beta = 0.0;
      if( precond )
      {
         A. vectorProduct( x, _M_tmp );
         for( i = 0; i < _size; i ++ )
            M_tmp[ i ] = b[ i ] - M_tmp[ i ];
         precond -> Solve( M_tmp, r );
         for( i = 0; i < _size; i ++ )
            beta += r[ i ] * r[ i ];
      }
      else
      {
         A. vectorProduct( x, _r );
         tnlSAXMY( ( Real ) 1.0, b, _r );
         beta = tnlLpNorm( _r, ( Real ) 2.0 );
      }
      //beta = sqrt( beta );
      //dbgCout_ARRAY( r, size );
      //dbgExpr( beta );
      //dbgExpr( beta / normb );
      this -> residue = beta / normb;
      this -> iteration ++;
   }
   if( this -> verbosity > 0 )
      this -> printOut();
   if( this -> iteration == max_iterations ) return false;
   return true;
};

template< typename Real, tnlDevice Device, typename Index >
tnlGMRESSolver< Real, Device, Index > :: ~tnlGMRESSolver()
{
};

template< typename Real, tnlDevice Device, typename Index >
void tnlGMRESSolver< Real, Device, Index > :: update( Index k,
                                                      Index m,
                                                      const tnlLongVector< Real, tnlHost, Index >& H,
                                                      const tnlLongVector< Real, tnlHost, Index >& s,
                                                      tnlLongVector< Real, Device, Index >& v,
                                                      tnlLongVector< Real, Device, Index >& x )
{
   //dbgFunctionName( "tnlGMRESSolver", "Update" );
   tnlLongVector< Real, tnlHost, Index > y( "tnlGMRESSolver::update:y" );
   y. setSize( m + 1 );

   Index i, j;
   for( i = 0; i <= m ; i ++ )
      y[ i ] = s[ i ];

   //dbgCout_ARRAY( y, m + 1 );
   // Backsolve:
   for( i = k; i >= 0; i--)
   {
      y[ i ] /= H[ i + i * ( m + 1 ) ];
      for( j = i - 1; j >= 0; j--)
         y[ j ] -= H[ j + i * ( m + 1 ) ] * y[ i ];
   }
   //dbgCout_ARRAY( y, m + 1 );

   tnlLongVector< Real, Device, Index > vi( "tnlGMRESSolver::update:vi" );
   for( i = 0; i <= k; i++)
   {
      vi. setSharedData( &( v. getVector()[ i * this -> size ] ), x. getSize() );
      tnlSAXPY( y[ i ], vi, x );
   }
};

template< typename Real, tnlDevice Device, typename Index >
void tnlGMRESSolver< Real, Device, Index > :: generatePlaneRotation( Real &dx,
                                                                     Real &dy,
                                                                     Real &cs,
                                                                     Real &sn )
{
   if( dy == 0.0 )
   {
      cs = 1.0;
      sn = 0.0;
   }
   else
      if( fabs( dy ) > fabs( dx ) )
      {
         Real temp = dx / dy;
         sn = 1.0 / sqrt( 1.0 + temp * temp );
         cs = temp * sn;
      }
      else
      {
         Real temp = dy / dx;
         cs = 1.0 / sqrt( 1.0 + temp * temp );
         sn = temp * cs;
      }
};

template< typename Real, tnlDevice Device, typename Index >
void tnlGMRESSolver< Real, Device, Index > :: applyPlaneRotation( Real &dx,
                                                                  Real &dy,
                                                                  Real &cs,
                                                                  Real &sn )
{
   Real temp  =  cs * dx + sn * dy;
   dy =  cs * dy - sn * dx;
   dx = temp;
};

template< typename Real, tnlDevice Device, typename Index >
bool tnlGMRESSolver< Real, Device, Index > :: setSize( Index _size, Index m )
{
   if( size == _size && restarting == m ) return true;
   size = _size;
   restarting = m;
   if( ! _r. setSize( size ) ||
       ! _w. setSize( size ) ||
       ! _s. setSize( restarting + 1 ) ||
       ! _cs. setSize( restarting + 1 ) ||
       ! _sn. setSize( restarting + 1 ) ||
       ! _v. setSize( size * ( restarting + 1 ) ) ||
       ! _H. setSize( ( restarting + 1 ) * restarting ) ||
       ! _M_tmp. setSize( size ) )
   {
      cerr << "I could not allocated all supporting arrays for the CG solver." << endl;
      return false;
   }
   return true;
};

#endif
