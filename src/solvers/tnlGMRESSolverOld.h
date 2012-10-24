/***************************************************************************
                          tnlGMRESSolverOld.h  -  description
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

#ifndef tnlGMRESSolverOldH
#define tnlGMRESSolverOldH


#include <math.h>
#include <solvers/tnlMatrixSolver.h>

template< typename Real, typename Device = tnlHost, typename Index = int >
class tnlGMRESSolverOld : public tnlMatrixSolver< Real, Device, Index >
{
   public:

   tnlGMRESSolverOld( const tnlString& name );

   tnlString getType() const;

   void setRestarting( Index rest );

   bool solve( const tnlMatrix< Real, Device, Index >& A,
               const tnlVector< Real, Device, Index >& b,
               tnlVector< Real, Device, Index >& x,
               const Real& max_residue,
               const Index max_iterations,
               tnlPreconditioner< Real >* precond = 0 );

   ~tnlGMRESSolverOld();

   protected:

   /*!**
    * Here the parameter v is not constant because setSharedData is used 
    * on it inside of this method. It is not changed however.
    */
   void update( Index k,
                Index m,
                const tnlVector< Real, tnlHost, Index >& H,
                const tnlVector< Real, tnlHost, Index >& s,
                tnlVector< Real, Device, Index >& v,
                tnlVector< Real, Device, Index >& x );

   void generatePlaneRotation( Real &dx,
                               Real &dy,
                               Real &cs,
                               Real &sn );

   void applyPlaneRotation( Real &dx,
                            Real &dy,
                            Real &cs,
                            Real &sn );


   bool setSize( Index _size, Index m );

   tnlVector< Real, Device, Index > _r, _w, _v, _M_tmp;
   tnlVector< Real, tnlHost, Index > _s, _cs, _sn, _H;

   Index size, restarting;
};

template< typename Real, typename Device, typename Index >
tnlGMRESSolverOld< Real, Device, Index > :: tnlGMRESSolverOld( const tnlString& name )
: tnlMatrixSolver< Real, Device, Index >( name ),
  _r( "tnlGMRESSolverOld::_r" ),
  _w( "tnlGMRESSolverOld::_w" ),
  _v( "tnlGMRESSolverOld::_v" ),
  _M_tmp( "tnlGMRESSolverOld::_M_tmp" ),
  _s( "tnlGMRESSolverOld::_s" ),
  _cs( "tnlGMRESSolverOld::_cs" ),
  _sn( "tnlGMRESSolverOld::_sn" ),
  _H( "tnlGMRESSolverOld:_H" ),
  size( 0 ),
  restarting( 0 )
{
};
   
template< typename Real, typename Device, typename Index >
tnlString tnlGMRESSolverOld< Real, Device, Index > :: getType() const
{
   return tnlString( "tnlGMRESSolverOld< " ) +
          tnlString( GetParameterType( ( Real ) 0.0 ) ) +
          tnlString( ", " ) +
          Device :: getDeviceType() +
          tnlString( ", " ) +
          tnlString( GetParameterType( ( Index ) 0 ) ) +
          tnlString( " >" );
}

template< typename Real, typename Device, typename Index >
void tnlGMRESSolverOld< Real, Device, Index > :: setRestarting( Index rest )
{
   if( size != 0 )
      setSize( size, rest );
   restarting = rest;
};

template< typename Real, typename Device, typename Index >
bool tnlGMRESSolverOld< Real, Device, Index > :: solve( const tnlMatrix< Real, Device, Index >& A,
                                                     const tnlVector< Real, Device, Index >& b,
                                                     tnlVector< Real, Device, Index >& x,
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

   Real *r = _r. getData();
   Real *w = _w. getData();
   Real *s = _s. getData();
   Real *cs = _cs. getData();
   Real *sn = _sn. getData();
   Real *v = _v. getData();
   Real *H = _H. getData();
   Real *M_tmp = _M_tmp. getData();

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
      normb = b. lpNorm( ( Real ) 2.0 );
      _r. saxmy( ( Real ) 1.0, b );
      beta = _r. lpNorm( ( Real ) 2.0 );
   }

   if( normb == 0.0 ) normb = 1.0;

   this -> iteration = 0;
   this -> residue = beta / normb;

   tnlVector< Real, Device, Index > vi( "tnlGMRESSolverOld::vi" );
   tnlVector< Real, Device, Index > vk( "tnlGMRESSolverOld::vk" );
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
      vi. setSharedData( _v. getData(), size );
      vi. saxpy( ( Real ) 1.0 / beta, _r );
                
      _s. setValue( ( Real ) 0.0 );
      _s[ 0 ] = beta;
      


      //dbgCout( " ----------- Starting m-loop -----------------" );
      for( i = 0; i < m && this -> iteration <= max_iterations; i++ )
      {
         vi. setSharedData( &( _v. getData()[ i * size ] ), size );
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
            vk. setSharedData( &( _v. getData()[ k * _size ] ), _size );
            /***
             * H_{k,i} = ( w, v_k )
             */
            Real H_k_i = vk. sdot( _w );
            H[ k + i * ( m + 1 ) ] = H_k_i;
            
            /****
             * w = w - H_{k,i} v_k
             */
            _w. saxpy( -H_k_i, vk );
         }
         /***
          * H_{i+1,i} = |w|
          */
         Real normw = _w. lpNorm( ( Real ) 2.0 );
         H[ i + 1 + i * ( m + 1 ) ] = normw;

         /***
          * v_{i+1} = w / |w|
          */
         vi. setSharedData( &( _v. getData()[ ( i + 1 ) * size ] ), size );
         vi. saxpy( ( Real ) 1.0 / normw, _w );


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
         _r. saxmy( ( Real ) 1.0, b );
         beta = _r. lpNorm( ( Real ) 2.0 );
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

template< typename Real, typename Device, typename Index >
tnlGMRESSolverOld< Real, Device, Index > :: ~tnlGMRESSolverOld()
{
};

template< typename Real, typename Device, typename Index >
void tnlGMRESSolverOld< Real, Device, Index > :: update( Index k,
                                                      Index m,
                                                      const tnlVector< Real, tnlHost, Index >& H,
                                                      const tnlVector< Real, tnlHost, Index >& s,
                                                      tnlVector< Real, Device, Index >& v,
                                                      tnlVector< Real, Device, Index >& x )
{
   //dbgFunctionName( "tnlGMRESSolverOld", "Update" );
   tnlVector< Real, tnlHost, Index > y( "tnlGMRESSolverOld::update:y" );
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

   tnlVector< Real, Device, Index > vi( "tnlGMRESSolverOld::update:vi" );
   for( i = 0; i <= k; i++)
   {
      vi. setSharedData( &( v. getData()[ i * this -> size ] ), x. getSize() );
      x. saxpy( y[ i ], vi );
   }
};

template< typename Real, typename Device, typename Index >
void tnlGMRESSolverOld< Real, Device, Index > :: generatePlaneRotation( Real &dx,
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

template< typename Real, typename Device, typename Index >
void tnlGMRESSolverOld< Real, Device, Index > :: applyPlaneRotation( Real &dx,
                                                                  Real &dy,
                                                                  Real &cs,
                                                                  Real &sn )
{
   Real temp  =  cs * dx + sn * dy;
   dy =  cs * dy - sn * dx;
   dx = temp;
};

template< typename Real, typename Device, typename Index >
bool tnlGMRESSolverOld< Real, Device, Index > :: setSize( Index _size, Index m )
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
