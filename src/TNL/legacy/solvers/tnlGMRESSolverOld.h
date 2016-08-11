/***************************************************************************
                          GMRESOld.h  -  description
                             -------------------
    begin                : 2007/07/31
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef GMRESOldH
#define GMRESOldH


#include <math.h>
#include <TNL/Containers/SharedVector.h>
#include <TNL/legacy/solvers/MatrixSolver.h>

template< typename Real, typename Device = Devices::Host, typename Index = int >
class GMRESOld : public MatrixSolver< Real, Device, Index >
{
   public:

   GMRESOld( const String& name );

   String getType() const;

   void setRestarting( Index rest );

   bool solve( const Matrix< Real, Device, Index >& A,
               const Vector< Real, Device, Index >& b,
               Vector< Real, Device, Index >& x,
               const Real& max_residue,
               const Index max_iterations,
               tnlPreconditioner< Real >* precond = 0 );

   ~GMRESOld();

   protected:

   /*!**
    * Here the parameter v is not constant because bind is used
    * on it inside of this method. It is not changed however.
    */
   void update( Index k,
                Index m,
                const Vector< Real, Devices::Host, Index >& H,
                const Vector< Real, Devices::Host, Index >& s,
                Vector< Real, Device, Index >& v,
                Vector< Real, Device, Index >& x );

   void generatePlaneRotation( Real &dx,
                               Real &dy,
                               Real &cs,
                               Real &sn );

   void applyPlaneRotation( Real &dx,
                            Real &dy,
                            Real &cs,
                            Real &sn );


   bool setSize( Index _size, Index m );

   Vector< Real, Device, Index > _r, _w, _v, _M_tmp;
   Vector< Real, Devices::Host, Index > _s, _cs, _sn, _H;

   Index size, restarting;
};

template< typename Real, typename Device, typename Index >
GMRESOld< Real, Device, Index > :: GMRESOld( const String& name )
: MatrixSolver< Real, Device, Index >( name ),
  _r( "GMRESOld::_r" ),
  _w( "GMRESOld::_w" ),
  _v( "GMRESOld::_v" ),
  _M_tmp( "GMRESOld::_M_tmp" ),
  _s( "GMRESOld::_s" ),
  _cs( "GMRESOld::_cs" ),
  _sn( "GMRESOld::_sn" ),
  _H( "GMRESOld:_H" ),
  size( 0 ),
  restarting( 0 )
{
};
 
template< typename Real, typename Device, typename Index >
String GMRESOld< Real, Device, Index > :: getType() const
{
   return String( "GMRESOld< " ) +
          String( getType( ( Real ) 0.0 ) ) +
          String( ", " ) +
          Device :: getDeviceType() +
          String( ", " ) +
          String( getType( ( Index ) 0 ) ) +
          String( " >" );
}

template< typename Real, typename Device, typename Index >
void GMRESOld< Real, Device, Index > :: setRestarting( Index rest )
{
   if( size != 0 )
      setSize( size, rest );
   restarting = rest;
};

template< typename Real, typename Device, typename Index >
bool GMRESOld< Real, Device, Index > :: solve( const Matrix< Real, Device, Index >& A,
                                                     const Vector< Real, Device, Index >& b,
                                                     Vector< Real, Device, Index >& x,
                                                     const Real& max_residue,
                                                     const Index max_iterations,
                                                     tnlPreconditioner< Real >* precond )
{
   if( restarting <= 0 )
   {
      std::cerr << "I have wrong value for the restarting of the GMRES solver. It is set to " << restarting
           << ". Please set some positive value using the SetRestarting method." << std::endl;
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
      _r.alphaXPlusBetaY( ( Real ) 1.0, b, -1.0 );
      beta = _r. lpNorm( ( Real ) 2.0 );
   }

   if( normb == 0.0 ) normb = 1.0;

   this->iteration = 0;
   this->residue = beta / normb;

   SharedVector< Real, Device, Index > vi;
   SharedVector< Real, Device, Index > vk;
   while( this->iteration < max_iterations &&
          this->residue > max_residue )
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
      vi. bind( _v. getData(), size );
      vi. alphaXPlusY( ( Real ) 1.0 / beta, _r );
 
      _s. setValue( ( Real ) 0.0 );
      _s[ 0 ] = beta;
 


      //dbgCout( " ----------- Starting m-loop -----------------" );
      for( i = 0; i < m && this->iteration <= max_iterations; i++ )
      {
         vi. bind( &( _v. getData()[ i * size ] ), size );
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
            vk. bind( &( _v. getData()[ k * _size ] ), _size );
            /***
             * H_{k,i} = ( w, v_k )
             */
            Real H_k_i = vk. scalarProduct( _w );
            H[ k + i * ( m + 1 ) ] = H_k_i;
 
            /****
             * w = w - H_{k,i} v_k
             */
            _w. alphaXPlusY( -H_k_i, vk );
         }
         /***
          * H_{i+1,i} = |w|
          */
         Real normw = _w. lpNorm( ( Real ) 2.0 );
         H[ i + 1 + i * ( m + 1 ) ] = normw;

         /***
          * v_{i+1} = w / |w|
          */
         vi. bind( &( _v. getData()[ ( i + 1 ) * size ] ), size );
         vi. alphaXPlusY( ( Real ) 1.0 / normw, _w );


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

         this->residue = fabs( s[ i + 1 ] ) / normb;

         if( this->iteration % 10 == 0 &&
             this->verbosity > 1 )
            this->printOut();
         if( this->residue < max_residue )
         {
            update( i, m, _H, _s, _v, x );
            if( this->verbosity > 0 )
               this->printOut();
            return true;
         }
         //DBG_WAIT;
         this->iteration ++;
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
         _r.alphaXPlusBetaY( ( Real ) 1.0, b, -1.0 );
         beta = _r. lpNorm( ( Real ) 2.0 );
      }
      //beta = ::sqrt( beta );
      //dbgCout_ARRAY( r, size );
      //dbgExpr( beta );
      //dbgExpr( beta / normb );
      this->residue = beta / normb;
      this->iteration ++;
   }
   if( this->verbosity > 0 )
      this->printOut();
   if( this->iteration == max_iterations ) return false;
   return true;
};

template< typename Real, typename Device, typename Index >
GMRESOld< Real, Device, Index > :: ~GMRESOld()
{
};

template< typename Real, typename Device, typename Index >
void GMRESOld< Real, Device, Index > :: update( Index k,
                                                      Index m,
                                                      const Vector< Real, Devices::Host, Index >& H,
                                                      const Vector< Real, Devices::Host, Index >& s,
                                                      Vector< Real, Device, Index >& v,
                                                      Vector< Real, Device, Index >& x )
{
   //dbgFunctionName( "GMRESOld", "Update" );
   Vector< Real, Devices::Host, Index > y( "GMRESOld::update:y" );
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

   SharedVector< Real, Device, Index > vi;
   for( i = 0; i <= k; i++)
   {
      vi. bind( &( v. getData()[ i * this->size ] ), x. getSize() );
      x. alphaXPlusY( y[ i ], vi );
   }
};

template< typename Real, typename Device, typename Index >
void GMRESOld< Real, Device, Index > :: generatePlaneRotation( Real &dx,
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
         sn = 1.0 / ::sqrt( 1.0 + temp * temp );
         cs = temp * sn;
      }
      else
      {
         Real temp = dy / dx;
         cs = 1.0 / ::sqrt( 1.0 + temp * temp );
         sn = temp * cs;
      }
};

template< typename Real, typename Device, typename Index >
void GMRESOld< Real, Device, Index > :: applyPlaneRotation( Real &dx,
                                                                  Real &dy,
                                                                  Real &cs,
                                                                  Real &sn )
{
   Real temp  =  cs * dx + sn * dy;
   dy =  cs * dy - sn * dx;
   dx = temp;
};

template< typename Real, typename Device, typename Index >
bool GMRESOld< Real, Device, Index > :: setSize( Index _size, Index m )
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
      std::cerr << "I could not allocated all supporting arrays for the CG solver." << std::endl;
      return false;
   }
   return true;
};

#endif
