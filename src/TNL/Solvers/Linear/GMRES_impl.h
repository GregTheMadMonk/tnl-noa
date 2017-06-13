/***************************************************************************
                          GMRES_impl.h  -  description
                             -------------------
    begin                : Nov 25, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include "GMRES.h"

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix,
           typename Preconditioner >
GMRES< Matrix, Preconditioner >::
GMRES()
: size( 0 ),
  restarting_min( 10 ),
  restarting_max( 10 ),
  restarting_step_min( 3 ),
  restarting_step_max( 3 )
{
   /****
    * Clearing the shared pointer means that there is no
    * preconditioner set.
    */
   this->preconditioner.clear();
}

template< typename Matrix,
          typename Preconditioner >
GMRES< Matrix, Preconditioner >::
~GMRES()
{
}

template< typename Matrix,
          typename Preconditioner >
String
GMRES< Matrix, Preconditioner >::
getType() const
{
   return String( "GMRES< " ) +
          this->matrix -> getType() + ", " +
          this->preconditioner -> getType() + " >";
}

template< typename Matrix,
          typename Preconditioner >
void
GMRES< Matrix, Preconditioner >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   //IterativeSolver< RealType, IndexType >::configSetup( config, prefix );
   config.addEntry< int >( prefix + "gmres-restarting-min", "Minimal number of iterations after which the GMRES restarts.", 10 );
   config.addEntry< int >( prefix + "gmres-restarting-max", "Maximal number of iterations after which the GMRES restarts.", 10 );
   config.addEntry< int >( prefix + "gmres-restarting-step-min", "Minimal adjusting step for the adaptivity of the GMRES restarting parameter.", 3 );
   config.addEntry< int >( prefix + "gmres-restarting-step-max", "Maximal adjusting step for the adaptivity of the GMRES restarting parameter.", 3 );
}

template< typename Matrix,
          typename Preconditioner >
bool
GMRES< Matrix, Preconditioner >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   IterativeSolver< RealType, IndexType >::setup( parameters, prefix );
   restarting_min = parameters.getParameter< int >( "gmres-restarting-min" );
   this->setRestarting( parameters.getParameter< int >( "gmres-restarting-max" ) );
   restarting_step_min = parameters.getParameter< int >( "gmres-restarting-step-min" );
   restarting_step_max = parameters.getParameter< int >( "gmres-restarting-step-max" );
   return true;
}

template< typename Matrix,
          typename Preconditioner >
void
GMRES< Matrix, Preconditioner >::
setRestarting( IndexType rest )
{
   if( size != 0 )
      setSize( size, rest );
   restarting_max = rest;
}

template< typename Matrix,
          typename Preconditioner >
void
GMRES< Matrix, Preconditioner >::
setMatrix( const MatrixPointer& matrix )
{
   this->matrix = matrix;
}

template< typename Matrix,
          typename Preconditioner >
void
GMRES< Matrix, Preconditioner >::
setPreconditioner( const PreconditionerPointer& preconditioner )
{
   this->preconditioner = preconditioner;
}

template< typename Matrix,
          typename Preconditioner >
   template< typename Vector, typename ResidueGetter >
bool
GMRES< Matrix, Preconditioner >::
solve( const Vector& b, Vector& x )
{
   TNL_ASSERT( matrix, std::cerr << "No matrix was set in GMRES. Call setMatrix() before solve()." << std::endl );
   if( restarting_min <= 0 || restarting_max <= 0 || restarting_min > restarting_max )
   {
      std::cerr << "Wrong value for the GMRES restarting parameters: r_min = " << restarting_min
                << ", r_max = " << restarting_max << std::endl;
      return false;
   }
   if( restarting_step_min < 0 || restarting_step_max < 0 || restarting_step_min > restarting_step_max )
   {
      std::cerr << "Wrong value for the GMRES restarting adjustment parameters: d_min = " << restarting_step_min
                << ", d_max = " << restarting_step_max << std::endl;
      return false;
   }
   if( ! setSize( matrix -> getRows(), restarting_max ) )
   {
       std::cerr << "I am not able to allocate enough memory for the GMRES solver. You may try to decrease the restarting parameter." << std::endl;
       return false;
   }

   IndexType _size = size;
 
   //RealType *w = _w.getData();
   RealType *s = _s.getData();
   RealType *cs = _cs.getData();
   RealType *sn = _sn.getData();
   RealType *v = _v.getData();
   RealType *H = _H.getData();
   RealType *M_tmp = _M_tmp.getData();

   RealType normb( 0.0 ), beta( 0.0 );
   /****
    * 1. Solve r from M r = b - A x_0
    */
   if( preconditioner )
   {
      this->preconditioner->solve( b, _M_tmp );
      normb = _M_tmp.lpNorm( ( RealType ) 2.0 );

      matrix -> vectorProduct( x, _M_tmp );
      _M_tmp.addVector( b, ( RealType ) 1.0, -1.0 );

      this->preconditioner->solve( _M_tmp, _r );
   }
   else
   {
      matrix -> vectorProduct( x, _r );
      normb = b.lpNorm( ( RealType ) 2.0 );
      _r.addVector( b, ( RealType ) 1.0, -1.0 );
   }
   beta = _r.lpNorm( ( RealType ) 2.0 );
 
   //cout << "norm b = " << normb << std::endl;
   //cout << " beta = " << beta << std::endl;


   if( normb == 0.0 ) normb = 1.0;

   this->resetIterations();
   this->setResidue( beta / normb );

   // parameters for the adaptivity of the restarting parameter
         RealType beta_ratio = 1;           // = beta / beta_ratio (small value indicates good convergence rate)
   const RealType max_beta_ratio = 0.99;    // = cos(8°) \approx 0.99
   const RealType min_beta_ratio = 0.175;   // = cos(80°) \approx 0.175
         int restart_cycles = 0;    // counter of restart cycles
         int m = restarting_max;    // current restarting parameter

   Containers::Vector< RealType, DeviceType, IndexType > vi, vk;
   while( this->checkNextIteration() )
   {
      // adaptivity of the restarting parameter
      // reference:  A.H. Baker, E.R. Jessup, Tz.V. Kolev - A simple strategy for varying the restart parameter in GMRES(m)
      //             http://www.sciencedirect.com/science/article/pii/S0377042709000132
      if( restarting_max > restarting_min && restart_cycles > 0 ) {
         if( beta_ratio > max_beta_ratio )
            // near stagnation -> set maximum
            m = restarting_max;
         else if( beta_ratio >= min_beta_ratio ) {
            // the step size is determined based on current m using linear interpolation
            // between restarting_step_min and restarting_step_max
            const int step = restarting_step_min + (float) ( restarting_step_max - restarting_step_min ) /
                                                           ( restarting_max - restarting_min ) *
                                                           ( m - restarting_min );
            if( m - step >= restarting_min )
               m -= step;
            else
               // set restarting_max when we hit restarting_min (see Baker et al. (2009))
               m = restarting_max;
         }
//         std::cerr << "restarting: cycle = " << restart_cycles << ", beta_ratio = " << beta_ratio << ", m = " << m << "    " << std::endl;
      }

      for( IndexType i = 0; i < m + 1; i ++ )
         H[ i ] = s[ i ] = cs[ i ] = sn[ i ] = 0.0;

      /****
       * v = 0
       */
      _v.setValue( ( RealType ) 0.0 );

      /***
       * v_0 = r / | r | =  1.0 / beta * r
       */
      vi.bind( _v.getData(), size );
      vi.addVector( _r, ( RealType ) 1.0 / beta );

      _s.setValue( ( RealType ) 0.0 );
      _s[ 0 ] = beta;



      /****
       * Starting m-loop
       */
      for( IndexType i = 0; i < m && this->nextIteration(); i++ )
      {
         vi.bind( &( _v.getData()[ i * size ] ), size );
         /****
          * Solve w from M w = A v_i
          */
         if( preconditioner )
         {
            matrix->vectorProduct( vi, _M_tmp );
            this->preconditioner->solve( _M_tmp, w );
         }
         else
             matrix->vectorProduct( vi, w );
 
         //cout << " i = " << i << " vi = " << vi << std::endl;

         for( IndexType k = 0; k <= i; k++ )
            H[ k + i * ( m + 1 ) ] = 0.0;
         for( IndexType l = 0; l < 2; l++ )
            for( IndexType k = 0; k <= i; k++ )
            {
               vk.bind( &( _v.getData()[ k * _size ] ), _size );
               /***
                * H_{k,i} = ( w, v_k )
                */
               RealType H_k_i = w.scalarProduct( vk );
               H[ k + i * ( m + 1 ) ] += H_k_i;

               /****
                * w = w - H_{k,i} v_k
                */
               w.addVector( vk, -H_k_i );

               //cout << "H_ki = " << H_k_i << std::endl;
               //cout << "w = " << w << std::endl;
            }
         /***
          * H_{i+1,i} = |w|
          */
         RealType normw = w.lpNorm( ( RealType ) 2.0 );
         H[ i + 1 + i * ( m + 1 ) ] = normw;

         //cout << "normw = " << normw << std::endl;
 
         /***
          * v_{i+1} = w / |w|
          */
         vi.bind( &( _v.getData()[ ( i + 1 ) * size ] ), size );
         vi.addVector( w, ( RealType ) 1.0 / normw );
 
         //cout << "vi = " << vi << std::endl;
 
         /****
          * Applying the Givens rotations
          */
         for( IndexType k = 0; k < i; k++ )
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

         this->setResidue( std::fabs( s[ i + 1 ] ) / normb );
         if( ! this->checkNextIteration() ) {
            update( i, m, _H, _s, _v, x );
            this->refreshSolverMonitor( true );
            return this->checkConvergence();
         }
         else
         {
            this->refreshSolverMonitor();
         }
      }
      //cout << "x = " << x << std::endl;
      update( m - 1, m, _H, _s, _v, x );
      //cout << "x = " << x << std::endl;

      /****
       * r = M.solve(b - A * x);
       */
      const RealType beta_old = beta;
      beta = 0.0;
      if( preconditioner )
      {
         matrix -> vectorProduct( x, _M_tmp );
         _M_tmp.addVector( b, ( RealType ) 1.0, -1.0 );
         preconditioner -> solve( _M_tmp, _r );
         beta = _r.lpNorm( ( RealType ) 2.0 );
      }
      else
      {
         matrix -> vectorProduct( x, _r );
         _r.addVector( b, ( RealType ) 1.0, -1.0 );
         beta = _r.lpNorm( ( RealType ) 2.0 );
      }
      this->setResidue( beta / normb );

      //cout << " x = " << x << std::endl;
      //cout << " beta = " << beta << std::endl;
      //cout << "residue = " << beta / normb << std::endl;

      // update parameters for the adaptivity of the restarting parameter
      ++restart_cycles;
      beta_ratio = beta / beta_old;
   }
   this->refreshSolverMonitor( true );
   return this->checkConvergence();
}

template< typename Matrix,
          typename Preconditioner >
   template< typename VectorT >
void
GMRES< Matrix, Preconditioner >::
update( IndexType k,
        IndexType m,
        const Containers::Vector< RealType, Devices::Host, IndexType >& H,
        const Containers::Vector< RealType, Devices::Host, IndexType >& s,
        Containers::Vector< RealType, DeviceType, IndexType >& v,
        VectorT& x )
{
   Containers::Vector< RealType, Devices::Host, IndexType > y;
   y.setSize( m + 1 );

   IndexType i, j;
   for( i = 0; i <= m ; i ++ )
      y[ i ] = s[ i ];

   // Backsolve:
   for( i = k; i >= 0; i--)
   {
      //cout << " y = " << y << std::endl;
      y[ i ] /= H[ i + i * ( m + 1 ) ];
      for( j = i - 1; j >= 0; j--)
         y[ j ] -= H[ j + i * ( m + 1 ) ] * y[ i ];
   }

   Containers::Vector< RealType, DeviceType, IndexType > vi;
   for( i = 0; i <= k; i++)
   {
      vi.bind( &( v.getData()[ i * this->size ] ), x.getSize() );
      x.addVector( vi, y[ i ] );
   }
}

template< typename Matrix,
          typename Preconditioner >
void
GMRES< Matrix, Preconditioner >::
generatePlaneRotation( RealType& dx,
                       RealType& dy,
                       RealType& cs,
                       RealType& sn )
{
   if( dy == 0.0 )
   {
      cs = 1.0;
      sn = 0.0;
   }
   else
      if( std::fabs( dy ) > std::fabs( dx ) )
      {
         RealType temp = dx / dy;
         sn = 1.0 / std::sqrt( 1.0 + temp * temp );
         cs = temp * sn;
      }
      else
      {
         RealType temp = dy / dx;
         cs = 1.0 / std::sqrt( 1.0 + temp * temp );
         sn = temp * cs;
      }
}

template< typename Matrix,
          typename Preconditioner >
void
GMRES< Matrix, Preconditioner >::
applyPlaneRotation( RealType& dx,
                    RealType& dy,
                    RealType& cs,
                    RealType& sn )
{
   RealType temp  =  cs * dx + sn * dy;
   dy =  cs * dy - sn * dx;
   dx = temp;
}

template< typename Matrix,
          typename Preconditioner >
bool
GMRES< Matrix, Preconditioner >::
setSize( IndexType _size, IndexType m )
{
   if( size == _size && restarting_max == m ) return true;
   size = _size;
   restarting_max = m;
   if( ! _r.setSize( size ) ||
       ! w.setSize( size ) ||
       ! _s.setSize( m + 1 ) ||
       ! _cs.setSize( m + 1 ) ||
       ! _sn.setSize( m + 1 ) ||
       ! _v.setSize( size * ( m + 1 ) ) ||
       ! _H.setSize( ( m + 1 ) * m ) ||
       ! _M_tmp.setSize( size ) )
   {
      std::cerr << "I could not allocate all supporting arrays for the GMRES solver." << std::endl;
      return false;
   }
   return true;
}

} // namespace Linear
} // namespace Solvers
} // namespace TNL
