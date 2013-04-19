/***************************************************************************
                          navierStokesSolverMonitor_impl.h  -  description
                             -------------------
    begin                : Mar 13, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLNAVIERSTOKESSOLVERMONITOR_IMPL_H_
#define TNLNAVIERSTOKESSOLVERMONITOR_IMPL_H_

#include <fstream>

using namespace std;

template< typename Real, typename Index >
navierStokesSolverMonitor< Real, Index > :: navierStokesSolverMonitor()
: beta_max( 0.0 ),
  eps_s_max( 0.0 ),
  u_s_max( 0.0 ),
  u_g_max( 0.0 ),
  G_max( 0.0 ),
  eps_rho_g_max( 0.0 ),
  eps_rho_u1_g_max( 0.0 ),
  eps_rho_u2_g_max( 0.0 ),
  eps_rho_g_t_max( 0.0 ),
  eps_rho_u1_g_t_max( 0.0 ),
  eps_rho_u2_g_t_max( 0.0 ),
  eps_rho_s_max( 0.0 ),
  eps_rho_u1_s_max( 0.0 ),
  eps_rho_u2_s_max( 0.0 ),
  eps_rho_s_t_max( 0.0 ),
  eps_rho_u1_s_t_max( 0.0 ),
  eps_rho_u2_s_t_max( 0.0 )
{
   /****
    * Reset statistics files
    */
   fstream file;
   file. open( "tau.txt", ios :: out );
   file. close();
   file. open( "beta-max.txt", ios :: out );
   file. close();
   file. open( "eps-s-max.txt", ios :: out );
   file. close();
   file. open( "u-s-max.txt", ios :: out );
   file. close();
   file. open( "u-g-max.txt", ios :: out );
   file. close();
   file. open( "eps-rho-g-max.txt", ios :: out );
   file. close();
   file. open( "eps-rho-u1-g-max.txt", ios :: out );
   file. close();
   file. open( "eps-rho-u2-g-max.txt", ios :: out );
   file. close();
   file. open( "eps-rho-g-t-max.txt", ios :: out );
   file. close();
   file. open( "eps-rho-u1-g-t-max.txt", ios :: out );
   file. close();
   file. open( "eps-rho-u2-g-t-max.txt", ios :: out );
   file. close();
   file. open( "eps-rho-s-max.txt", ios :: out );
   file. close();
   file. open( "eps-rho-u1-s-max.txt", ios :: out );
   file. close();
   file. open( "eps-rho-u2-s-max.txt", ios :: out );
   file. close();
   file. open( "eps-rho-s-t-max.txt", ios :: out );
   file. close();
   file. open( "eps-rho-u1-s-t-max.txt", ios :: out );
   file. close();
   file. open( "eps-rho-u2-s-t-max.txt", ios :: out );
   file. close();
}

template< typename Real, typename Index >
void navierStokesSolverMonitor< Real, Index > :: refresh()
{
   /****
    * Write to statistics files
    */
   fstream file;
   file. open( "tau.txt", ios :: out | ios :: app );
   file << this -> getTime() << " " << log10( this -> getTimeStep()  ) << endl;
   file. close();
   file. open( "beta-max.txt", ios :: out | ios :: app );
   file << this -> getTime() << " " << log10( this -> beta_max ) << endl;
   file. close();
   file. open( "eps-s-max.txt", ios :: out | ios :: app );
   file << this -> getTime() << " " << this -> eps_s_max << endl;
   file. close();
   file. open( "u-s-max.txt", ios :: out | ios :: app );
   file << this -> getTime() << " " << this -> u_s_max << endl;
   file. close();
   file. open( "u-g-max.txt", ios :: out | ios :: app );
   file << this -> getTime() << " " << this -> u_g_max << endl;
   file. close();
   file. open( "eps-rho-g-max.txt", ios :: out | ios :: app );
   file << this -> getTime() << " " << this -> eps_rho_g_max << endl;
   file. close();
   file. open( "eps-rho-u1-g-max.txt", ios :: out | ios :: app );
   file << this -> getTime() << " " << this -> eps_rho_u1_g_max << endl;
   file. close();
   file. open( "eps-rho-u2-g-max.txt", ios :: out | ios :: app );
   file << this -> getTime() << " " << this -> eps_rho_u2_g_max << endl;
   file. close();
   file. open( "eps-rho-g-t-max.txt", ios :: out | ios :: app );
   file << this -> getTime() << " " << this -> eps_rho_g_t_max << endl;
   file. close();
   file. open( "eps-rho-u1-g-t-max.txt", ios :: out | ios :: app );
   file << this -> getTime() << " " << this -> eps_rho_u1_g_t_max << endl;
   file. close();
   file. open( "eps-rho-u2-g-t-max.txt", ios :: out | ios :: app );
   file << this -> getTime() << " " << this -> eps_rho_u2_g_t_max << endl;
   file. close();
   file. open( "eps-rho-s-max.txt", ios :: out | ios :: app );
   file << this -> getTime() << " " << this -> eps_rho_s_max << endl;
   file. close();
   file. open( "eps-rho-u1-s-max.txt", ios :: out | ios :: app );
   file << this -> getTime() << " " << this -> eps_rho_u1_s_max << endl;
   file. close();
   file. open( "eps-rho-u2-s-max.txt", ios :: out | ios :: app );
   file << this -> getTime() << " " << this -> eps_rho_u2_s_max << endl;
   file. close();
   file. open( "eps-rho-s-t-max.txt", ios :: out | ios :: app );
   file << this -> getTime() << " " << this -> eps_rho_s_t_max << endl;
   file. close();
   file. open( "eps-rho-u1-s-t-max.txt", ios :: out  | ios :: app);
   file << this -> getTime() << " " << this -> eps_rho_u1_s_t_max << endl;
   file. close();
   file. open( "eps-rho-u2-s-t-max.txt", ios :: out | ios :: app );
   file << this -> getTime() << " " << this -> eps_rho_u2_s_t_max << endl;
   file. close();



   if( this -> verbose > 0 && this -> refreshing % this -> outputPeriod == 0 )
   {
      /*cout << "u-max I: "<< setw( 3 ) << this -> u_s_max_i
           << " J: "<< setw( 3 ) << this -> u_s_max_j
           << " G_max: " << setw( 6 ) << this -> G_max;*/
      //cout << setprecision( 5 )
      //     << "betaMax: " << setw( 8 ) << this -> beta_max
      //     << " epsSMax: " << setw( 8 ) << this -> eps_s_max;
   }
   tnlODESolverMonitor< Real, Index > :: refresh();
}

template< typename Real, typename Index >
void navierStokesSolverMonitor< Real, Index > :: setBetaMax( const Real& beta_max )
{
   this -> beta_max = beta_max;
}

template< typename Real, typename Index >
void navierStokesSolverMonitor< Real, Index > :: setEpsSMax( const Real& eps_s_max )
{
   this -> eps_s_max = eps_s_max;
}

template< typename Real, typename Index >
void navierStokesSolverMonitor< Real, Index > :: setUSMax( const Real& u_s_max )
{
   this -> u_s_max = u_s_max;
}

template< typename Real, typename Index >
void navierStokesSolverMonitor< Real, Index > :: setUGMax( const Real& u_g_max )
{
   this -> u_g_max = u_g_max;
}

template< typename Real, typename Index >
void navierStokesSolverMonitor< Real, Index > :: setEps_rho_g_max( const Real& eps_rho_g_max )
{
   this -> eps_rho_g_max = eps_rho_g_max;
}

template< typename Real, typename Index >
void navierStokesSolverMonitor< Real, Index > :: setEps_rho_u1_g_max( const Real& eps_rho_u1_g_max )
{
   this -> eps_rho_u1_g_max = eps_rho_u1_g_max;
}

template< typename Real, typename Index >
void navierStokesSolverMonitor< Real, Index > :: setEps_rho_u2_g_max( const Real& eps_rho_u2_g_max )
{
   this -> eps_rho_u2_g_max = eps_rho_u2_g_max;
}

template< typename Real, typename Index >
void navierStokesSolverMonitor< Real, Index > :: setEps_rho_g_t_max( const Real& eps_rho_g_t_max )
{
   this -> eps_rho_g_t_max = eps_rho_g_t_max;
}

template< typename Real, typename Index >
void navierStokesSolverMonitor< Real, Index > :: setEps_rho_u1_g_t_max( const Real& eps_rho_u1_g_t_max )
{
   this -> eps_rho_u1_g_t_max = eps_rho_u1_g_t_max;
}

template< typename Real, typename Index >
void navierStokesSolverMonitor< Real, Index > :: setEps_rho_u2_g_t_max( const Real& eps_rho_u2_g_t_max )
{
   this -> eps_rho_u2_g_t_max = eps_rho_u2_g_t_max;
}

template< typename Real, typename Index >
void navierStokesSolverMonitor< Real, Index > :: setEps_rho_s_max( const Real& eps_rho_s_max )
{
   this -> eps_rho_s_max = eps_rho_s_max;
}

template< typename Real, typename Index >
void navierStokesSolverMonitor< Real, Index > :: setEps_rho_u1_s_max( const Real& eps_rho_u1_s_max )
{
   this -> eps_rho_u1_s_max = eps_rho_u1_s_max;
}

template< typename Real, typename Index >
void navierStokesSolverMonitor< Real, Index > :: setEps_rho_u2_s_max( const Real& eps_rho_u2_s_max )
{
   this -> eps_rho_u2_s_max = eps_rho_u2_s_max;
}

template< typename Real, typename Index >
void navierStokesSolverMonitor< Real, Index > :: setEps_rho_s_t_max( const Real& eps_rho_s_t_max )
{
   this -> eps_rho_s_t_max = eps_rho_s_t_max;
}

template< typename Real, typename Index >
void navierStokesSolverMonitor< Real, Index > :: setEps_rho_u1_s_t_max( const Real& eps_rho_u1_s_t_max )
{
   this -> eps_rho_u1_s_t_max = eps_rho_u1_s_t_max;
}

template< typename Real, typename Index >
void navierStokesSolverMonitor< Real, Index > :: setEps_rho_u2_s_t_max( const Real& eps_rho_u2_s_t_max )
{
   this -> eps_rho_u2_s_t_max = eps_rho_u2_s_t_max;
}




#endif /* TNLNAVIERSTOKESSOLVERMONITOR_IMPL_H_ */
