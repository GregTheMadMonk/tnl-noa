/***************************************************************************
                          navierStokesSolverMonitor.h  -  description
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

#ifndef NAVIERSTOKESSOLVERMONITOR_H_
#define NAVIERSTOKESSOLVERMONITOR_H_

#include <solvers/ode/tnlODESolverMonitor.h>

template< typename Real, typename Index >
class navierStokesSolverMonitor : public tnlODESolverMonitor< Real, Index >
{
   public:

   navierStokesSolverMonitor();

   void refresh();

   void setBetaMax( const Real& betaMax );

   void setEpsSMax( const Real& epsSMax );

   void setUSMax( const Real& u_s_max );

   void setUGMax( const Real& u_g_max );

   void setEps_rho_g_max( const Real& eps_rho_g_max );
   void setEps_rho_u1_g_max( const Real& eps_rho_u1_g_max );
   void setEps_rho_u2_g_max( const Real& eps_rho_u2_g_max );
   void setEps_rho_g_t_max( const Real& eps_rho_g_t_max );
   void setEps_rho_u1_g_t_max( const Real& eps_rho_u1_g_t_max );
   void setEps_rho_u2_g_t_max( const Real& eps_rho_u2_g_t_max );
   void setEps_rho_s_max( const Real& eps_rho_s_max );
   void setEps_rho_u1_s_max( const Real& eps_rho_u1_s_max );
   void setEps_rho_u2_s_max( const Real& eps_rho_u2_s_max );
   void setEps_rho_s_t_max( const Real& eps_rho_s_t_max );
   void setEps_rho_u1_s_t_max( const Real& eps_rho_u1_s_t_max );
   void setEps_rho_u2_s_t_max( const Real& eps_rho_u2_s_t_max );

   protected:

   public:

   Real beta_max, eps_s_max, u_s_max, u_g_max, G_max;
   Real eps_rho_g_max, eps_rho_u1_g_max, eps_rho_u2_g_max,
        eps_rho_g_t_max, eps_rho_u1_g_t_max, eps_rho_u2_g_t_max,
        eps_rho_s_max, eps_rho_u1_s_max, eps_rho_u2_s_max,
        eps_rho_s_t_max, eps_rho_u1_s_t_max, eps_rho_u2_s_t_max;

   public:
   Index u_s_max_i, u_s_max_j;
};

#include "navierStokesSolverMonitor_impl.h"

#endif /* NAVIERSTOKESSOLVERMONITOR_H_ */
